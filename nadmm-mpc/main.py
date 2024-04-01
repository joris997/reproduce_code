import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca
import casadi.tools as ca_tools
import time
from Robot import Robot, ReferenceTrajectory
from Plotter import Plotter
    
    
class CentralizedSolver:
    def __init__(self,robots,t_range):
        self.robots = robots
        self.t_range = t_range
        self.dT = self.t_range[-1] - self.t_range[0]
        self.dt = 0.1

        self.nx = 6
        self.nu = 2
        self.N = 10

        self.Q = ca.MX(np.diag([10,10,1,1,1,1]))
        self.R = ca.MX(np.diag([1,1]))
        self.P = ca.MX(np.diag([1,1,1,1,1,1]))

    def solve(self):
        print("Solving")
        x0 = [self.robots[i].x0 for i in range(len(self.robots))]

        for t in np.arange(self.t_range[0],self.t_range[-1]+1e-5,self.dt):
            opti = ca.Opti()

            x_vars = [[opti.variable(self.nx) for _ in range(self.N+1)] for _ in range(len(self.robots))]
            u_vars = [[opti.variable(self.nu) for _ in range(self.N)] for _ in range(len(self.robots))]

            con_eq = []
            con_ineq = []
            con_ineq_lb = []
            con_ineq_ub = []
            obj = ca.MX(0)

            # for each robot
            for i in range(len(self.robots)):
                # for each time-step in the future
                for j in range(self.N):
                    # Eq. 9b
                    con_eq.append(x_vars[i][j+1] - self.robots[i].dynamics(x_vars[i][j],u_vars[i][j],self.dt))
                    # # Eq. 9d
                    con_ineq.append(x_vars[i][j])
                    con_ineq_lb.append(self.robots[i].x_lb)
                    con_ineq_ub.append(self.robots[i].x_ub)
                    con_ineq.append(u_vars[i][j])
                    con_ineq_lb.append(self.robots[i].u_lb)
                    con_ineq_ub.append(self.robots[i].u_ub)

                    # Eq. 9a
                    x_ref = self.robots[i].get_x_ref(t + j*self.dt)
                    obj += (x_vars[i][j]-x_ref).T@self.Q@(x_vars[i][j]-x_ref)
                    obj += u_vars[i][j].T@self.R@u_vars[i][j]

                    # Set initial condition equal to reference state
                    if len(self.robots[i].x_vars_hist) > 0:
                        opti.set_initial(x_vars[i][j],self.robots[i].x_vars_hist[-1][j])

                    for i_2 in range(len(self.robots)):
                        if i_2 != i:
                            # Eq. 9e
                            x_diff = x_vars[i][j][0:2] - x_vars[i_2][j][0:2]
                            eq4_matrix = ca.MX(np.array([[1/(self.robots[i].R + self.robots[i_2].R)**2, 0],
                                                        [0, 1/(self.robots[i].R + self.robots[i_2].R)**2]]))
                            con_ineq.append(x_diff.T@eq4_matrix@x_diff)
                            con_ineq_lb.append(1)
                            con_ineq_ub.append(ca.inf)


                # Eq. 9a
                x_ref = self.robots[i].get_x_ref(t + self.N*self.dt)
                obj += (x_vars[i][self.N]-x_ref).T@self.P@(x_vars[i][self.N]-x_ref)

                # Eq. 9c
                con_eq.append(x_vars[i][0] - x0[i])

            # Solve
            opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.tol':1e-3}
            opti.solver('ipopt',opts)

            # asign variables, cost, and constraints to opti
            opti.minimize(obj)
            for i in range(len(con_eq)):
                opti.subject_to(con_eq[i] == 0)
            for i in range(len(con_ineq)):
                opti.subject_to(opti.bounded(con_ineq_lb[i],con_ineq[i],con_ineq_ub[i]))
            
            solve_time = -time.time()
            try:
                sol = opti.solve()
            except RuntimeError as e:
                # Assuming x_vars and u_vars are your decision variables
                for i, x_var in enumerate(x_vars[0]):
                    x_ref = self.robots[0].get_x_ref(t + i*self.dt)
                    print(f"x[{i}] reference value:", x_ref)
                    try:
                        print(f"x[{i}] latest value:", opti.debug.value(x_var))
                    except Exception as e:
                        print("Error retrieving latest value for x[{}]: {}".format(i, e))
                for i, u_var in enumerate(u_vars[0]):
                    try:
                        print(f"u[{i}] latest value:", opti.debug.value(u_var))
                    except Exception as e:
                        print("Error retrieving latest value for u[{}]: {}".format(i, e))
            solve_time += time.time()
            print("-------------------")
            print("Time:       ", t)
            print("Status:     ", sol.stats()['return_status'])
            print("x[0]:  ", np.round(sol.value(x_vars[0][0]),2))
            print("x[-1]: ", np.round(sol.value(x_vars[0][-1]),2))
            print("u: ", np.round(sol.value(u_vars[0][0]),2))

            # save results
            for i in range(len(self.robots)):
                x_vars_i = [sol.value(x_vars[i][j]) for j in range(self.N+1)]
                u_vars_i = [sol.value(u_vars[i][j]) for j in range(self.N)]
                self.robots[i].x_vars_hist.append(x_vars_i)
                self.robots[i].u_vars_hist.append(u_vars_i)

            x0 = [sol.value(x_vars[i][1]) for i in range(len(self.robots))]


class DecentralizedSolver(CentralizedSolver):
    def __init__(self,robots,t_range):
        super().__init__(robots,t_range)

    def solve(self):
        print("Solving")
        max_iter = 100
        x0 = [self.robots[i].x0 for i in range(len(self.robots))]

        # populate x_vars_hist and u_vars_hist with the initial values
        for i in range(len(self.robots)):
            self.robots[i].x_vars_hist.append([self.robots[i].x0 for _ in range(self.N+1)])
            self.robots[i].u_vars_hist.append([np.zeros(self.robots[i].nu) for _ in range(self.N)])

        # Decentralized NADMM for MPC problems
        self.set_initial_values()

        for t in np.arange(self.t_range[0],self.t_range[-1]+1e-5,self.dt):
            for _ in range(max_iter):
                for i in range(len(self.robots)):
                    # update b_i
                    for j in range(self.N):
                        A_ixsi_i = (self.robots[i].A_i@self.robots[i].xsi[j]).flatten()
                        B_iy_i = self.robots[i].B_i@self.robots[i].y[j]

                        self.robots[i].b_i[j] = A_ixsi_i + B_iy_i

                    # \hat\lambda_i^s <- \lambda_i^s - \rho(1-\beta)*(A_i \xsi_i^s + B_i y_i^s - b_i)
                    for j in range(self.N):
                        A_ixsi_i = (self.robots[i].A_i@self.robots[i].xsi[j]).flatten()
                        B_iy_i = self.robots[i].B_i@self.robots[i].y[j]
                        b_i = self.robots[i].b_i[j]

                        self.robots[i].lambda_[j] -= self.robots[i].rho*(1-self.robots[i].beta)*(A_ixsi_i + B_iy_i - b_i)
                    
                    # \xsi_i^{s+1} <- \argmin_{\xsi_i} L_i(\xsi_i,y_i^s,\lambda_i^s)
                    xsi,y = self.solve_local_problem(t,i)

                    # \lambda_i^{s+1} <- \hat\lambda_^s + \rho(A_i\xsi_i^{s+1} + B_i y_i^s - b_i)
                    for j in range(self.N):
                        A_ixsi_i = self.robots[i].A_i@xsi[j]
                        B_iy_i = self.robots[i].B_i@y[j]
                        b_i = self.robots[i].b_i[j]

                        self.robots[i].lambda_[j] += self.robots[i].rho*(A_ixsi_i + B_iy_i - b_i)
                    
                    # Robot i sends/receives updates to/from neighbors
                    # return 0
                
                for i in range(len(self.robots)):
                    # update y_i^{s+1} (Eq. 15)
                    # \Delta p_{ij} = ((p_i - p_{j|i}) + (p_j - p_{i|j}))/2
                    # \eta_{j|i} = \eta_j
                    for j in range(self.N):
                        yj_ij = np.array([])
                        for i_2 in range(len(self.robots)):
                            if i != i_2:
                                Delta_ij = ((self.robots[i].xsi[j][0:2] - self.robots[i_2].b_i[j][i*2:i*2+2]) +
                                            (self.robots[i].b_i[j][i_2*2:i_2*2+2] - self.robots[i_2].xsi[j][0:2]))/2
                                yj_ij = np.hstack((yj_ij, Delta_ij))    
                        self.robots[i].y[j] = yj_ij    

                    # Robot i sends/receives updates to/from neighbors
                    # return 0
                
            for i in range(len(self.robots)):
                # Select u_i(1) and implement it
                x_vars_i = [self.robots[i].xsi[j][0:self.nx] for j in range(self.N)]
                u_vars_i = [self.robots[i].xsi[j][self.nx:] for j in range(self.N)]
                self.robots[i].x_vars_hist.append(x_vars_i)
                self.robots[i].u_vars_hist.append(u_vars_i)

            # Update \xsi_i^0 
            x0 = [self.robots[i].xsi[0][0:self.nx] for i in range(len(self.robots))] 
            for i in range(len(self.robots)):
                self.robots[i].x0 = x0[i]   

    
    def set_initial_values(self):
        # - Each robot initializes \xsi_i^0, y_i^0, \lambda_i^0, \rho, \beta
        for i in range(len(self.robots)):
            self.robots[i].xsi = [np.zeros(self.robots[i].nx + self.robots[i].nu) for _ in range(self.N)]
            self.robots[i].y = []
            for j in range(self.N):
                yj_ij = np.array([])
                yj_ji = np.array([])
                for i_2 in range(len(self.robots)):
                    if i != i_2:
                        # append Deltap_ij to yj_ij and Deltap_ji to yj_ji
                        Deltap = np.array(self.robots[i].x0[0:2] - self.robots[i_2].x0[0:2])
                        yj_ij = np.hstack((yj_ij, Deltap.reshape(1,2).flatten()))
                        # yj_ji = np.hstack((yj_ji, -Deltap.reshape(1,2).flatten()))
                yj = np.hstack((yj_ij,yj_ji))
                self.robots[i].y.append(yj)

            self.robots[i].lambda_ = [np.zeros(2*2) for _ in range(self.N)]
            self.robots[i].rho = 2
            self.robots[i].beta = 1
            # also initialize b_i ([p_{j|i}^T p_{i|j}^T]) with the ground truth of x0
            self.robots[i].b_i = []
            for j in range(self.N):
                # bj_ij = np.zeros((2*len(self.robots)))
                # bj_ji = np.zeros((2*len(self.robots)))
                bj_ij = np.array([])
                bj_ji = np.array([])
                for i_2 in range(len(self.robots)):
                    if i != i_2:
                        # bj_ij[2*i_2:2*i_2+2] = self.robots[i_2].x0[0:2]
                        # bj_ji[2*i_2:2*i_2+2] = self.robots[i].x0[0:2]
                        bj_ij = np.hstack((bj_ij,self.robots[i_2].x0[0:2]))
                        bj_ji = np.hstack((bj_ji,self.robots[i].x0[0:2]))
                bj = np.hstack((bj_ij,bj_ji))
                self.robots[i].b_i.append(bj)

            # also create A_i and B_i
            E_i = np.array([[1,0,0,0,0,0, 0,0],[0,1,0,0,0,0, 0,0]])
            F = E_i
            self.robots[i].A_i = np.concatenate((E_i,F),axis=0)
            # B = [-I, -I, I, I] where the number of -I and I is equal to len(self.robots)-1
            self.robots[i].B_i = np.hstack([np.tile(-np.eye(2), (1, len(self.robots)-1)), np.tile(np.eye(2), (1, len(self.robots)-1))]).T


    def solve_local_problem(self,t,i):
        # Solve the local optimization problem in Eq. (13)
        opti = ca.Opti()

        x_vars = [opti.variable(self.nx) for _ in range(self.N+1)]
        u_vars = [opti.variable(self.nu) for _ in range(self.N)]

        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        obj = ca.MX(0)

        # Eq. 9b
        for j in range(self.N):
            con_eq.append(x_vars[j+1] - self.robots[i].dynamics(x_vars[j],u_vars[j],self.dt))
        # Eq. 9c
        con_eq.append(x_vars[0] - self.robots[i].x0)
        # Eq. 9d
        for j in range(self.N):
            con_ineq.append(x_vars[j])
            con_ineq_lb.append(self.robots[i].x_lb)
            con_ineq_ub.append(self.robots[i].x_ub)
            con_ineq.append(u_vars[j])
            con_ineq_lb.append(self.robots[i].u_lb)
            con_ineq_ub.append(self.robots[i].u_ub)

        # Eq. 10f
        # p_i^h(t+k) = R^h(z_i(t+k))z_i(t+k)
        # We can skip this because we only have one circle, which lays at the center of the robot
            
        # Eq. 10g
        # c_{(i,j)|i}(t+k) > 0, j\neq i
        for j in range(self.N):
            cnt = 0
            for i_2 in range(len(self.robots)):
                if i_2 != i:
                    # TODO: fix this belief part
                    x_diff = x_vars[j][0:2] - self.robots[i].b_i[j][2*cnt:2*cnt+2]
                    eq4_matrix = ca.MX(np.array([[1/(self.robots[i].R + self.robots[i_2].R)**2, 0],
                                                 [0, 1/(self.robots[i].R + self.robots[i_2].R)**2]]))
                    con_ineq.append(x_diff.T@eq4_matrix@x_diff)
                    con_ineq_lb.append(1)
                    con_ineq_ub.append(ca.inf)
                    cnt += 1
        
        # Eq. 10h
        # c_{(j,k)|i}(t+k) > 0, j\neq i
        # Do we really need this? Seems trivial w.r.t. Eq. 10g
                    
        # Lagrangian
        for j in range(self.N):
            x_ref = self.robots[i].get_x_ref(t + j*self.dt)
            obj += (x_vars[j]-x_ref).T@self.Q@(x_vars[j]-x_ref)
            obj += u_vars[j].T@self.R@u_vars[j]
        
        # for all combinations (i,j), (i,k), (j,k), dimension of (i,j) itself is 2, put it into a vector
        Deltap_vars = [opti.variable(2*(len(self.robots)-1)) for _ in range(self.N)]
        for j in range(self.N):
            # We only take the first 6 columns of A_i, because those are for the states, others are for control
            obj += ca.dot(ca.MX(self.robots[i].lambda_[j]),
                          self.robots[i].A_i[:,0:6]@x_vars[j] + self.robots[i].B_i@Deltap_vars[j] - ca.MX(self.robots[i].b_i[j]))
        
        for j in range(self.N):
            obj += (self.robots[i].rho / 2) * ca.norm_2(
                self.robots[i].A_i[:,0:6]@x_vars[j] + self.robots[i].B_i@Deltap_vars[j] - ca.MX(self.robots[i].b_i[j]))**2

        # Solve
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.tol':1e-3}
        opti.solver('ipopt',opts)

        # asign variables, cost, and constraints to opti
        opti.minimize(obj)
        for i in range(len(con_eq)):
            opti.subject_to(con_eq[i] == 0)
        for i in range(len(con_ineq)):
            opti.subject_to(opti.bounded(con_ineq_lb[i],con_ineq[i],con_ineq_ub[i]))
        
        solve_time = -time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            # Assuming x_vars and u_vars are your decision variables
            for i, x_var in enumerate(x_vars):
                x_ref = self.robots[0].get_x_ref(t + i*self.dt)
                print(f"x[{i}] reference value:", x_ref)
                try:
                    print(f"x[{i}] latest value:", opti.debug.value(x_var))
                except Exception as e:
                    print("Error retrieving latest value for x[{}]: {}".format(i, e))
            for i, u_var in enumerate(u_vars):
                try:
                    print(f"u[{i}] latest value:", opti.debug.value(u_var))
                except Exception as e:
                    print("Error retrieving latest value for u[{}]: {}".format(i, e))
        solve_time += time.time()
        print("-------------------")
        print("Time:       ", t)
        print("Status:     ", sol.stats()['return_status'])
        print("x[0]:  ", np.round(sol.value(x_vars[0]),2))
        print("x[-1]: ", np.round(sol.value(x_vars[-1]),2))
        print("u: ", np.round(sol.value(u_vars[0]),2))

        # Create xsi which is an array (for each j in N) of the stacked x and u
        xsi = np.vstack([np.concatenate((sol.value(x_vars[j]), sol.value(u_vars[j]))) for j in range(self.N)])
        
        # Create y which is an array (for each j in N) of \Deltap
        y = np.vstack([sol.value(Deltap_vars[j]) for j in range(self.N)])
        
        return xsi, y



if __name__ == "__main__":
    t_range = np.array([0,10])

    x_ref_r1 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[-20,20,-np.deg2rad(45),0,0,0],
                                                                      [20,-20,-np.deg2rad(45),0,0,0],]))
    r1 = Robot(x0=np.array([-20,20,-np.deg2rad(45),0,0,0]),x_ref=x_ref_r1,id=1)

    x_ref_r2 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[20,20,-np.deg2rad(135),0,0,0],
                                                                      [-20,-20,-np.deg2rad(135),0,0,0],]))
    r2 = Robot(x0=np.array([20,20,-np.deg2rad(135),0,0,0]),x_ref=x_ref_r2,id=2)

    x_ref_r3 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[20,-20,np.deg2rad(135),0,0,0],
                                                                      [-20,20,np.deg2rad(135),0,0,0],]))
    r3 = Robot(x0=np.array([20,-20,np.deg2rad(135),0,0,0]),x_ref=x_ref_r3,id=3)

    x_ref_r4 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[-20,-20,np.deg2rad(45),0,0,0],
                                                                      [20,20,np.deg2rad(45),0,0,0],]))
    r4 = Robot(x0=np.array([-20,-20,np.deg2rad(45),0,0,0]),x_ref=x_ref_r4,id=4)

    robots = [r1,r2]
    # solver = CentralizedSolver(robots,t_range)
    solver = DecentralizedSolver(robots,t_range)
    solver.solve()

    plotter = Plotter(solver)
    plotter.plot_results()
    plotter.plot_animation()
               
