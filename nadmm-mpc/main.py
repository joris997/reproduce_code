import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca
import casadi.tools as ca_tools
import time
from Robot import Robot, ReferenceTrajectory
from Plotter import Plotter
from helpers import stack_dict, smooth_min, smooth_max
    
def stack_dict(dict):
    # stack all dictionary items in increasing order of key
    # also consider they might not exist
    stack = np.array([])
    for key in sorted(dict.keys()):
        stack = np.hstack((stack, dict[key]))
    return stack
    
class CentralizedSolver:
    def __init__(self,robots,t_range):
        self.robots = robots
        self.t_range = t_range
        self.dT = self.t_range[-1] - self.t_range[0]
        self.dt = 0.1

        self.nx = self.robots[0].nx
        self.nu = self.robots[0].nu
        self.N = 10

        # cost matrices, same for all robots
        self.Q = ca.MX(np.diag([1,1,0.1,0.1,0.1,0.1]))
        self.R = ca.MX(np.diag([0.1,0.1]))
        self.P = ca.MX(np.diag([0.1,0.1,0.1,0.1,0.1,0.1]))

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
                    print(f"x[{i}] reference value: ", x_ref)
                    try:
                        print(f"x[{i}] latest value: ", opti.debug.value(x_var))
                    except Exception as e:
                        print("Error retrieving latest value for x[{}]: {}".format(i, e))
                for i, u_var in enumerate(u_vars[0]):
                    try:
                        print(f"u[{i}] latest value: ", opti.debug.value(u_var))
                    except Exception as e:
                        print("Error retrieving latest value for u[{}]: {}".format(i, e))
            solve_time += time.time()
            print("-------------------")
            print("Time:   ", t)
            print("Status: ", sol.stats()['return_status'])
            print("x[0]:   ", np.round(sol.value(x_vars[0][0]),2))
            print("x[-1]:  ", np.round(sol.value(x_vars[0][-1]),2))
            print("u:      ", np.round(sol.value(u_vars[0][0]),2))

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
        self.N = 10

        self.Q = ca.MX(np.diag([1,1,0.1,0.1,0.1,0.1]))
        self.R = ca.MX(np.diag([0.1,0.1]))
        self.P = ca.MX(np.diag([0.1,0.1,0.1,0.1,0.1,0.1]))

    def solve(self):
        print("Solving")
        max_iter = 2
        x0 = [self.robots[i].x0 for i in range(len(self.robots))]

        # Decentralized NADMM for MPC problems
        self.set_initial_values()

        for t in np.arange(self.t_range[0],self.t_range[-1]+1e-5,self.dt):
            for it in range(max_iter):
                for i in range(len(self.robots)):
                    print("------- t: ", t, " Iteration: ",it," Robot: ",i," -------")
                    # update b_i
                    # first creat y by looping through the .y[j] dictionary and stacking them in increasing order of key
                    for j in range(self.N):
                        A_ixsi_i = self.robots[i].A_i[0:2,:]@self.robots[i].xsi[j]
                        B_iy_i = 0
                        if j == 0:
                            print("Robot ",i, " b_i was: ", self.robots[i].b_i[j])
                            
                        for i_2 in range(len(self.robots)):
                            if i != i_2:
                                try:
                                    # is this correct? now we're directly communicating p_j to p_i
                                    # TODO: reach out  to Prof. Ferranti?
                                    self.robots[i].b_i[j][i_2] = self.robots[i].b_others[j][i_2]
                                except:
                                    B_iy_i = self.robots[i].B_i[0:2,0:2]@self.robots[i].y[j][i_2]
                                    self.robots[i].b_i[j][i_2] = A_ixsi_i + B_iy_i

                        if j == 0:
                            print("Robot ",i, " b_i new: ", self.robots[i].b_i[j])
                            print("taking the sum: ", A_ixsi_i, " + ", B_iy_i)

                    # \hat\lambda_i^s <- \lambda_i^s - \rho(1-\beta)*(A_i \xsi_i^s + B_i y_i^s - b_i)
                    for j in range(self.N):
                        A_ixsi_i = self.robots[i].A_i@self.robots[i].xsi[j]
                        B_iy_i = self.robots[i].B_i@stack_dict(self.robots[i].y[j])
                        b_i = stack_dict(self.robots[i].b_i[j])

                        self.robots[i].lambda_[j] -= self.robots[i].rho*(1-self.robots[i].beta)*(A_ixsi_i + B_iy_i - b_i)
                    
                    # \xsi_i^{s+1} <- \argmin_{\xsi_i} L_i(\xsi_i,y_i^s,\lambda_i^s)
                    self.robots[i].xsi,self.robots[i].y = self.solve_local_problem(t,i)
                    print("y from opt: ", self.robots[i].y[0])

                    # \lambda_i^{s+1} <- \hat\lambda_^s + \rho(A_i\xsi_i^{s+1} + B_i y_i^s - b_i)
                    for j in range(self.N):
                        A_ixsi_i = self.robots[i].A_i@self.robots[i].xsi[j]
                        B_iy_i = self.robots[i].B_i@stack_dict(self.robots[i].y[j])
                        b_i = stack_dict(self.robots[i].b_i[j])

                        self.robots[i].lambda_[j] += self.robots[i].rho*(A_ixsi_i + B_iy_i - b_i)

                    # Robot i sends/receives updates to/from neighbors
                    # so robot i sends its y = b_{i_2|i} - p_i to robot i_2
                    for j in range(self.N):
                        for i_2 in range(len(self.robots)):
                            if i != i_2:
                                for j in range(self.N):
                                    # \Delta p_{ij} = p_{j} - p_{i|j}
                                    self.robots[i_2].y_others[j][i] = self.robots[i].y[j][i_2]
                                    self.robots[i_2].b_others[j][i] = self.robots[i].xsi[j][0:2]
                                    # self.robots[i_2].y_others[j][i] = self.robots[i].xsi[j][0:2] - self.robots[i].b_i[j][i_2]
                
                for i in range(len(self.robots)):
                    # update y_i^{s+1} (Eq. 15)
                    # Robot i sends/receives updates to/from neighbors
                    # \Delta p_{ij} = ((p_i - p_{j|i}) + (p_j - p_{i|j}))/2
                    # \eta_{j|i} = \eta_jm/
                    for j in range(self.N):
                        yj_ij = {}
                        term1 = 0
                        term2 = 0
                        for i_2 in range(len(self.robots)):
                            if i != i_2:
                                # yj_ij[i_2] = ((self.robots[i].xsi[j][0:2] - self.robots[i].b_i[j][i_2]) + 
                                #               (self.robots[i_2].b_i[j][i] - self.robots[i_2].xsi[j][0:2]))/2
                                term1 = (self.robots[i].xsi[j][0:2] - self.robots[i].b_i[j][i_2])
                                term2 = (self.robots[i].y_others[j][i_2])
                                yj_ij[i_2] = (term1 - term2)/2
                                print("we think Robot ", i_2, " is at: ", self.robots[i].b_i[j][i_2])
                        if j == 0:
                            print("Robot ",i, " y was:     ", self.robots[i].y[j])
                            print("Robot ",i, " y becomes: ", yj_ij)
                            print("taking the sum: ", term1, " - ", term2, " / 2")
                            
                        self.robots[i].y[j] = yj_ij

            print("\n======== taking step =========")
            for i in range(len(self.robots)):
                # Select u_i(1) and implement it
                x_vars_i = [self.robots[i].x0] + [self.robots[i].xsi[j][0:self.nx] for j in range(self.N)]
                u_vars_i = [self.robots[i].xsi[j][self.nx:] for j in range(self.N)]
                self.robots[i].x_vars_hist.append(x_vars_i)
                self.robots[i].u_vars_hist.append(u_vars_i)

                # save local variables that change in loop: b, y, lambda
                self.robots[i].b_hist.append(stack_dict(self.robots[i].b_i[0]))
                self.robots[i].y_hist.append(stack_dict(self.robots[i].y[0]))
                self.robots[i].lambda_hist.append(self.robots[i].lambda_[0])

            # Update \xsi_i^0 
            x0 = [self.robots[i].xsi[0][0:self.nx] for i in range(len(self.robots))] 
            for i in range(len(self.robots)):
                self.robots[i].x0 = x0[i]  
                self.robots[i].xsi = self.robots[i].xsi[1:] + [self.robots[i].xsi[-1]]
                print("Robot ",i, " x0: ", self.robots[i].x0)


    def print_robot_params(self,i):
        # print some ADMM information of the agent, round to two digits
        print("Robot ",i)
        print("xsi:    ",np.round(self.robots[i].xsi[0],2))
        print("y:      ",self.robots[i].y[0])
        print("lambda: ",np.round(self.robots[i].lambda_[0]))
        print("b_i:    ",self.robots[i].b_i[0])
        print("")


    def set_initial_values(self):
        # - Each robot initializes \xsi_i^0, y_i^0, \lambda_i^0, \rho, \beta
        for i in range(len(self.robots)):
            self.robots[i].xsi = []
            for j in range(self.N):
                self.robots[i].xsi.append(np.concatenate((self.robots[i].x0, np.zeros(self.robots[i].nu))))

            self.robots[i].y = []
            for j in range(self.N):
                yj_ij = {}
                for i_2 in range(len(self.robots)):
                    if i != i_2:
                        yj_ij[i_2] = self.robots[i].x0[0:2] - self.robots[i_2].x0[0:2]
                self.robots[i].y.append(yj_ij)

            self.robots[i].lambda_ = [np.zeros(2*(len(self.robots)-1)) for _ in range(self.N)]
            self.robots[i].rho = 0.01
            self.robots[i].beta = 0.5

            # also initialize b_i ([p_{j|i}^T p_{i|j}^T]) with the ground truth of x0
            self.robots[i].b_i = []
            for j in range(self.N):
                bj_ij = {}
                for i_2 in range(len(self.robots)):
                    if i != i_2:
                        bj_ij[i_2] = self.robots[i_2].x0[0:2]
                self.robots[i].b_i.append(bj_ij)

            # also create A_i and B_i
            F = np.eye(2, self.robots[i].nx + self.robots[i].nu)
            # repeat this vertically N times
            self.robots[i].A_i = np.tile(F, (len(self.robots)-1, 1))
            # B = [-I, -I, I, I] where the number of -I and I is equal to len(self.robots)-1
            I = -np.eye(2)  # Repeat mI N times
            # repeat put this as block-diagonal N times
            self.robots[i].B_i = np.kron(np.eye(len(self.robots)-1), I)

            self.print_robot_params(i)

            # populate x_vars_hist and u_vars_hist with the initial values
            self.robots[i].x_vars_hist.append([self.robots[i].x0 for _ in range(self.N+1)])
            self.robots[i].u_vars_hist.append([np.zeros(self.robots[i].nu) for _ in range(self.N)])

            # create local information of other robots trajectory
            # this should be an array (size N) of empty dictionaries
            self.robots[i].y_others = [{} for _ in range(self.N)]
            self.robots[i].b_others = [{} for _ in range(self.N)]


    def set_initial_guess(self,opti,x_vars,u_vars,y_vars,i,t):
        # # Set initial condition equal to reference state
        # if len(self.robots[i].x_vars_hist) > 0:
        #     for j in range(self.N):
        #         # get some random noise
        #         # r = np.random.normal(0, 0.1, x_vars[j].shape[0])
        #         opti.set_initial(x_vars[j],self.robots[i].x_vars_hist[-1][j+1])
        #     # r = np.random.normal(0, 0.1, x_vars[-1].shape[0])
        #     opti.set_initial(x_vars[-1],self.robots[i].x_vars_hist[-1][-1])

        # if len(self.robots[i].u_vars_hist) > 0:
        #     for j in range(self.N-1):
        #         opti.set_initial(u_vars[j],self.robots[i].u_vars_hist[-1][j+1])
        #     opti.set_initial(u_vars[-1],self.robots[i].u_vars_hist[-1][-1])
        
        for j in range(self.N):
            # initial condition
            cnt = 0
            for i_2 in range(len(self.robots)):
                if i != i_2:
                    # print("setting init condition: ", self.robots[i].y[j][i_2])
                    opti.set_initial(y_vars[j][2*cnt:2*cnt+2],self.robots[i].y[j][i_2])
                    cnt += 1


    def solve_local_problem(self,t,i):
        # Solve the local optimization problem in Eq. (13)
        opti = ca.Opti()

        x_vars = [opti.variable(self.nx) for _ in range(self.N+1)]
        u_vars = [opti.variable(self.nu) for _ in range(self.N)]
        xsi_vars = [ca.vertcat(x_vars[j+1], u_vars[j]) for j in range(self.N)]
        y_vars = [opti.variable(2*(len(self.robots)-1)) for _ in range(self.N)]

        # and some helper variables, to keep track of time
        t_vars = [t + j*self.dt for j in range(self.N+1)]

        # Set initial condition equal to reference state
        self.set_initial_guess(opti, x_vars, u_vars, y_vars, i, t)

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
        for j in range(self.N+1):
            con_ineq.append(x_vars[j])
            con_ineq_lb.append(self.robots[i].x_lb)
            con_ineq_ub.append(self.robots[i].x_ub)
        for j in range(self.N):
            con_ineq.append(u_vars[j])
            con_ineq_lb.append(self.robots[i].u_lb)
            con_ineq_ub.append(self.robots[i].u_ub)

        # Eq. 10f
        # p_i^h(t+k) = R^h(z_i(t+k))z_i(t+k)
        # We can skip this because we only have one circle, which lays at the center of the robot
            
        # Eq. 10g
        # c_{(i,j)|i}(t+k) > 0, j\neq i
        for j in range(self.N):
            for i_2 in range(len(self.robots)):
                if i_2 != i:
                    x_diff = x_vars[j+1][0:2] - self.robots[i].b_i[j][i_2]
                    eq4_matrix = ca.MX(np.array([[1/(self.robots[i].R + self.robots[i_2].R)**2, 0],
                                                 [0, 1/(self.robots[i].R + self.robots[i_2].R)**2]]))
                    con_ineq.append(x_diff.T@eq4_matrix@x_diff)
                    con_ineq_lb.append(1)
                    con_ineq_ub.append(ca.inf)
        
        # Eq. 10h
        # c_{(j,k)|i}(t+k) > 0, j\neq i
        # Do we really need this? Seems trivial w.r.t. Eq. 10g
                    
        # # Eq. 13c
        # # A_i \xsi_i^s + B_i y_i^s = b_i
        # # We don't actually need this, the Lagrangian already implicitly contains this
        # for j in range(self.N):
        #     con_eq.append(self.robots[i].A_i@xsi_vars[j] + self.robots[i].B_i@y_vars[j] == ca.MX(stack_dict(self.robots[i].b_i[j])))            
        
        # Lagrangian
        for j in range(self.N+1):
            x_ref = self.robots[i].get_x_ref(t + j*self.dt)
            obj += (x_vars[j]-x_ref).T@self.Q@(x_vars[j]-x_ref)
        for j in range(self.N):
            obj += u_vars[j].T@self.R@u_vars[j]

        # <\lambda_i^s, A_i \xsi_i^s + B_i y_i^s - b_i>
        for j in range(self.N):
            obj += ca.dot(ca.MX(self.robots[i].lambda_[j]),
                self.robots[i].A_i@xsi_vars[j] + self.robots[i].B_i@y_vars[j] - ca.MX(stack_dict(self.robots[i].b_i[j])))
        
        # \rho/2 ||A_i \xsi_i^s + B_i y_i^s - b_i||^2
        for j in range(self.N):
            obj += (self.robots[i].rho / 2) * ca.sumsqr(
                self.robots[i].A_i@xsi_vars[j] + self.robots[i].B_i@y_vars[j] - ca.MX(stack_dict(self.robots[i].b_i[j])))

        # ###########
        # ### STL ###
        # ###########
        # if hasattr(self.robots[i],"p"):
        #     if any(self.robots[i].I[0] <= t_vars[j] <= self.robots[i].I[1] for j in range(self.N+1)):
        #         mu = self.robots[i].get_p(x_vars,t_vars)
        #         con_ineq.append(mu)
        #         con_ineq_lb.append(0)
        #         con_ineq_ub.append(ca.inf)

        #         obj += 10*mu

        # Solve
        opts = {'ipopt.print_level':5, 'print_time':0, 'ipopt.tol':1e-3}
        opti.solver('ipopt',opts)

        # asign variables, cost, and constraints to opti
        opti.minimize(obj)
        for k in range(len(con_eq)):
            opti.subject_to(con_eq[k] == 0)
        for k in range(len(con_ineq)):
            opti.subject_to(opti.bounded(con_ineq_lb[k],con_ineq[k],con_ineq_ub[k]))
        
        solve_time = -time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            a=1
            # opti.debug.show_infeasibilities()
            # # Assuming x_vars and u_vars are your decision variables
            # for j, x_var in enumerate(x_vars):
            #     x_ref = self.robots[0].get_x_ref(t + i*self.dt)
            #     print(f"x[{i}] reference value:", x_ref)
            #     try:
            #         print(f"x[{i}] latest value:", opti.debug.value(x_var))
            #     except Exception as e:
            #         print("Error retrieving latest value for x[{}]: {}".format(i, e))
            # for j, u_var in enumerate(u_vars):
            #     try:
            #         print(f"u[{i}] latest value:", opti.debug.value(u_var))
            #     except Exception as e:
            #         print("Error retrieving latest value for u[{}]: {}".format(i, e))

        solve_time += time.time()
        print("Time:   ", t)
        print("Status: ", sol.stats()['return_status'])
        print("x[0]:   ", np.round(sol.value(x_vars[0]),2))
        print("x[-1]:  ", np.round(sol.value(x_vars[-1]),2))
        print("u:      ", np.round(sol.value(u_vars[0]),2))
        print("Lag[0]: ", np.round(self.robots[i].A_i@sol.value(xsi_vars[0]) + self.robots[i].B_i@sol.value(y_vars[0]) - stack_dict(self.robots[i].b_i[0]),2))
        try:
            print("mu:     ", np.round(sol.value(mu),2))
        except:
            pass
        # Create xsi which is an array (for each j in N) of the stacked x and u
        xsi = [sol.value(xsi_vars[j]) for j in range(self.N)]
        
        # Create y which is an array (for each j in N) of \Deltap
        y = []
        for j in range(self.N):
            y_j = {}
            cnt = 0
            for i_2 in range(len(self.robots)):
                if i != i_2:
                    y_j[i_2] = sol.value(y_vars[j][2*cnt:2*cnt+2])
                    cnt += 1
            y.append(y_j)
        
        return xsi, y



if __name__ == "__main__":
    t_range = np.array([0,10])

    x_ref_r1 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[-20,18,-np.deg2rad(45),0,0,0],
                                                                      [18,-20,-np.deg2rad(45),0,0,0],]))
    r1 = Robot(x0=np.array([-20,20,-np.deg2rad(45),0,0,0]),x_ref=x_ref_r1,id=1)
    # r1.add_phi(0)

    x_ref_r2 = ReferenceTrajectory(t_range=t_range, x_range=np.array([[-18,-20,np.deg2rad(45),0,0,0],
                                                                      [10,0,np.deg2rad(45),0,0,0],]))
    r2 = Robot(x0=np.array([-20,-20,np.deg2rad(45),0,0,0]),x_ref=x_ref_r2,id=2)

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
               
