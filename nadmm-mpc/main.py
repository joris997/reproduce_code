import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca
import casadi.tools as ca_tools
import time

class ReferenceTrajectory:
    def __init__(self, t_range,x_range):
        self.t_range = t_range
        self.x_range = x_range

class Robot:
    def __init__(self, x0, x_ref, id):
        self.x0 = x0
        self.x_ref = x_ref
        self.id = id

        self.nx = 6
        self.nu = 2

        self.x_lim = np.array([-21,21])
        self.y_lim = np.array([-21,21])
        self.eta_lim = np.array([-np.inf, np.inf])
        self.vx_lim = np.array([-1, 15])
        self.ay_lim = np.array([-3,3])
        self.omega_lim = np.array([-np.inf, np.inf])
        # x = [x, y, eta, v_x, a_y, omega]
        self.x_lb = np.array([self.x_lim[0],self.y_lim[0],self.eta_lim[0],self.vx_lim[0],self.ay_lim[0],self.omega_lim[0]])
        self.x_ub = np.array([self.x_lim[1],self.y_lim[1],self.eta_lim[1],self.vx_lim[1],self.ay_lim[1],self.omega_lim[1]])

        self.ax_lim = np.array([-6,6])
        self.delta_lim = np.array([-0.5, 0.5])
        # u = [a_x, delta]
        self.u_lb = np.array([self.ax_lim[0],self.delta_lim[0]])
        self.u_ub = np.array([self.ax_lim[1],self.delta_lim[1]])

        self.L = 4
        self.R = 1

        # saving stuff
        self.x_hist = []
        self.x_vars_hist = []
        self.u_vars_hist = []

    def dynamics(self, x, u, dt=0.1):
        # x = [x, y, eta, v_x, a_y, omega]
        # u = [a_x, delta]
        x_next = ca.MX(np.zeros(6))
        x_next[0] = x[0] + (x[3]*np.cos(x[2]))*dt
        x_next[1] = x[1] + (x[3]*np.sin(x[2]))*dt
        x_next[2] = x[2] + ((x[3]*np.tan(x[4]))/self.L)*dt
        x_next[3] = x[3] + (u[0])*dt
        x_next[4] = x[4] + ((2*u[0]*x[5] + x[3]*u[1])*(x[3]/self.L))*dt
        x_next[5] = x[5] + (u[1])*dt
        return x_next
    
    def get_x_ref(self, t):
        x_ref = np.zeros(6)
        if t < self.x_ref.t_range[0]:
            x_ref = self.x_ref.x_range[0]
        elif t > self.x_ref.t_range[-1]:
            x_ref = self.x_ref.x_range[-1]
        else:
            for i in range(self.nx):
                x_ref[i] = np.interp(t,self.x_ref.t_range,self.x_ref.x_range[:,i])
        return x_ref
    
    
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
            print("Solve time: ", solve_time)
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

    def plot_results(self):
        # Plot the world bounding box
        plt.plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                 [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        
        # plot the initial states
        for i in range(len(self.robots)):
            plt.plot(self.robots[i].x0[0],self.robots[i].x0[1],'o')

        # plot the mpc trajectory, only for robot 0
        for i in range(len(self.robots)):
            for j in range(len(self.robots[i].x_vars_hist)):
                x_vars = self.robots[i].x_vars_hist[j]
                x_traj = np.array([x_vars[i][0] for i in range(self.N+1)])
                y_traj = np.array([x_vars[i][1] for i in range(self.N+1)])
                plt.plot(x_traj,y_traj,'k')
                # plt.plot(x_traj[0],y_traj[0],'ro')

        plt.axis('equal')
        plt.savefig("nadmm-mpc/results.png")
    
    def plot_animation(self):
        # Create an animation replaying the multi-agent MPC
        self.fig, self.ax = plt.subplots()
        anim = FuncAnimation(self.fig, self.update, frames=range(len(self.robots[0].x_vars_hist)), init_func=self.init, interval=100)
        anim.save('nadmm-mpc/animation.gif', writer='imagemagick')

    def init(self):
        self.ax.clear()
        self.ax.plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                     [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        self.ax.axis('equal')
    
    def update(self, j):
        self.ax.clear()
        self.ax.plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                     [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        for i in range(len(self.robots)):
            x_vars = self.robots[i].x_vars_hist[j]
            x_traj = np.array([x_vars[k][0] for k in range(self.N+1)])
            y_traj = np.array([x_vars[k][1] for k in range(self.N+1)])
            self.ax.plot(x_traj,y_traj,'k')
            self.ax.plot(x_traj[0],y_traj[0],'ro')
        self.ax.axis('equal')



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

    robots = [r1,r2,r3,r4]
    solver = CentralizedSolver(robots,t_range)
    solver.solve()
    solver.plot_results()
    solver.plot_animation()
               
