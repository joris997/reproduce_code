import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Plotter():
    def __init__(self,solver):
        self.robots = solver.robots
        self.N = solver.N
        
    def plot_results(self):
        # create 1x2 subplot
        fig, axs = plt.subplots(1,3,figsize=(18,6))

        # Plot the world bounding box
        axs[0].plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                 [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        
        # plot the initial states
        for i in range(len(self.robots)):
            axs[0].plot(self.robots[i].x_vars_hist[0][0][0],self.robots[i].x_vars_hist[0][0][1],'o')
            axs[0].text(self.robots[i].x_vars_hist[0][0][0],self.robots[i].x_vars_hist[0][0][1],str(i))

        # plot the mpc trajectory
        for i in range(len(self.robots)):
            for j in range(len(self.robots[i].x_vars_hist)):
                x_vars = self.robots[i].x_vars_hist[j]
                x_traj = np.array([x_vars[i][0] for i in range(self.N)]) # +1
                y_traj = np.array([x_vars[i][1] for i in range(self.N)]) # +1
                axs[0].plot(x_traj,y_traj,'k')
        axs[0].axis('equal')

        if len(self.robots[i].b_hist) > 0:
            bx_traj = [self.robots[0].b_hist[j][0] for j in range(len(self.robots[0].b_hist))]
            by_traj = [self.robots[0].b_hist[j][1] for j in range(len(self.robots[0].b_hist))]
            axs[1].plot(bx_traj,'r')
            axs[1].plot(by_traj,'b')

            # bx_traj = [self.robots[1].b_hist[j][0] for j in range(len(self.robots[1].b_hist))]
            # by_traj = [self.robots[1].b_hist[j][1] for j in range(len(self.robots[1].b_hist))]
            # axs[1].plot(bx_traj,'r--')
            # axs[1].plot(by_traj,'b--')

            axs[1].legend(['b_x r0','b_y r0','b_x r1','b_y r1'])
            axs[1].set_title('Belief of other agent state')

        if len(self.robots[i].y_hist) > 0:
            yx_traj = [self.robots[0].y_hist[j][0] for j in range(len(self.robots[0].y_hist))]
            yy_traj = [self.robots[0].y_hist[j][1] for j in range(len(self.robots[0].y_hist))]
            axs[2].plot(yx_traj,'r')
            axs[2].plot(yy_traj,'b')

            # yx_traj = [self.robots[1].y_hist[j][0] for j in range(len(self.robots[1].y_hist))]
            # yy_traj = [self.robots[1].y_hist[j][1] for j in range(len(self.robots[1].y_hist))]
            # axs[2].plot(yx_traj,'r--')
            # axs[2].plot(yy_traj,'b--')

            axs[2].legend(['y_x r0','y_y r0','y_x r1','y_y r1'])
            axs[2].set_title('Shared variable, vector distance to other agent')

            

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
            x_traj = np.array([x_vars[k][0] for k in range(self.N)]) # +1
            y_traj = np.array([x_vars[k][1] for k in range(self.N)]) # +1
            self.ax.plot(x_traj,y_traj,'k')
            self.ax.plot(x_traj[0],y_traj[0],'ro')
        self.ax.axis('equal')