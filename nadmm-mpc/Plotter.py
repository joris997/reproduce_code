import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Plotter():
    def __init__(self,solver):
        self.robots = solver.robots
        self.N = solver.N
        
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