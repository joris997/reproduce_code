import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class Plotter():
    def __init__(self,solver):
        self.robots = solver.robots
        self.N = solver.N
        
    def plot_results(self):
        # create 1x2 subplot
        fig, axs = plt.subplots(2,3,figsize=(18,12))

        # Plot the world bounding box
        axs[0,0].plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                    [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        
        # plot the STL predicates
        for i in range(len(self.robots)):
            for p in self.robots[i].preds:
                circle = patches.Circle((p["c"][0],p["c"][1]),p["c_r"],fill=True)
                axs[0,0].add_patch(circle)
                
        # plot the initial states
        for i in range(len(self.robots)):
            # create cirkel with self.robots[i].R radius
            circle = patches.Circle((self.robots[i].x_vars_hist[0][0][0],self.robots[i].x_vars_hist[0][0][1]),self.robots[i].R,fill=True)
            axs[0,0].add_patch(circle)
            axs[0,0].text(self.robots[i].x_vars_hist[0][0][0],self.robots[i].x_vars_hist[0][0][1],str(i),fontsize=24,color='red')

        # plot the mpc trajectory
        for i in range(len(self.robots)):
            for j in range(len(self.robots[i].x_vars_hist)):
                x_vars = self.robots[i].x_vars_hist[j]
                x_traj = np.array([x_vars[i][0] for i in range(self.N+1)]) # +1
                y_traj = np.array([x_vars[i][1] for i in range(self.N+1)]) # +1
                axs[0,0].plot(x_traj,y_traj,'k')
        axs[0,0].axis('equal')
        axs[0,0].grid(True)

        if len(self.robots[i].b_hist) > 0:
            bx_traj = [self.robots[0].b_hist[j][0] for j in range(len(self.robots[0].b_hist))]
            by_traj = [self.robots[0].b_hist[j][1] for j in range(len(self.robots[0].b_hist))]
            axs[0,1].plot(bx_traj,'r')
            axs[0,1].plot(by_traj,'b')
            axs[0,1].legend(['x p_[1|0]','y p_[1|0]'])
            axs[0,1].set_title('Belief of agent 1 by agent 0')
            axs[0,1].grid(True)

            bx_traj = [self.robots[1].b_hist[j][0] for j in range(len(self.robots[1].b_hist))]
            by_traj = [self.robots[1].b_hist[j][1] for j in range(len(self.robots[1].b_hist))]
            axs[1,1].plot(bx_traj,'r')
            axs[1,1].plot(by_traj,'b')
            axs[1,1].legend(['x p_[0|1]','y p_[0|1]'])            
            axs[1,1].set_title('Belief of agent 0 by agent 1')
            axs[1,1].grid(True)

        if len(self.robots[i].y_hist) > 0:
            yx_traj = [self.robots[0].y_hist[j][0] for j in range(len(self.robots[0].y_hist))]
            yy_traj = [self.robots[0].y_hist[j][1] for j in range(len(self.robots[0].y_hist))]
            axs[0,2].plot(yx_traj,'r')
            axs[0,2].plot(yy_traj,'b')
            axs[0,2].legend(['y_x r0','y_y r0','y_x r1','y_y r1'])
            axs[0,2].set_title('Shared variable, vector distance to other agent')
            axs[0,2].grid(True)

            yx_traj = [self.robots[1].y_hist[j][0] for j in range(len(self.robots[1].y_hist))]
            yy_traj = [self.robots[1].y_hist[j][1] for j in range(len(self.robots[1].y_hist))]
            axs[1,2].plot(yx_traj,'r')
            axs[1,2].plot(yy_traj,'b')
            axs[1,2].legend(['y_x r0','y_y r0','y_x r1','y_y r1'])
            axs[1,2].set_title('Shared variable, vector distance to other agent')
            axs[1,2].grid(True)

            

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
        self.ax.grid(True)
    
    def update(self, j):
        self.ax.clear()
        self.ax.plot([self.robots[0].x_lim[0],self.robots[0].x_lim[1],self.robots[0].x_lim[1],self.robots[0].x_lim[0],self.robots[0].x_lim[0]],
                     [self.robots[0].y_lim[0],self.robots[0].y_lim[0],self.robots[0].y_lim[1],self.robots[0].y_lim[1],self.robots[0].y_lim[0]],'k')
        for i in range(len(self.robots)):
            x_vars = self.robots[i].x_vars_hist[j]
            x_traj = np.array([x_vars[k][0] for k in range(self.N+1)]) # +1
            y_traj = np.array([x_vars[k][1] for k in range(self.N+1)]) # +1
            self.ax.plot(x_traj,y_traj,'k')
            circle = patches.Circle((x_traj[0],y_traj[0]),self.robots[i].R,fill=True)
            self.ax.add_patch(circle)
        self.ax.axis('equal')
        self.ax.grid(True)