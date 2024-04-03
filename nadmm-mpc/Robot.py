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

        self.L = 3   # wheelbase
        self.R = 1.5 # radius of the robot

        # saving stuff
        self.x_hist = []
        self.x_vars_hist = []
        self.u_vars_hist = []

        self.b_hist = []
        self.y_hist = []
        self.lambda_hist = []

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

    # def run_synchronous_mpc(self):

