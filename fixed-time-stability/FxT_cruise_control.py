import numpy as np
from scipy.integrate import solve_ivp
import qpsolvers
import matplotlib.pyplot as plt
import math

f0 = 0.1
f1 = 5
f2 = 0.05
ag = 9.81
M = 1650

# desired
tau_d = 1.8
v_d = 22

# controller
Tud = 10
mu = 5
alpha1 = (mu*math.pi)/(2*Tud)
alpha2 = (mu*math.pi)/(2*Tud)
gamma1 = 1 + 1/mu
gamma2 = 1 - 1/mu

def get_h_G(x):
    return (x[0] - v_d)**2
def get_dhdx_G(x):
    return np.array([2*(x[0]-v_d), 0, 0])

def get_h_S(x):
    return (tau_d*x[0] - x[2])
def get_dhdx_S(x):
    return np.array([tau_d, 0, -1])

def get_Fr(x):
    vf = x[0]
    return f0 + f1*vf + f2*(vf**2)

def dx(t,x):
    vf = x[0]
    vl = x[1]

    if t < 10:
        aL = 0
    elif t>=10 and t<18:
        aL = (22-10)/(18-10)
    elif t>=18 and t<30:
        aL = 0
    elif t>=30 and t<40:
        aL = (28-22)/(40-30)
    elif t>=40:
        aL = 0
    # aL = -al*ag

    fx = np.array([-get_Fr(x)/M, aL, vl-vf])
    gx = np.array([1/M, 0., 0.])
    # u = 0.0

    ## QP setup
    H = np.array([[1,0,0],
                  [0,1e5,0],
                  [0,0,1e5]])
    F = np.array([0,1e5,0])
    A_ineq_G = np.block([[get_dhdx_G(x)@gx, -get_h_G(x), 0]])
    # A_ineq_G = np.block([[get_dhdx_G(x)@gx, 0, 0]])
    ub_ineq_G = np.array([-get_dhdx_G(x)@fx - alpha1*max(0,get_h_G(x))**gamma1 - alpha2*max(0,get_h_G(x))**gamma2])
    
    # A_ineq_S = np.block([[-get_dhdx_S(x)@gx - get_h_S(x), 0, get_h_S(x)]])
    A_ineq_S = np.block([[get_dhdx_S(x)@gx, 0, 0]])
    ub_ineq_S = np.array([-get_dhdx_S(x)@fx])
    
    use_G = True
    use_S = True
    if use_S and use_G:
        A_ineq = np.vstack((A_ineq_G,A_ineq_S))
        ub_ineq = np.vstack((ub_ineq_G,ub_ineq_S))
    elif use_S and not use_G:
        A_ineq = A_ineq_S
        ub_ineq = ub_ineq_S
    elif not use_S and use_G:
        A_ineq = A_ineq_G
        ub_ineq = ub_ineq_G


    # u = solve_qp(H,F,A_ineq,ub_ineq,lb=-umax,ub=umax,solver='gurobi')
    # u = solve_qp(H,F,A_ineq_G,ub_ineq_G,lb=-umax,ub=umax,solver='gurobi')
    # u = solve_qp(H,F,A_ineq_S,ub_ineq_S,lb=-umax,ub=umax,solver='gurobi')
    umin = np.array([-0.25*M*ag,0,0])
    umax = np.array([ 0.25*M*ag,1e10,1e10])

    u = qpsolvers.solve_qp(H,F,A_ineq,ub_ineq,solver='gurobi',verbose=False)
    # u = qpsolvers.solve_qp(H,F,A_ineq,ub_ineq,solver='quadprog',verbose=True)
    # u = qpsolvers.solve_qp(np.eye(3),np.zeros((3,1)),G=A_ineq,h=ub_ineq,solver='cvxopt',verbose=True)
    if u is None:
        print("QP failed, solution is None")
        print(qpsolvers.available_solvers)
        u = np.zeros((3,))

    print("A_ineq: ", A_ineq)
    print("ub_ineq: ", ub_ineq)
    print("u: ", u)

    return fx + gx*u[0]

if __name__ == "__main__":
    x0 = np.array([22, 10, 150])

    trange = [0, 100]
    sol = solve_ivp(dx,trange,x0,max_step=0.1)

    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs = axs.ravel()
    axs[0].plot(sol.t,sol.y[0,:])
    axs[0].plot(sol.t,sol.y[1,:])
    axs[0].grid(True)
    axs[0].legend(['vf','vl'])

    axs[1].plot(sol.t,sol.y[2,:])
    axs[1].grid(True)
    axs[1].legend(['d'])

    h_G = []
    h_S = []
    for i in range(0,len(sol.t)):
        h_G = np.append(h_G,get_h_G(sol.y[:,i]))
        h_S = np.append(h_S,get_h_S(sol.y[:,i]))

    axs[2].plot(sol.t,h_G)
    axs[2].grid(True)
    axs[2].legend(['h_G'])

    axs[3].plot(sol.t,h_S)
    axs[3].grid(True)
    axs[3].legend(['h_S'])

    plt.savefig("fluff/FxT_cruise_control.png")

