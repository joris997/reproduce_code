### (Joris Verhagen, jorisv@kth.se)
# non-convex STL motion planning of a DT single integrator agent
# 
# min_{u,x} u^T Q u
#   s.t.    x \satsifies \phi
#           x_{k+1} = x_k + u
#           u \in U
#           x_0 = p_0
#           x \in X
#
# var:  u \in \mathcal{R}^{2n}
#       x \in \mathcal{R}^{2n}

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (MathematicalProgram, eq, ge, le)
from pydrake.solvers.ipopt import IpoptSolver

# locations of interest, [xmin,xmax,   ymin,ymax]
location_A = [5.0, 6.0,   5.0, 6.0]
location_B = [8.0, 9.0,  -1.0, 2.0]

class Spec(object):
    def __init__(self,type,range,location,location2=[]):
        self.type = type
        self.range = range
        self.location = location
        self.location2 = location2  # used for UNTIL operator


def minish(x):
    gamma = 10
    return -1/gamma*np.log(np.sum(np.exp(-gamma*x),axis=0))

def maxish(x):
    eta = 10
    return (np.sum(x*np.exp(eta*x),axis=0))/(np.sum(np.exp(eta*x),axis=0))

def dynamic_constraints(prog, X, U):
    for k in range(np.size(X,0)-1): # for each discretization point
        for j in range(np.size(X,1)): # for each state
            c = prog.AddConstraint(eq(X[k+1,j]-X[k,j] - U[k,j],[0]))
            c.evaluator().set_description(f"dynamics_{k}_{j}")

def control_constraints(prog, U, umax):
    for j in range(np.size(U,1)): # for each control input
        c = prog.AddBoundingBoxConstraint(-umax[j],umax[j], U[:,j])
        c.evaluator().set_description(f"control_constraint_{j}")

def state_constraints(prog, X, xmax):
    for j in range(np.size(X,1)): # for each control input
        c = prog.AddBoundingBoxConstraint(-xmax[j],xmax[j], X[:,j])
        c.evaluator().set_description(f"state_constraint_{j}")

def init_state_constraint(prog, X, p0):
    c = prog.AddConstraint(eq(X[0,:],p0))
    c.evaluator().set_description("init_state_constraint")

def STL_constraints(prog, X, U, dt, phi):
    N = np.size(X,0) - 1
    for spec in phi: # we assume it's always an \and spec

        # get the location of interest
        if spec.location == "A":
            loc = location_A
        elif spec.location == "B":
            loc = location_B
        else:
            print(f"location {spec.location} not recognized, so ignored")

        # get the indices that are relevant to this predicate
        idxs = []
        for idx,t in enumerate(np.linspace(0,N*dt,N)):
            if spec.range[0] <= t <= spec.range[1]:
                idxs.append(idx)
        print("times: ", np.linspace(0,N*dt,N)[idxs])

        # ALWAYS
        if spec.type == "G":
            g_xi = minish(X[idxs])
            c = prog.AddConstraint(ge(g_xi[0],[loc[0]])) 
            c = prog.AddConstraint(le(g_xi[0],[loc[1]]))
            c = prog.AddConstraint(ge(g_xi[1],[loc[2]]))
            c = prog.AddConstraint(le(g_xi[1],[loc[3]]))

        # EVENTUALLY
        elif spec.type == "F":
            f_xi = maxish(X[idxs])
            c = prog.AddConstraint(ge(f_xi[0],[loc[0]])) 
            c = prog.AddConstraint(le(f_xi[0],[loc[1]]))
            c = prog.AddConstraint(ge(f_xi[1],[loc[2]]))
            c = prog.AddConstraint(le(f_xi[1],[loc[3]]))

        # UNTIL
        elif spec.type == "U":
            # u_xi = []
            # for i in idxs:
            #     u_xi.append( minish([X[i],minish([[X[j] for j in range(idxs[0],i)],X[i::]])]) )
            # u_xi = maxish(u_xi)
            print("Until not yet implemented")
            
        else:
            print(f"predicate {spec.type} implemented, so ignored")



if __name__ == "__main__":
    p0 = [0.0, 0.0]

    # specification
    # \phi = G_[5,10](A) \top F_[12,15](B)

    # discretization points
    N = 100
    # final time
    tf = 15
    dt = tf/N

    n_states = len(p0)
    n_ctrl = 2

    umax = [0.5, 0.5]
    xmax = [10, 10]

    prog = MathematicalProgram()
    X = prog.NewContinuousVariables(N+1, n_states, "X")
    U = prog.NewContinuousVariables(N+1, n_ctrl, "U")

    # add dynamics constraints
    dynamic_constraints(prog, X, U)
    # add u constraints
    control_constraints(prog, U, umax)
    # add x constraints
    state_constraints(prog, X, xmax)
    # add initial state constraint
    init_state_constraint(prog, X, p0)

    # cost function
    R = np.ones(n_ctrl)
    c = prog.AddQuadraticCost(sum([np.matmul(R[i]*U[:,i],U[:,i].T) for i in range(n_ctrl)]))

    ### STL specification
    phi = [Spec("G",[5,10],"A"),
           Spec("F",[12,15],"B"),
           Spec("U",[10,15],"A","B")]
    STL_constraints(prog, X, U, dt, phi)

    ### solve the non-convex opt
    solver = IpoptSolver()
    result = solver.Solve(prog)
    x = result.GetSolution(X)
    u = result.GetSolution(U)
    t = np.linspace(0,tf,N+1)
    

    fig, axs = plt.subplots(2,2,figsize=(10,10))

    axs[0,0].plot(t,x[:,0])
    axs[0,0].plot(t,x[:,1])
    axs[0,0].set_xlabel('time [s]')
    axs[0,0].set_ylabel('x [m]')
    axs[0,0].grid(True)

    axs[0,1].plot(t,u[:,0])
    axs[0,1].plot(t,u[:,1])
    axs[0,1].set_xlabel('time [s]')
    axs[0,1].set_ylabel('u [m]')
    axs[0,1].grid(True)

    regionA = plt.Rectangle((location_A[0],location_A[2]),location_A[1]-location_A[0],location_A[3]-location_A[2])
    axs[1,0].add_patch(regionA)
    regionB = plt.Rectangle((location_B[0],location_B[2]),location_B[1]-location_B[0],location_B[3]-location_B[2])
    axs[1,0].add_patch(regionB)
    axs[1,0].plot(x[:,0],x[:,1],color='red')
    axs[1,0].set_xlabel('x pos [m]')
    axs[1,0].set_xlabel('y pos [m]')
    axs[1,0].grid(True)
    
    plt.savefig("STL/NonConvex_STL/result.png")
