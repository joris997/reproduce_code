import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cp
from cvxopt import matrix, solvers
import gurobipy as gp
from gurobipy import GRB


# phi = G_[7.5,10](||x||^2 < 5^2)

if __name__ == "__main__":
    # phi = G_[7.5,10](||x||^2 < 5^2)
    # h = 5^2 - ||x||^2
    t_star = 7.5
    h = lambda x: 5**2 - cp.norm(x,2) #(x[0]*x[0] + x[1]*x[1]) # x.T@x
    gamma = lambda t: (gamma_inf - gamma_0)/(t_star)*t + gamma_0 if t < t_star else gamma_inf
    b = lambda t,x: -gamma(t) + h(x)

    x0 = np.array([3.68, 3.68])
    chi = 0.0
    eps = 1e-5
    # print("h(x0): ", h(x0))

    # p: number of barrier functions (conjunctions)
    # l: {1,...,p+1} the index for the barrier

    ## create variables, where are the binary?
    eta       = cp.Variable(1,name="eta")
    r         = cp.Variable(1,name="r")
    D         = cp.Variable(1,name="D")
    gamma_0   = cp.Variable(1,name="gamma_0")
    gamma_inf = cp.Variable(1,name="gamma_inf")
    xi        = cp.Variable(2,name="xi")

    max_val   = cp.Variable(1,name="max_val")

    ## set the objective
    prob = cp.Problem(cp.Maximize(r),
                      [b(0,x0) >= chi,
                       b(10,xi) >= chi,
                       gamma_0 <= h(x0) + eps,
                       max_val <= cp.minimum(r,gamma_0) + eps,
                       max_val >= cp.maximum(r,gamma_0) - eps,
                       max_val + eps <= gamma_inf,
                       gamma_inf <= h(xi) - eps,
                       eta >= eps,
                       r >= eps,
                       D >= eps])
    prob.solve()
    print("\nThe optimal value is", prob.value)
    print("eta: ", eta.value)
    print("r: ", r.value)
    print("D: ", D.value)
    print("gamma_0: ", gamma_0.value)
    print("gamma_inf: ", gamma_inf.value)
    print("xi: ", xi.value)

    ## print
    gamma_inf = gamma_inf.value
    gamma_0 = gamma_0.value
    gamma = lambda t: (gamma_inf - gamma_0)/(t_star)*t + gamma_0 if t < t_star else gamma_inf
    
    t_range = np.linspace(0,10,100)
    gamma_t = np.zeros(100)
    for idx,t in enumerate(t_range):
        gamma_t[idx] = gamma(t)
    fig,axs = plt.subplots(2,1)
    axs.ravel()

    axs[0].plot(t_range,gamma_t)
    axs[0].grid(True)
    plt.savefig("lindemann/gamma.png")


    ## now do some control with it
    f = np.array([[0,0,1,0],
                  [0,0,0,1],
                  [0,0,0,0],
                  [0,0,0,0]])
    g = np.array([[0,0],
                  [0,0],
                  [1,0],
                  [0,1]])
    sol = solvers.qp




#### using gurobi, MILP (but I was wrong, it's not a MILP)
# # phi = G_[7.5,10](||x||^2 < 5^2)
# # h = 5^2 - ||x||^2
# t_star = 7.5
# h = lambda x: 5**2 - (x[0]*x[0] + x[1]*x[1]) # x.T@x
# gamma = lambda t: (gamma_inf - gamma_0)/(t_star)*t + gamma_0 if t < t_star else gamma_inf
# b = lambda t,x: -gamma(t) + h(x)

# x0 = np.array([3.68, 3.68])
# chi = 0.0
# eps = 1e-10
# print("h(x0): ", h(x0))

# # p: number of barrier functions (conjunctions)
# # l: {1,...,p+1} the index for the barrier

# ## create the model
# m = gp.Model("STL_CBF")
# m.Params.DualReductions = 0
# m.reset(0)

# ## create variables, where are the binary?
# eta       = m.addVar(vtype=GRB.CONTINUOUS, name="eta")
# r         = m.addVar(vtype=GRB.CONTINUOUS, name="r")
# D         = m.addVar(vtype=GRB.CONTINUOUS, name="D")
# gamma_0   = m.addVar(vtype=GRB.CONTINUOUS, name="gamma_0")
# gamma_inf = m.addVar(vtype=GRB.CONTINUOUS, name="gamma_inf")
# xi        = m.addMVar(shape=2, vtype=GRB.CONTINUOUS, name="xi")

# max_val   = m.addVar(vtype=GRB.CONTINUOUS, name="max_val")

# ## set the objective
# m.setObjective(r, GRB.MAXIMIZE)

# ## add constraints
# # 5.9b
# m.addConstr(b(0,x0) >= chi, name="5.9b")
# # 5.9c
# m.addConstr(b(10,xi) >= chi, name="c5.9c")
# # 5.9d
# m.addConstr(gamma_0 <= h(x0) + eps, name="c5.9d")
# # 5.9e
# m.addConstr(max_val == gp.max_([r,gamma_0]), name="c5.9e0")
# m.addConstr(max_val + eps <= gamma_inf, name="c5.9e1")
# m.addConstr(gamma_inf <= h(xi) - eps, name="c5.9e2")
# # 5.9f
# m.addConstr(eta >= eps, name="c5.9f_eta")
# m.addConstr(r >= eps, name="c5.9f_r")
# m.addConstr(D >= eps, name="c5.9f_D")

# ## pre
# m.computeIIS()
# m.write("model.ilp")
# if m.status == GRB.INFEASIBLE:
#     m.feasRelaxS(1, False, False, True)
#     m.optimize()

# ## optimize
# m.optimize()
# for v in m.getVars():
#     print('%s: %g' % (v.VarName, v.X))
# print('Obj: %g' % m.ObjVal)