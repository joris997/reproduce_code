from gurobipy import Model
import gurobipy as gp
from gurobipy import GRB

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from scipy.optimize import root_scalar

from helpers.helpers_classes import Spec, Pred, MyPolygon, MySquare, Mu, BezierSolution, BezierSolutions
from helpers.helpers_spline_opt import get_halfspace_polyhedral, get_derivative_control_points, Polygon, \
    get_derivative_control_points_gurobi, divide_bezier, divide_bezier_2, minkowski_sum
from helpers.MVEEn import MVEE_opt, MVEE_it

from planning.robustMILP2.STL_qualitative import qual_AND, qual_OR, qual_parse_operator, qual_MU, parse_time, \
    add_temporal_robustness, qual_space_and_time
from planning.robustMILP2.STL_quantitative import quant_parse_operator
from planning.robustMILP2.STL_specs import get_stl_spec

from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)
from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
    Trajectory,
)

class BezierSTL_MILP(object):
    def __init__(self,spec,robot,world,objective="theta_p"):
        print("\n")
        # STL
        self.spec = spec

        self.robot = robot
        self.world = world

        self.objective = objective

        self.n_agents = spec.n_agents
        self.dim = spec.dim

        # some timing constants
        self.t0 = spec.t0
        self.tf = spec.tf
        self.dt = spec.tf - spec.t0
        
        self.N = spec.N
        self.Ts = self.tf/(self.N)
        self.dh = self.Ts

        # shrinking and bloating of the areas of interest with safety margin
        self.shrink = 0.10 #self.robots[0].radius + 0.05

        self.bigM = 1e4

        # lower and upper bound of the time derivative
        self.dh_lb = 1e-2
        self.dh_ub = 1e4

        self.evaluated = False

    def construct(self):
        self.order = self.spec.order
        self.n_cp = self.order + 1

        # set up the model
        self.prog = gp.Model("prog1")
        self.prog.setParam(GRB.Param.OutputFlag, 0)
        self.prog.setParam(GRB.Param.IntFeasTol,1e-5)
        self.prog.setParam(GRB.Param.MIPGap,    1e-5)
        self.prog.setParam(GRB.Param.IterationLimit, 100000)

        # add continuous state and input variables
        self.r_var = self.prog.addMVar((self.n_agents*self.dim,self.n_cp,self.N),\
                                       lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        self.h_var = self.prog.addMVar((1,self.n_cp,self.N),\
                                       lb=self.t0,ub=self.tf,vtype=GRB.CONTINUOUS)
        self.prog.update()

        # now obtain the control points of the derivatives as linear combinations 
        # of the control points of the original bezier curve
        self.dr_var = get_derivative_control_points_gurobi(self.r_var,1)
        self.ddr_var = get_derivative_control_points_gurobi(self.r_var,2)

        self.dh_var = get_derivative_control_points_gurobi(self.h_var,1)
        self.ddh_var = get_derivative_control_points_gurobi(self.h_var,2)

        # add constraints of the end-point of one segment to the start-point of the next
        for i in range(self.n_agents*self.dim):
            for j in range(self.N-1):
                self.prog.addConstr(self.r_var[i,-1,j] == self.r_var[i,0,j+1])
                self.prog.addConstr(self.dr_var[i,-1,j] == self.dr_var[i,0,j+1])
                self.prog.addConstr(self.ddr_var[i,-1,j] == self.ddr_var[i,0,j+1])

        for j in range(self.N-1):
            self.prog.addConstr(self.h_var[0,-1,j] == self.h_var[0,0,j+1])
            self.prog.addConstr(self.dh_var[0,-1,j] == self.dh_var[0,0,j+1])
            self.prog.addConstr(self.ddh_var[0,-1,j] == self.ddh_var[0,0,j+1])
        
        # add initial and final state
        self._initial_position_constraint()
        self._initial_time_constraint()
        self._initial_velocity_constraint()
        self._final_position_constraint()
        self._final_time_constraint()
        self._final_velocity_constraint()
        self._positive_time_derivative_constraint()

        # add dynamic constraints
        self._velocity_constraints()
        # self._acceleration_constraints()

        # add the STL constraints
        self._stl_constraints()
     
        self.prog.update()

        self.num_vars = self.prog.getAttr('NumVars')
        self.num_bin_vars = self.prog.getAttr('NumBinVars')
        self.num_int_vars = self.prog.getAttr('NumIntVars')
        self.num_constrs = self.prog.getAttr('NumConstrs')

        print('NumConVars: %d'%self.num_vars)
        print('NumBinVars: %d'%self.num_bin_vars)
        # print('NumIntVars: %d'%self.num_int_vars)
        print('NumConstr:  %d'%self.num_constrs)



    ###############
    ### SOLVING ###
    ###############
    def solve(self):
        start = time.time()
        self.prog.optimize()
        end = time.time()
        self.solve_time = end-start
        print("\nSolving took %.3f seconds"%self.solve_time)
        print("N: ", self.N, "\t order: ", self.order)
        # print("Solved:     ", self.prog.status)
        # print("Objective:  ", self.prog.objVal)
        # print("Theta_m:    ", self.spec.theta_m.X)
        # print("Theta_p:    ", self.spec.theta_p.X)
        # print('* infeasible constraints: ')
        # for c in self.prog.getConstrs():
        #     if c.Slack > 1e-5:
        #         print("- ",c.ConstrName," slack = ",c.Slack)
        self.prog.printQuality()



    ############
    ### COST ###
    ############
    def add_cost(self):
        self.compute_objectives()
        # self.prog.setObjective(0.01*self.objV - self.spec.theta_m, GRB.MINIMIZE)
        if self.objective == "theta_m":
            self.prog.setObjective(- self.spec.theta_m, GRB.MINIMIZE)
        elif self.objective == "theta_p":
            # self.prog.setObjective(- self.spec.theta_p, GRB.MINIMIZE)
            print("theta_p: ", self.spec.pred.theta_p)
            self.prog.setObjective(1e-4*self.objV - self.spec.pred.theta_p, GRB.MINIMIZE)

        self.prog.update()
        # print if any constraints are not satisfied

    def compute_objectives(self):
        ### path length integral

        ### acceleration
        self.z_var = self.prog.addMVar(self.ddr_var.shape,
                                        lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        self.z2_var = self.prog.addMVar(self.ddr_var.shape,
                                         lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)

        for i in range(self.ddr_var.shape[0]):            # = dim
            for j in range(self.ddr_var.shape[1]):        # = n_cp
                for k in range(self.ddr_var.shape[2]):    # = N-1
                    self.prog.addConstr(self.z_var[i,j,k] == self.Ts*self.ddr_var[i,j,k])
                    self.prog.addConstr(self.z2_var[i,j,k] == gp.abs_(self.z_var[i,j,k]))
                    # self.prog.addConstr(self.z2_var[i,j,k] == gp.abs_(self.ddr_var[i,j,k]))
                    # print("adding c: ", i, j, k)

        self.objU = gp.quicksum(self.z2_var[i,j,k] for i in range(self.z2_var.shape[0]) \
                                                    for j in range(self.z2_var.shape[1]) \
                                                    for k in range(self.z2_var.shape[2]) )

        ### velocity
        self.z_var = self.prog.addMVar(self.dr_var.shape,
                                        lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        self.z2_var = self.prog.addMVar(self.dr_var.shape,
                                         lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)

        for i in range(self.dr_var.shape[0]):            # = dim
            for j in range(self.dr_var.shape[1]):        # = n_cp
                for k in range(self.dr_var.shape[2]):    # = N-1
                    self.prog.addConstr(self.z_var[i,j,k] == self.Ts*self.dr_var[i,j,k])
                    self.prog.addConstr(self.z2_var[i,j,k] == gp.abs_(self.z_var[i,j,k]))
                    # self.prog.addConstr(self.z2_var[i,j,k] == gp.abs_(self.ddr_var[i,j,k]))
                    # print("adding c: ", i, j, k)

        self.objV = gp.quicksum(self.z2_var[i,j,k] for i in range(self.z2_var.shape[0]) \
                                                    for j in range(self.z2_var.shape[1]) \
                                                    for k in range(self.z2_var.shape[2]) )

        self.prog.update()


    ########################
    ### MILP CONSTRAINTS ###
    ########################
    ### initial
    def _initial_position_constraint(self):
        self.prog.addConstrs((self.r_var[i,0,0] == self.spec.q0[i] for i in range(self.n_agents*self.dim)),
                             name="initial_position_constraint")      
        
    def _initial_velocity_constraint(self):
        self.prog.addConstrs((self.dr_var[i,0,0] == self.dh_var[0,0,0]*self.spec.dq0[i] for i in range(self.n_agents*self.dim)),
                             name="initial_velocity_constraint")
    
    def _initial_time_constraint(self):
        self.prog.addConstr(self.h_var[0,0,0] == self.spec.t0, 
                            name="initial_time_constraint")

    ### final
    def _final_position_constraint(self):
        self.prog.addConstrs((self.r_var[i,-1,-1] == self.spec.qf[i] for i in range(self.n_agents*self.dim)),
                             name="final_position_constraint")
        
    def _final_velocity_constraint(self):
        self.prog.addConstrs((self.dr_var[i,-1,-1] == self.dh_var[0,-1,-1]*self.spec.dqf[i] for i in range(self.n_agents*self.dim)),
                             name="final_velocity_constraint")
        
    def _final_time_constraint(self):
        self.prog.addConstr(self.h_var[0,-1,-1] == self.spec.tf, 
                            name="final_time_constraint")

    def _positive_time_derivative_constraint(self):
        self.prog.addConstrs((self.dh_var[0,j,idx] >= self.dh_lb for j in range(self.n_cp-1) for idx in range(self.N)),
                             name="positive_time_derivative_constraint")
        # and an upper bound, should be high as it limits the temporal robustness
        # when we have one bezier curve per satisfaction
        self.prog.addConstrs((self.dh_var[0,j,idx] <= self.dh_ub for j in range(self.n_cp-1) for idx in range(self.N)),
                             name="positive_time_derivative_constraint")

    ### others
    def _velocity_constraints(self):
        # dr = dq*dh
        # for the beziers, we do the actuation limits already in the dynamics
        # for each dimension
        for i in range(self.n_agents*self.dim):
            # inbetween each pair of nodes
            for idx in range(self.N):
                # for each control point in ddx
                for k in range(self.n_cp-1):
                    self.prog.addConstr(self.dr_var[i,k,idx] <= (self.robot.dqub_eval[i][0])*self.dh_var[0,k,idx],
                                        name=f"velocity constraint ub {i} {idx} {k}")
                    self.prog.addConstr((self.robot.dqlb_eval[i][0])*self.dh_var[0,k,idx] <= self.dr_var[i,k,idx],
                                        name=f"velocity constraint lb {i} {idx} {k}")
    
    def _acceleration_constraints(self):
        # ddr = ddq*dh^2 + dq*ddh
        # we can use a convex overapproximation of the quadratic term by taking its lower bound
        # for each dimension
        for i in range(self.n_agents*self.dim):
            # inbetween each pair of nodes
            for idx in range(self.N):
                # for each control point in ddx
                for k in range(self.n_cp-2):
                    self.prog.addConstr(self.ddr_var[i,k,idx] <= (self.robot.ddqub_eval[i][0])*self.dh_lb**2 + \
                                                            (self.robot.dqub_eval[i][0])*self.ddh_var[0,k,idx],
                                        name=f"acceleration constraint ub {i} {idx} {k}")
                    self.prog.addConstr((self.robot.ddqlb_eval[i][0])*self.dh_lb**2 + \
                                        (self.robot.dqlb_eval[i][0])*self.ddh_var[0,k,idx] <= self.ddr_var[i,k,idx],
                                        name=f"acceleration constraint lb {i} {idx} {k}")
  
    def _stl_constraints(self):
        self.zs = []
        self.theta_ps = []
        self.theta_ms = []

        quant_parse_operator(self,self.spec.pred)


    #######################
    ### POST-PROCESSING ###
    ####################### 
    def evaluate(self):
        # for the spec
        self.solve_status = self.prog.status
        self.obj_value = self.prog.objVal
        self.spec.z_sol = self.spec.z.X
        self.spec.theta_m_sol = self.spec.theta_m.X
        self.spec.theta_p_sol = self.spec.theta_p.X

        # for the preds
        for pred in self.spec.preds:
            try:
                if pred.add_tr and pred.z.X == 1:
                    pred.mu.z_space_sol = pred.mu.z_space.X
                    pred.z_time_sol = pred.z_time.X
                    pred.z_sol = pred.z.X
                    pred.theta_m_sol = pred.theta_m.X
                    pred.theta_p_sol = pred.theta_p.X
                    pred.h_min_sol = pred.h_min.X
                    pred.h_max_sol = pred.h_max.X
                    try:
                        pred.I_sol[0] = pred.I[0].X
                        pred.I_sol[1] = pred.I[1].X
                    except:
                        pred.I_sol[0] = pred.I[0]
                        pred.I_sol[1] = pred.I[1]

            except:
                pred.mu.z_space_sol = np.zeros((self.N,1))
                pred.z_time_sol = np.zeros((self.N,1))
                pred.z_sol = 0
                pred.theta_m_sol = 0
                pred.theta_p_sol = 0
                pred.h_min_sol = 0
                pred.h_max_sol = 0
                pred.I_sol = [self.t0,self.tf]
            
            if pred.type == "F":
                try:
                    pred.t_star_p_sol = pred.t_star_p.X
                    pred.t_star_m_sol = pred.t_star_m.X
                except:
                    pred.t_star_p_sol = 0
                    pred.t_star_m_sol = 0
            else:
                pred.t_star_p_sol = 0
                pred.t_star_m_sol = 0
                
            # print("----------------------------------------------------")
            # print(f"Predicate {pred.get_string()}")
            # print(f"z_space: ", pred.mu.z_space_sol.T)
            # print(f"z_time:  ", pred.z_time_sol.T)
            # print(f"z:       ", pred.z_sol)
            # print(f"t_star_p: ", pred.t_star_p_sol)
            # print(f"t_star_m: ", pred.t_star_m_sol)
            # print(f"h_min: {pred.h_min_sol} \t h_max: {pred.h_max_sol}")
            # print(f"theta_m: ", pred.theta_m_sol)
            # print(f"theta_p: ", pred.theta_p_sol)

        # get the bezier control points for r, h, and q
        self.r_cps_sol = self.r_var.X
        self.dr_cps_sol = get_derivative_control_points_gurobi(self.r_cps_sol,1)
        self.ddr_cps_sol = get_derivative_control_points_gurobi(self.r_cps_sol,2)

        self.h_cps_sol = self.h_var.X
        self.dh_cps_sol = get_derivative_control_points_gurobi(self.h_cps_sol,1)
        self.ddh_cps_sol = get_derivative_control_points_gurobi(self.h_cps_sol,2)

        self.q_cps_sol = self.r_cps_sol
        self.dq_cps_sol = np.zeros_like(self.dr_cps_sol)
        self.ddq_cps_sol = np.zeros_like(self.ddr_cps_sol)
        for i in range(self.N):
            for j in range(self.dq_cps_sol.shape[1]):
                self.dq_cps_sol[:,j,i] = self.dr_cps_sol[:,j,i]/self.dh_cps_sol[0,j,i]
            for j in range(self.ddq_cps_sol.shape[1]):
                self.ddq_cps_sol[:,j,i] = self.ddr_cps_sol[:,j,i]/self.Ts**2

        # print("\n----------------------------------------------------")
        # ts = [round(self.h_cps_sol[0,0,i],2) for i in range(self.N)]
        # ts.append(round(self.h_cps_sol[0,-1,-1],2))
        # print("t: ", ts)
        # print("----------------------------------------------------\n")

        # for i in range(len(self.preds_nested_stl)):
        #     print("---")
        #     print(f"pred[{i}].I:       {self.preds_nested_stl[i].I.X.T}")
        #     print(f"pred[{i}].I_var:   {self.preds_Ivar[i].X}")
        #     print(f"pred[{i}].z_space: {self.preds_nested_stl[i].mu.z_space.X.T}")
        #     print(f"pred[{i}].z_time:  {self.preds_nested_stl[i].z_time.X.T}")
        #     print(f"pred[{i}].z AND:   {self.preds_nested_stl[i].cs.X.T}")
        #     print(f"pred[{i}].z:       {self.preds_nested_stl[i].z.X}")

        # type of temporal constraint
        for pred in self.spec.preds:
            # now we need to check fo all z_time_sol = True, what kind overlap we have with the pred.range
            pred.z_time_type_sol = np.zeros((self.N,1))
            for idx in range(self.N):
                if pred.mu.z_space_sol[idx] == 1:
                    if self.h_cps_sol[0,0,idx] <= pred.I_sol[0] and self.h_cps_sol[0,-1,idx] <= pred.I_sol[1]:
                        # we start before but end within the time-range
                        pred.z_time_type_sol[idx] = 1
                    elif self.h_cps_sol[0,0,idx] >= pred.I_sol[0] and self.h_cps_sol[0,-1,idx] >= pred.I_sol[1]:
                        # we start within but end after the time-range
                        pred.z_time_type_sol[idx] = 3
                    elif self.h_cps_sol[0,0,idx] >= pred.I_sol[0] and self.h_cps_sol[0,-1,idx] <= pred.I_sol[1]:
                        # we are within the time-range
                        pred.z_time_type_sol[idx] = 2


        # now gather all the spline results, similar to the MultiSplineOptState
        self.bSols = []
        for i in range(self.N):
            self.bSols.append(BezierSolution(self,i))

        for idx in range(self.N):
            self.bSols[idx].generate_eval()

        # # check the temporary c arrays for the predicate if we wish to check some stuff
        # for pred in self.spec.preds:
        #     print("----------------------------------------------------")
        #     print(f"Predicate {pred.get_string()}")
        #     print(f"z_space: ", pred.z_space_sol.T)
        #     # print the first and last control point of h_var_sol to indicate the time-range of each Bezier curve
        #     print(f"h_var:   ", [[self.h_cps_sol[0,0,i], self.h_cps_sol[0,-1,i]] for i in range(self.N)])
        #     for i in range(len(pred.c_tmp)):
        #         print(f"c_tmp[{i}]: ", pred.c_tmp[i].getAttr("RHS"))
        #         # print(f"c_tmp[{i}]: ", pred.c_tmp[i].getAttr("LHS"), " <= ", pred.c_tmp[i].getAttr("RHS"))
        #         print(f"eq_tmp[{i}]: ", pred.eq_tmp[i].getValue())

        self.evaluated = True

    def eval_t(self,t):
        # get an array of all the times
        t_array = np.array([self.bSols[i].h_cps[0,0] for i in range(self.N)])

        idx = np.where(t_array <= t)[0][-1]
        # get the phasing variable by using the inverse of the h_spline
        t_traj = BsplineTrajectory(BsplineBasis(self.n_cp,self.n_cp),self.bSols[idx].h_cps)
        

        error = lambda s: t_traj.value(s)[0,0] - t
        # start = time.time()
        s = root_scalar(error, bracket=[0,1])
        return idx, s.root

    def eval_q(self,t):
        # check which curve we need to evaluate
        if t < self.spec.t0:
            return self.bSols[0].h_cps[0,0]
        elif t > self.spec.tf:
            return self.bSols[-1].h_cps[0,-1]
        else:
            idx, s = self.eval_t(t)

            q_traj = BsplineTrajectory(BsplineBasis(self.n_cp,self.n_cp),self.bSols[idx].q_cps)
            dq_traj = BsplineTrajectory(BsplineBasis(self.n_cp-1,self.n_cp-1),self.bSols[idx].dq_cps)

            q = q_traj.value(s)
            dq = dq_traj.value(s)
            return q, dq, idx



    #################
    ### ANIMATING ###
    #################
    def animate(self,axs_arg=None,path=None,verbose=True):
        Neval = 250

        # now pretend we traverse the curve
        trange = np.linspace(self.spec.t0,self.spec.tf,Neval)
        x = np.zeros((4,len(trange)))
        for i,t in enumerate(trange):
            q,dq = self.eval_q(t)
        
            x[0:2,i] = q[:,0]
            x[2:4,i] = dq[:,0]

        self.t_anim = t
        self.x_anim = x

        self.fig_anim = plt.figure(figsize=(10,10))
        self.ax_anim = plt.axes()
        anim = animation.FuncAnimation(self.fig_anim, self.animate_update,
                                       frames=Neval, interval=self.spec.tf/Neval*1e6)

        anim.save(path, writer = 'ffmpeg', fps = Neval/self.spec.tf)
        
    def animate_update(self,anim_i):
        self.ax_anim.clear()

        # plot the world
        self.world.plot(self.ax_anim)
        # plot the trajectory
        for i in range(self.N):
            self.ax_anim.plot(self.bSols[i].r_cps[0,:],self.bSols[i].r_cps[1,:],'ro',markersize=4)
            # the start and end control point in blue
            self.ax_anim.plot(self.bSols[i].r_cps[0,0],self.bSols[i].r_cps[1,0],'bo',markersize=4)
            self.ax_anim.plot(self.bSols[i].r_cps[0,-1],self.bSols[i].r_cps[1,-1],'bo',markersize=4)
        for i in range(self.N):
            self.ax_anim.plot(self.bSols[i].r_eval[0,:],self.bSols[i].r_eval[1,:],'k',markersize=4)
        self.ax_anim.grid(True)
        self.ax_anim.set_xlabel('x [m]')
        self.ax_anim.set_ylabel('y [m]')
        self.ax_anim.set_title('position')
        self.ax_anim.set_aspect('equal')

        # plot the robot as a circle
        robot = Circle((self.x_anim[0,anim_i],self.x_anim[1,anim_i]), \
                       self.robot.radius,facecolor='r')
        self.ax_anim.add_patch(robot)
        self.ax_anim.plot(self.x_anim[0,anim_i],self.x_anim[1,anim_i],'go')

        self.ax_anim.set_aspect('equal', 'box')
        return self.ax_anim
    


    ################
    ### PLOTTING ###
    ################
    def plot(self,axs_arg=None,path=None,verbose=True):
        if not self.evaluated:
            self.evaluate()

        if axs_arg is None:
            plot, axs = plt.subplots(3,3)
        else:
            axs = axs_arg

        try:
            size = 16, 12
            plot.set_size_inches(size)
            # plot.tight_layout()
        except:
            pass

        if True:
            if self.dim > 1:
                # plot the specification over x and y
                for idx,sp in enumerate(self.spec.preds):
                    sp.plot(axs[0,0],cnt=None,axis=None,pv="pos",obj=self.objective)
                # plot the world
                self.world.plot(axs[0,0])
                # plot the trajectory for each in n_robots
                for i_agent in range(self.n_agents):
                    i_agent_x = i_agent*self.dim
                    i_agent_y = i_agent_x + 1

                    for idx in range(self.N):
                        axs[0,0].plot(self.bSols[idx].r_cps[i_agent_x,:],self.bSols[idx].r_cps[i_agent_y,:],'ro',markersize=4)
                        # the start and end control point in blue
                        axs[0,0].plot(self.bSols[idx].r_cps[i_agent_x,0],self.bSols[idx].r_cps[i_agent_y,0],'bo',markersize=4)
                        axs[0,0].plot(self.bSols[idx].r_cps[i_agent_x,-1],self.bSols[idx].r_cps[i_agent_y,-1],'bo',markersize=4)
                    for idx in range(self.N):
                        axs[0,0].plot(self.bSols[idx].r_eval[i_agent_x,:],self.bSols[idx].r_eval[i_agent_y,:],'k',markersize=4)
                axs[0,0].grid(True)
                axs[0,0].set_xlabel('x [m]')
                axs[0,0].set_ylabel('y [m]')
                axs[0,0].set_title('position')
                axs[0,0].set_aspect('equal')    

            ### distance over time
            # plot the specification in the time-trajectory plot
            for idx,sp in enumerate(self.spec.preds):
                sp.plot(axs[0,1],idx,"x","pos")
            # plot the trajectory for each in n_robots
            for i_agent in range(self.n_agents):
                i_agent_x = i_agent*self.dim
                i_agent_y = i_agent_x + 1
                # plot the trajectory
                for idx in range(self.N):
                    axs[0,1].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].r_eval[i_agent_x,:],'k',markersize=4)
                # plot the waypoints
                for idx in range(self.N):
                    axs[0,1].plot(self.bSols[idx].h_cps[0,:],self.bSols[idx].r_cps[i_agent_x,:],'ro')
                    axs[0,1].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].r_cps[i_agent_x,0],'bo')
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel('pos [m]')
            axs[0,1].set_title('x-position over time')
            axs[0,1].grid(True)

            ### distance over time
            if self.dim > 1:
                # plot the specification in the time-trajectory plot
                for idx,sp in enumerate(self.spec.preds):
                    sp.plot(axs[0,2],idx,"y","pos")
                
                # plot the trajectory for each in n_robots
                for i_agent in range(self.n_agents):
                    i_agent_x = i_agent*self.dim
                    i_agent_y = i_agent_x + 1
                    # plot the trajectory
                    for idx in range(self.N):
                        axs[0,2].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].r_eval[i_agent_y,:],'k',markersize=4)
                    # plot the waypoints
                    for idx in range(self.N):
                        axs[0,2].plot(self.bSols[idx].h_cps[0,:],self.bSols[idx].r_cps[i_agent_y,:],'ro')
                        axs[0,2].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].r_cps[i_agent_y,0],'bo')
                axs[0,2].set_xlabel('t [s]')
                axs[0,2].set_ylabel('pos [m]')
                axs[0,2].set_title('y-position over time')
                axs[0,2].grid(True)
        
        if True:
            if self.dim > 1:
                ### velocity over time
                # plot the specification over x and y
                for idx,sp in enumerate(self.spec.preds):
                    sp.plot(axs[1,0],cnt=None,axis="x",pv="vel")
                    sp.plot(axs[1,0],cnt=None,axis="y",pv="vel")
                axs[1,0].axhline(y=self.robot.dqub_eval[0][0],color='k',linestyle=':')
                axs[1,0].axhline(y=self.robot.dqlb_eval[0][0],color='k',linestyle=':')
                # adding these only for the legend
                axs[1,0].plot(self.bSols[0].h_eval[0,:],self.bSols[0].dq_eval[0,:],'k',markersize=4,label='dx')
                axs[1,0].plot(self.bSols[0].h_eval[0,:],self.bSols[0].dq_eval[1,:],'k--',markersize=4,label='dy')
                # but this is the main loop
                for idx in range(self.N):
                    axs[1,0].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].dq_eval[0,:],'k',markersize=4)
                    axs[1,0].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].dq_eval[1,:],'k--',markersize=4)
                # plot the waypoints
                for idx in range(self.N):
                    axs[1,0].plot(self.bSols[idx].h_cps[0,:-1],self.bSols[idx].dq_cps[0,:],'ro')
                    axs[1,0].plot(self.bSols[idx].h_cps[0,:-1],self.bSols[idx].dq_cps[1,:],'rv')
                    axs[1,0].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].dq_cps[0,0],'bo')
                    axs[1,0].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].dq_cps[1,0],'bv')
                axs[1,0].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].dq_cps[0,-1],'bo')
                axs[1,0].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].dq_cps[1,-1],'bv')
                axs[1,0].set_xlabel('t [-]')
                axs[1,0].set_ylabel('vel [m/s]')
                axs[1,0].legend()
                axs[1,0].set_title('velocity')
                axs[1,0].grid(True)

            ### velocity x over time
            # plot the specification in the time-trajectory plot
            for idx,sp in enumerate(self.spec.preds):
                sp.plot(axs[1,1],idx,"x","vel")
            axs[1,1].axhline(y=self.robot.dqub_eval[0][0],color='k',linestyle=':')
            axs[1,1].axhline(y=self.robot.dqlb_eval[0][0],color='k',linestyle=':')
            # adding these only for the legend
            axs[1,1].plot(self.bSols[0].h_eval[0,:],self.bSols[0].dq_eval[0,:],'k',markersize=4,label='dx')
            # but this is the main loop
            for idx in range(self.N):
                axs[1,1].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].dq_eval[0,:],'k',markersize=4)
            # plot the waypoints
            for idx in range(self.N):
                axs[1,1].plot(self.bSols[idx].h_cps[0,:-1],self.bSols[idx].dq_cps[0,:],'ro')
                axs[1,1].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].dq_cps[0,0],'bo')
            axs[1,1].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].dq_cps[0,-1],'bo')
            axs[1,1].set_xlabel('t [-]')
            axs[1,1].set_ylabel('vel [m/s]')
            axs[1,1].legend()
            axs[1,1].set_title('velocity')
            axs[1,1].grid(True)

            if self.dim > 1:
                ### velocity y over time
                # plot the specification in the time-trajectory plot
                for idx,sp in enumerate(self.spec.preds):
                    sp.plot(axs[1,2],idx,"y","vel")
                axs[1,2].axhline(y=self.robot.dqub_eval[0][0],color='k',linestyle=':')
                axs[1,2].axhline(y=self.robot.dqlb_eval[0][0],color='k',linestyle=':')
                # adding these only for the legend
                axs[1,2].plot(self.bSols[0].h_eval[0,:],self.bSols[0].dq_eval[1,:],'k--',markersize=4,label='dy')
                # but this is the main loop
                for idx in range(self.N):
                    axs[1,2].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].dq_eval[1,:],'k--',markersize=4)
                # plot the waypoints
                for idx in range(self.N):
                    axs[1,2].plot(self.bSols[idx].h_cps[0,:-1],self.bSols[idx].dq_cps[1,:],'rv')
                    axs[1,2].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].dq_cps[1,0],'bv')
                axs[1,2].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].dq_cps[1,-1],'bv')
                axs[1,2].set_xlabel('t [-]')
                axs[1,2].set_ylabel('vel [m/s]')
                axs[1,2].legend()
                axs[1,2].set_title('velocity')
                axs[1,2].grid(True)

        if True:
            if self.dim > 1:
                ### acceleration
                for idx in range(self.N):
                    axs[2,0].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].ddq_eval[0,:],'k',markersize=4)
                    axs[2,0].plot(self.bSols[idx].h_eval[0,:],self.bSols[idx].ddq_eval[1,:],'k--',markersize=4)
                axs[2,0].axhline(y=self.robot.ddqub_eval[0][0],color='k',linestyle=':')
                axs[2,0].axhline(y=self.robot.ddqlb_eval[0][0],color='k',linestyle=':')
                # plot the waypoints
                for idx in range(self.N):
                    axs[2,0].plot(self.bSols[idx].h_cps[0,:-2],self.bSols[idx].ddq_cps[0,:],'ro')
                    axs[2,0].plot(self.bSols[idx].h_cps[0,:-2],self.bSols[idx].ddq_cps[1,:],'rv')
                    axs[2,0].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].ddq_cps[0,0],'bo')
                    axs[2,0].plot(self.bSols[idx].h_cps[0,0],self.bSols[idx].ddq_cps[1,0],'bv')
                axs[2,0].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].ddq_cps[1,-1],'bv')
                axs[2,0].plot(self.bSols[-1].h_cps[0,-1],self.bSols[-1].ddq_cps[0,-1],'bo')
                axs[2,0].set_xlabel('t [-]')
                axs[2,0].set_ylabel('u [m/s^2]')
                axs[2,0].legend(['ddx','ddy'])
                axs[2,0].set_title('acceleration')
                axs[2,0].grid(True)

            # time over [0,1]
            for idx in range(self.N):
                axs[2,1].plot(self.bSols[idx].evals,self.bSols[idx].h_eval[0,:],'k',markersize=4)
                # plot the waypoints
                axs[2,1].plot(np.linspace(0,1,self.n_cp),self.bSols[idx].h_cps[0,:],'ro')
                axs[2,1].plot(0,self.bSols[idx].h_cps[0,0],'bo')
                axs[2,1].plot(1,self.bSols[idx].h_cps[0,-1],'bo')
            axs[2,1].set_xlabel('s [-]')
            axs[2,1].set_ylabel('t [s]')
            axs[2,1].set_title('time over s')
            axs[2,1].grid(True)

            

            # dtime over [0,1]
            for idx in range(self.N):
                axs[2,2].plot(self.bSols[idx].evals,self.bSols[idx].dh_eval[0,:],'k',markersize=4)
                # plot the waypoints
                axs[2,2].plot(np.linspace(0,1,self.n_cp-1),self.bSols[idx].dh_cps[0,:],'ro')
                axs[2,2].plot(0,self.bSols[idx].dh_cps[0,0],'bo')
                axs[2,2].plot(1,self.bSols[idx].dh_cps[0,-1],'bo')
            axs[2,2].set_xlabel('s [-]')
            axs[2,2].set_ylabel('dt [s]')
            axs[2,2].set_title('dtime over s')
            axs[2,2].grid(True)

        if path is None:
            plt.show()
        else:
            plt.savefig(path,dpi=300)

    def divide_and_hull(self,path=None,cuts=0,plotting=False):
        # divides the convex hull into smaller convex hulls 
        # using De Casteljau's algorithm
        # cuts: number of cuts to be made (1 is divide into two etc.)
        if not self.evaluated:
            self.evaluate()

        r = self.robot.radius
        polyrobot = Polygon(np.array([[-r,r],
                                      [r,-r],
                                      [-r,-r],
                                      [r,r]]))
        
        hulls = [[] for i in range(cuts+1)]

        for idx in range(self.N):
            r1 = self.bSols[idx].r_cps.T
            h1 = self.bSols[idx].h_cps.T

            # get the index of middle time
            t_mid = 0.5*(h1[0]+h1[-1])
            _, s_mid = self.eval_t(t_mid)

            # get the default polygon, without cuts
            polyTraj = Polygon(r1)
            polySum = minkowski_sum(polyrobot,polyTraj)
            polySum_hull = polySum.convex_hull
            hulls[0].append(polySum_hull)

            for c in range(1,cuts+1):
                r1_1,r1_2,_,_ = divide_bezier_2(r1.T,h1.T,s_mid)

                polyTraj = Polygon(r1_1.T)
                polySum = minkowski_sum(polyrobot,polyTraj)
                polySum_hull = polySum.convex_hull
                hulls[c].append(polySum_hull)

                polyTraj = Polygon(r1_2.T)
                polySum = minkowski_sum(polyrobot,polyTraj)
                polySum_hull = polySum.convex_hull
                hulls[c].append(polySum_hull)

        ellipsoids = [[] for i in range(cuts+1)]
        for c in range(cuts+1):
            for i in range(len(hulls[c])):
                x = np.array(hulls[c][i].exterior.coords.xy)
                _,_,ellipse = MVEE_it(x.T)
                ellipsoids[c].append(ellipse)


        if plotting:
            # plot the trajectory r_eval and the convex hulls
            fig,axs = plt.subplots(1,cuts+1)

            for i_ax in range(cuts+1):
                # plot the world
                self.world.plot(axs[i_ax])
                # plot the trajectory
                for idx in range(self.N):
                    axs[i_ax].plot(self.bSols[idx].r_cps[0,:],self.bSols[idx].r_cps[1,:],'ro',markersize=4)
                    # the start and end control point in blue
                    axs[i_ax].plot(self.bSols[idx].r_cps[0,0],self.bSols[idx].r_cps[1,0],'bo',markersize=4)
                    axs[i_ax].plot(self.bSols[idx].r_cps[0,-1],self.bSols[idx].r_cps[1,-1],'bo',markersize=4)
                for idx in range(self.N):
                    axs[i_ax].plot(self.bSols[idx].r_eval[0,:],self.bSols[idx].r_eval[1,:],'k',markersize=4)
                axs[i_ax].grid(True)
                axs[i_ax].set_xlabel('x [m]')
                axs[i_ax].set_ylabel('y [m]')
                axs[i_ax].set_title('position')
                axs[i_ax].set_aspect('equal')    

            for c in range(cuts+1):
                for i in range(len(hulls[c])):
                    xp,yp = hulls[c][i].exterior.xy
                    axs[c].plot(xp,yp,'g',linewidth=1)
                    axs[c].add_patch(ellipsoids[c][i])
            
            if path is None:
                plt.show()
            else:
                plt.savefig(path,dpi=300)

