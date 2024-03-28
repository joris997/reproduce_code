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

from helpers.helpers_classes import Spec, Pred, MyPolygon, MySquare, waypointCBF, halfspaceCBF, \
    lineFunction, BezierSolution, BezierSolutions
from helpers.helpers_spline_opt import get_halfspace_polyhedral, get_derivative_control_points, Polygon, \
    get_derivative_control_points_gurobi, divide_bezier, divide_bezier_2, minkowski_sum, \
    get_derivative_control_points_drake
from helpers.helpers_functions import find_first_one, find_last_one, qual_AND_approx, qual_OR_approx
from helpers.MVEEn import MVEE_opt, MVEE_it

from planning.robustMILP2.STL_qualitative import qual_AND, qual_OR, qual_parse_operator, qual_MU
from planning.robustMILP2.bezier_STL_MILP import BezierSTL_MILP

from casadi import *

from pydrake.solvers import(
    MosekSolver,
    IpoptSolver,
    SnoptSolver,
    GurobiSolver,

    MathematicalProgram,
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    QuadraticCost,
    PerspectiveQuadraticCost,
)

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

class BezierSTL_NLP(BezierSTL_MILP):
    def __init__(self,spec,robot,N,world=[]):
        super(BezierSTL_NLP, self).__init__(spec,robot,N,world)

    def construct(self,bezier_order=5,add_stl=True):
        # use the same construction method, but don't add the STL constraints, we 
        # do those manually here in the LP
        self.order = bezier_order

        # set up the model
        self.prog = MathematicalProgram()

        # add continuous state and input variables
        self.r_var = []; self.h_var = []
        for _ in range(self.N):
            self.r_var.append(self.prog.NewContinuousVariables(self.dim,self.order))
            self.h_var.append(self.prog.NewContinuousVariables(1,self.order))

        self.dr_var = []; self.dh_var = []
        self.ddr_var = []; self.ddh_var = []
        for i in range(self.N):
            self.dr_var.append(get_derivative_control_points_drake(self.prog,self.r_var[i],1))
            self.dh_var.append(get_derivative_control_points_drake(self.prog,self.h_var[i],1))
            self.ddr_var.append(get_derivative_control_points_drake(self.prog,self.r_var[i],2))
            self.ddh_var.append(get_derivative_control_points_drake(self.prog,self.h_var[i],2))

        # continuity constraints
        for i in range(self.dim):
            for j in range(self.N-1):
                self.prog.AddLinearConstraint(self.r_var[j][i,-1] == self.r_var[j+1][i,0])
                self.prog.AddLinearConstraint(self.dr_var[j][i,-1] == self.dr_var[j+1][i,0])
                self.prog.AddLinearConstraint(self.ddr_var[j][i,-1] == self.ddr_var[j+1][i,0])
        
        for j in range(self.N-1):
            self.prog.AddLinearConstraint(self.h_var[j][0,-1] == self.h_var[j+1][0,0])
            self.prog.AddLinearConstraint(self.dh_var[j][0,-1] == self.dh_var[j+1][0,0])
            self.prog.AddLinearConstraint(self.ddh_var[j][0,-1] == self.ddh_var[j+1][0,0])

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

        # add the STL constraints
        if add_stl:
            self._stl_constraints()
        
        self._add_ellipse_constraints()

        # print('NumConVars: %d'%self.prog.getAttr('NumVars'))
        # print('NumBinVars: %d'%self.prog.getAttr('NumBinVars'))
        # print('NumIntVars: %d'%self.prog.getAttr('NumIntVars'))
        # print('NumConstr:  %d'%self.prog.getAttr('NumConstrs'))


    ###############
    ### SOLVING ###
    ###############
    def solve(self):
        # if self.initial_guess is None:
        #     self.initial_guess = np.zeros_like(self.vars)
        # self.prog.SetInitialGuess(self.vars,self.initial_guess)

        ### SOLVE
        # solver = IpoptSolver()
        solver = SnoptSolver()
        # solver = GurobiSolver()
        # solver = MosekSolver()

        t0 = time.time()
        self.result = solver.Solve(self.prog)
        print("Solved:       ", self.result.is_success())
        if not self.result.is_success():
            print("Ipopt solver status: ", self.result.get_solver_details().status, \
                  ", meaning ", self.result.get_solver_details().ConvertStatusToString())
        print("Solving took: ", time.time()-t0)
        try:
            infeasible = self.result.GetInfeasibleConstraints(self.prog)
            print('* infeasible constraints: ',len(infeasible))
            for c in infeasible:
                print("- ",c.evaluator().get_description())
        except:
            print("can't print constraint satisfaction")

    ############
    ### COST ###
    ############
    def add_cost(self):
        # self.prog.setObjective(0.01*self.objV - self.spec.theta_m, GRB.MINIMIZE)
        self.prog.AddCost(qual_AND_approx, vars=self.spec.theta_ms)


    ##########################
    ### LINEAR CONSTRAINTS ###
    ##########################
    ### initial
    def _initial_position_constraint(self):
        for i in range(self.dim):
            c = self.prog.AddLinearConstraint(self.r_var[0][i,0] == self.spec.q0[i])  
            c.evaluator().set_description(f"initial position constraint {i}")  
        
    def _initial_velocity_constraint(self):
        for i in range(self.dim):
            c = self.prog.AddLinearConstraint(self.dr_var[0][i,0] == self.dh_var[0][0,0]*self.spec.dq0[i])
            c.evaluator().set_description(f"initial velocity constraint {i}")  
    
    def _initial_time_constraint(self):
        c = self.prog.AddLinearConstraint(self.h_var[0][0,0] == self.spec.t0)
        c.evaluator().set_description(f"initial time constraint")  

    ### final
    def _final_position_constraint(self):
        for i in range(self.dim):
            c = self.prog.AddLinearConstraint(self.r_var[-1][i,-1] == self.spec.qf[i])
            c.evaluator().set_description(f"final position constraint {i}")  
        
    def _final_velocity_constraint(self):
        for i in range(self.dim):
            c = self.prog.AddLinearConstraint(self.dr_var[-1][i,-1] == self.dh_var[-1][0,-1]*self.spec.dqf[i])
            c.evaluator().set_description(f"final velocity constraint {i}")  
        
    def _final_time_constraint(self):
        c = self.prog.AddLinearConstraint(self.h_var[-1][0,-1] == self.spec.tf)
        c.evaluator().set_description(f"final time constraint")  

    def _positive_time_derivative_constraint(self):
        for j in range(self.order-1):
            for k in range(self.N):
                c = self.prog.AddLinearConstraint(self.dh_var[k][0,j] >= 1e-4)
                c.evaluator().set_description(f"dh_var lower bound {j} {k}")  
                c = self.prog.AddLinearConstraint(self.dh_var[k][0,j] <= 10)
                c.evaluator().set_description(f"dh_var upper bound {j} {k}")  

    ### others
    def _velocity_constraints(self):
        # for the beziers, we do the actuation limits already in the dynamics
        # for each dimension
        for i in range(self.dim):
            # inbetween each pair of nodes
            for j in range(self.N):
                # for each control point in ddx
                for k in range(self.order-1):
                    c = self.prog.AddLinearConstraint(self.dr_var[j][i,k] <= (self.robot.dqub_eval[i][0])*self.dh_var[j][0,k])
                    c.evaluator().set_description(f"velocity upper bound {i} {j} {k}")
                    c = self.prog.AddLinearConstraint((self.robot.dqlb_eval[i][0])*self.dh_var[j][0,k] <= self.dr_var[j][i,k])
                    c.evaluator().set_description(f"velocity lower bound {i} {j} {k}")

    def _stl_constraints(self):
        for pred in self.spec.preds:
            # create an array that is z_time_sol[i] AND z_space_sol[i]
            pred.z_time_space_sol = np.logical_and(pred.z_time_sol,pred.z_space_sol)
            # find the first index where z_time_sol is true, None if not existing
            idx0 = find_first_one(pred.z_time_space_sol)
            idxf = find_last_one(pred.z_time_space_sol)


            # add the temporal robustness variables
            if idx0 is not None:
                dt = self.tf - self.t0
                pred.theta_p = self.prog.NewContinuousVariables(1)#lb=-dt,ub=dt)
                pred.theta_m = self.prog.NewContinuousVariables(1)#lb=-dt,ub=dt)
                # lower and upper bound theta_p and theta_m
                self.prog.AddBoundingBoxConstraint(0,dt,pred.theta_p)
                self.prog.AddBoundingBoxConstraint(-dt,dt,pred.theta_m)

                if pred.type == "G":
                    # add the temporal robustness
                    c = self.prog.AddLinearConstraint(self.h_var[idx0][0,0] <= pred.range[0] - pred.theta_m[0])
                    c.evaluator().set_description(f"temporal robustness h_min {pred.get_string()}")
                    c = self.prog.AddLinearConstraint(self.h_var[idxf][0,-1] >= pred.range[1] + pred.theta_p[0])
                    c.evaluator().set_description(f"temporal robustness h_max {pred.get_string()}")
                elif pred.type == "F":
                    t_star = self.prog.NewContinuousVariables(1)#lb=pred.range[0],ub=pred.range[1])
                    c = self.prog.AddLinearConstraint(self.h_var[idx0][0,0] <= t_star[0] - pred.theta_m[0])
                    c.evaluator().set_description(f"temporal robustness h_min {pred.get_string()}")
                    c = self.prog.AddLinearConstraint(self.h_var[idxf][0,-1] >= t_star[0] + pred.theta_p[0])
                    c.evaluator().set_description(f"temporal robustness h_max {pred.get_string()}")
                    pred.t_star = t_star
                    
                

            # find the last index where z_time_sol is true, None if not existing
            for idx in range(self.N):
                # add the spatial constraint if z_space[idx] is true
                if pred.z_space_sol[idx]:
                    b = pred.area.get_b_eta(pred.io)
                    for cp in range(self.order):
                        ineqs = pred.area.H@self.r_var[idx][0:2,cp]
                        for face in range(pred.area.n_faces):
                            mu = ineqs[face] - b[face]
                            self.prog.AddLinearConstraint(mu <= 0)
            
                # add the temporal constraint if z_time[idx] is true
                if pred.z_time_sol[idx] and pred.z_space_sol[idx]:
                    if pred.z_time_type_sol[idx] == 1:
                        print("1 spec: ", pred.get_string())
                        self.prog.AddLinearConstraint(self.h_var[idx][0,0] <= pred.range[0])
                        self.prog.AddLinearConstraint(self.h_var[idx][0,-1] <= pred.range[1])
                        self.prog.AddLinearConstraint(self.h_var[idx][0,-1] >= pred.range[0])
                    elif pred.z_time_type_sol[idx] == 3:
                        print("3 spec: ", pred.get_string())
                        self.prog.AddLinearConstraint(self.h_var[idx][0,0] >= pred.range[0])
                        self.prog.AddLinearConstraint(self.h_var[idx][0,0] <= pred.range[1])
                        self.prog.AddLinearConstraint(self.h_var[idx][0,-1] >= pred.range[1])
                    elif pred.z_time_type_sol[idx] == 2:
                        print("2 spec: ", pred.get_string())
                        self.prog.AddLinearConstraint(self.h_var[idx][0,0] >= pred.range[0])
                        self.prog.AddLinearConstraint(self.h_var[idx][0,-1] <= pred.range[1])
        
        # now gather the switches, the negative- and positive temporal robustness from all predicates
        zs = self.spec.get_zs()
        theta_ms_tmp = self.spec.get_theta_ms()
        theta_ps_tmp = self.spec.get_theta_ps()
        # get them in the pydrake format
        theta_ms = self.prog.NewContinuousVariables(len(theta_ms_tmp))
        theta_ps = self.prog.NewContinuousVariables(len(theta_ps_tmp))
        for i in range(len(theta_ms_tmp)):
            self.prog.AddLinearConstraint(theta_ms[i] == theta_ms_tmp[i][0])
            self.prog.AddLinearConstraint(theta_ps[i] == theta_ps_tmp[i][0])

        self.spec.theta_ms = theta_ms
        self.spec.theta_ps = theta_ps

        # TODO: this still requires binary variables, but it shouldn't if we want to do non-convex
        #       gradient-based optimization
        theta_m = qual_AND_approx(theta_ms)
        theta_p = qual_AND_approx(theta_ps)
        self.spec.theta_m = theta_m
        self.spec.theta_p = theta_p
        
        # add to the world as well for later plotting
        for i in range(len(self.spec.preds)):
            self.world.add_objective(self.spec.preds[i])

    #############################
    ### NONLINEAR CONSTRAINTS ###
    #############################
    def _add_ellipse_constraints(self):
        Q = np.array([0.01,0,0,0.01]).reshape((2,2))
        mu = np.array([5.5,5.5])

        self.ellipse = MyEllipse(Q,mu)
        self.ellipse.set_t0_tf(6,9)

        # constrain the third bezier to adhere to these times so that we avoid
        idx = 2
        self.prog.AddLinearConstraint(self.r_var[idx][0,0] == self.ellipse.t0)
        self.prog.AddLinearConstraint(self.r_var[idx][0,-1] == self.ellipse.tf)

        lb = np.array([0])
        ub = np.array([1])

        ellipse_lambda = lambda x: (x-self.ellipse.mu).T@self.ellipse.Q@(x-self.ellipse.mu)
        for cp in range(self.order):
            x = self.r_var[idx][0:2,cp]
            self.prog.AddConstraint(ellipse_lambda,lb,ub,x)
            # self.prog.AddConstraint(self._ellipse_function,lb,ub,x)

    #######################
    ### POST-PROCESSING ###
    ####################### 
    def evaluate(self):
        for pred in self.spec.preds:
            if pred.add_theta and pred.z.X == 1:
                pred.z_space_sol = pred.z_space_sol # copy the solution from the previous MILP
                pred.z_time_sol = pred.z_time_sol 
                pred.z_sol = pred.z_sol
                pred.theta_m_sol = self.result.GetSolution(pred.theta_m)
                pred.theta_p_sol = self.result.GetSolution(pred.theta_p)
                pred.h_min_sol = pred.h_min_sol
                pred.h_max_sol = pred.h_max_sol
            else:
                pred.z_space_sol = np.zeros((self.N,1))
                pred.z_time_sol = np.zeros((self.N,1))
                pred.z_sol = 0
                pred.theta_m_sol = 0
                pred.theta_p_sol = 0
                pred.h_min_sol = 0
                pred.h_max_sol = 0
            
            if pred.type == "F":
                pred.t_star_sol = self.result.GetSolution(pred.t_star)
            else:
                pred.t_star_sol = 0
                
            # print thetap and thetap variable
            print("----------------------------------------------------")
            print(f"z_space: ", pred.z_space_sol.T)
            print(f"z_time:  ", pred.z_time_sol.T)
            print(f"z:       ", pred.z_sol)
            print(f"t_star:  ", pred.t_star_sol)
            print(f"h_min: {pred.h_min_sol} \t h_max: {pred.h_max_sol}")
            print(f"thetam {pred.get_string()}: ", pred.theta_m_sol)
            print(f"thetap {pred.get_string()}: ", pred.theta_p_sol)

        # get the bezier control points for r, h, and q
        self.r_cps_sol = np.zeros((self.dim,self.order,self.N))
        for i in range(self.N):
            self.r_cps_sol[:,:,i] = self.result.GetSolution(self.r_var[i])
        self.dr_cps_sol = get_derivative_control_points_gurobi(self.r_cps_sol,1)
        self.ddr_cps_sol = get_derivative_control_points_gurobi(self.r_cps_sol,2)

        self.h_cps_sol = np.zeros((1,self.order,self.N))
        for i in range(self.N):
            self.h_cps_sol[:,:,i] = self.result.GetSolution(self.h_var[i])
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


        # type of temporal constraint
        for pred in self.spec.preds:
            # now we need to check fo all z_time_sol = True, what kind overlap we have with the pred.range
            pred.z_time_type_sol = np.zeros((self.N,1))
            for idx in range(self.N):
                if pred.z_space_sol[idx] == 1:
                    if self.h_cps_sol[0,0,idx] <= pred.range[0] and self.h_cps_sol[0,-1,idx] <= pred.range[-1]:
                        # we start before but end within the time-range
                        pred.z_time_type_sol[idx] = 1
                    elif self.h_cps_sol[0,0,idx] >= pred.range[0] and self.h_cps_sol[0,-1,idx] >= pred.range[-1]:
                        # we start within but end after the time-range
                        pred.z_time_type_sol[idx] = 3
                    elif self.h_cps_sol[0,0,idx] >= pred.range[0] and self.h_cps_sol[0,-1,idx] <= pred.range[-1]:
                        # we are within the time-range
                        pred.z_time_type_sol[idx] = 2


        # now gather all the spline results, similar to the MultiSplineOptState
        self.bSols = []
        for i in range(self.N):
            self.bSols.append(BezierSolution(self,i))

        for i in range(self.N):
            self.bSols[i].generate_eval()

        # now check all the switches and add it to the specification so it can be used in an LP
        self.eval_switches()

        self.evaluated = True