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

from helpers.helpers_classes import Spec, Pred, MyPolygon, MySquare, waypointCBF, halfspaceCBF, lineFunction, BezierSolution, BezierSolutions
from helpers.helpers_spline_opt import get_halfspace_polyhedral, get_derivative_control_points, Polygon, \
    get_derivative_control_points_gurobi, divide_bezier, divide_bezier_2, minkowski_sum
from helpers.helpers_functions import find_first_one, find_last_one
from helpers.MVEEn import MVEE_opt, MVEE_it

from planning.robustMILP2.STL_qualitative import qual_AND, qual_OR, qual_parse_operator, qual_MU
from planning.robustMILP2.bezier_STL_MILP import BezierSTL_MILP

from casadi import *

class BezierSTL_LP(BezierSTL_MILP):
    def __init__(self,spec,robot,N,world=[]):
        super(BezierSTL_LP, self).__init__(spec,robot,N,world)

    def construct(self,bezier_order=5):
        # use the same construction method, but don't add the STL constraints, we 
        # do those manually here in the LP
        super().construct(bezier_order=bezier_order,add_stl=False)

        # add the STL constraints
        self._stl_constraints()

        self.prog.update()

        print('NumConVars: %d'%self.prog.getAttr('NumVars'))
        print('NumBinVars: %d'%self.prog.getAttr('NumBinVars'))
        print('NumIntVars: %d'%self.prog.getAttr('NumIntVars'))
        print('NumConstr:  %d'%self.prog.getAttr('NumConstrs'))

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
                pred.theta_p = self.prog.addVar(lb=-dt,ub=dt,vtype=GRB.CONTINUOUS)
                pred.theta_m = self.prog.addVar(lb=-dt,ub=dt,vtype=GRB.CONTINUOUS)
                if pred.type == "G":
                    # add the temporal robustness
                    self.prog.addConstr(self.h_var[0,0,idx0] <= pred.range[0] - pred.theta_m,
                                        name=f"temporal robustness h_min {pred.get_string()}")
                    self.prog.addConstr(self.h_var[0,-1,idxf] >= pred.range[1] + pred.theta_p,
                                        name=f"temporal robustness h_max {pred.get_string()}")
                elif pred.type == "F":
                    t_star = self.prog.addVar(lb=pred.range[0],ub=pred.range[1],vtype=GRB.CONTINUOUS)
                    self.prog.addConstr(self.h_var[0,0,idx0] <= t_star - pred.theta_m,
                                        name=f"temporal robustness h_min {pred.get_string()}")
                    self.prog.addConstr(self.h_var[0,-1,idxf] >= t_star + pred.theta_p,
                                        name=f"temporal robustness h_max {pred.get_string()}")
                    pred.t_star = t_star
                    
                

            # find the last index where z_time_sol is true, None if not existing
            for idx in range(self.N):
                # add the spatial constraint if z_space[idx] is true
                if pred.z_space_sol[idx]:
                    b = pred.area.get_b_eta(pred.io)
                    for cp in range(self.order):
                        ineqs = pred.area.H@self.r_var[0:2,cp,idx]
                        for face in range(pred.area.n_faces):
                            mu = ineqs[face] - b[face]
                            self.prog.addConstr(mu <= 0)
            
                # add the temporal constraint if z_time[idx] is true
                if pred.z_time_sol[idx] and pred.z_space_sol[idx]:
                    if pred.z_time_type_sol[idx] == 1:
                        print("1 spec: ", pred.get_string())
                        self.prog.addConstr(self.h_var[0,0,idx] <= pred.range[0])
                        self.prog.addConstr(self.h_var[0,-1,idx] <= pred.range[1])
                        self.prog.addConstr(self.h_var[0,-1,idx] >= pred.range[0])
                    elif pred.z_time_type_sol[idx] == 3:
                        print("3 spec: ", pred.get_string())
                        self.prog.addConstr(self.h_var[0,0,idx] >= pred.range[0])
                        self.prog.addConstr(self.h_var[0,0,idx] <= pred.range[1])
                        self.prog.addConstr(self.h_var[0,-1,idx] >= pred.range[1])
                    elif pred.z_time_type_sol[idx] == 2:
                        print("2 spec: ", pred.get_string())
                        self.prog.addConstr(self.h_var[0,0,idx] >= pred.range[0])
                        self.prog.addConstr(self.h_var[0,-1,idx] <= pred.range[1])
        
        # now gather the switches, the negative- and positive temporal robustness from all predicates
        zs = self.spec.get_zs()
        theta_ms = self.spec.get_theta_ms()
        theta_ps = self.spec.get_theta_ps()

        # TODO: this still requires binary variables, but it shouldn't if we want to do non-convex
        #       gradient-based optimization
        z,theta_m,theta_p = qual_AND(self,[],theta_ms,theta_ps)
        self.spec.z = z
        self.spec.theta_m = theta_m
        self.spec.theta_p = theta_p
        
        # add to the world as well for later plotting
        for i in range(len(self.spec.preds)):
            self.world.add_objective(self.spec.preds[i])
