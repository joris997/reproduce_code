import copy
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

from worlds.square_world import World
from helpers.helpers_classes import Spec, Pred, Mu, MyPolygon, MySquare, My1DBound, \
    waypointCBF, halfspaceCBF, lineFunction, BezierSolution, BezierSolutions
from helpers.helpers_spline_opt import get_halfspace_polyhedral, get_derivative_control_points, Polygon, \
    get_derivative_control_points_gurobi, divide_bezier, divide_bezier_2, minkowski_sum
from helpers.MVEEn import MVEE_opt, MVEE_it

from planning.robustMILP2.STL_qualitative import qual_AND, qual_OR, qual_parse_operator, qual_MU

from robot.SingleIntegrator import SingleIntegrator


def get_stl_spec(num=1,N=None,order=None):
    spec = Spec()

    ### ORDER OF THE BEZIER CURVES
    # 1: linear, 2 cp
    # 2: quadratic, 3 cp
    # 3: cubic, 4 cp
    # 4: quartic, 5 cp
    # 5: quintic, 6 cp
    # etc.

    if num == 1:
        spec.set_t0(0)
        spec.set_tf(13)

        # Beziers
        spec.set_N(7)
        spec.set_order(4)

        # Simple example specification
        goal1 = MySquare([2,2],0.5)
        goal2 = MySquare([6,9],0.5)
        goal3 = MySquare([4.8,10.1],0.5)

        spec.add_pred(Pred('G', [4,6],  Mu([goal1],["in"],["pos"])))
        spec.add_pred(Pred('F', [8,10], Mu([goal2],["in"],["pos"])))
        spec.add_pred(Pred('F', [9,11], Mu([goal3],["in"],["pos"])))

        spec.set_discon("AND")

        robot = SingleIntegrator(spec.tf,spec.N,v_max=2)

        world_size = MySquare([5,5],6.5)
        world = World()
        world.add_world_size(world_size)

    elif num == 2:
        spec.set_t0(0)
        spec.set_tf(60)

        # Beziers
        spec.set_N(22)
        spec.set_order(3)

        # Maximum complex example
        goals_upper = []
        goals_lower = []
        for i in range(4):
            goals_upper.append(MySquare([2+1.5*i,5],0.5))
            goals_lower.append(MySquare([2+1.5*i,2],0.5))
        
        for i in range(len(goals_upper)):
            spec.add_pred(Pred('F', [10*i,4.9+10*i],     Mu([goals_upper[i]],["in"],["pos"])))
            spec.add_pred(Pred('F', [5+10*i,5+4.9+10*i], Mu([goals_lower[i]],["in"],["pos"])))

        spec.set_discon("AND")

        robot = SingleIntegrator(spec.tf,spec.N,v_max=2)

        world_size = MySquare([5,5],6.5)
        world = World()
        world.add_world_size(world_size)

    elif num == 3:
        spec.set_t0(0)
        spec.set_tf(13)

        # Beziers
        spec.set_N(3)
        spec.set_order(4)

        goal1 = MySquare([2,2],0.5,name="goal1")
        posc1 = My1DBound(7,8,"x",name="posc1")
        velc1 = My1DBound(0.5,1,"y",name="velc1")

        spec.add_pred(Pred('G', [3,4],  Mu([goal1],["in"],["pos"])))
        spec.add_pred(Pred("G", [8,10], Mu([posc1],["in"],["pos"])))
        spec.add_pred(Pred("G", [5,6],  Mu([velc1],["in"],["vel"])))

        spec.set_discon("AND")
        
        robot = SingleIntegrator(spec.tf,spec.N,v_max=2)

        world_size = MySquare([5,5],6.5)
        world = World()
        world.add_world_size(world_size)

    elif num == 4:
        # Case Study 1: UAV Altitude Control
        if N is None:
            N = 5
        if order is None:
            order = 4
        
        spec.set_t0(0)
        spec.set_tf(100)

        # Beziers
        spec.set_N(N)
        spec.set_order(order)

        spec.set_dim(1)
        posc1 = My1DBound(20,40,"x",dim=1)
        posc2 = My1DBound(0,10, "x",dim=1)

        spec.add_pred(Pred("G", [20,30], Mu([posc1],["in"],["pos"])))
        spec.add_pred(Pred("G", [60,70], Mu([posc2],["in"],["pos"])))

        spec.set_discon("AND")

        robot = SingleIntegrator(spec.tf,spec.N,v_max=1.5,u_max=0.2)

        world_size = MySquare([5,5],6.5)
        world = World()
        world.add_world_size(world_size)
    
    elif num == 5:
        # Case Study 2: Single-agent surveillance
        if N is None:
            N = 6
        if order is None:
            order = 3

        spec.set_t0(0)
        spec.set_tf(60)
        spec.set_q0([11,11])
        spec.set_qf([12,9])

        # Beziers
        spec.set_N(N)
        spec.set_order(order)

        # Goals
        charge1 = MySquare([11.5,9.5],0.5)
        charge2 = MySquare([3.5,5.5],0.5)
        goal = MySquare([7.5,7.5],2.5)

        gs_mu = Mu([goal],["in"],["pos"])
        spec.add_pred(Pred("F",[0,20], gs_mu))        

        spec.set_discon("AND")

        robot = SingleIntegrator(spec.tf,spec.N,v_max=1.5,u_max=0.2)

        world_size = MySquare([7.5,7.5],6.5)
        world = World()
        world.add_world_size(world_size)

    elif num == 6:
        # Case Study 2: Multi-agent surveillance
        if N is None:
            N = 8
        if order is None:
            order = 3

        spec.set_t0(0)
        spec.set_tf(60)
        spec.set_q0([11,11, 4,4])
        spec.set_qf([12,9, 3,5])
        spec.set_dq0([0,0, 0,0])
        spec.set_dqf([0,0, 0,0])

        # Beziers
        spec.set_N(N)
        spec.set_order(order)
        spec.set_n_agents(2)

        # Goals
        charge1 = MySquare([11.5,9.5],0.5, name="charge1")
        charge2 = MySquare([3.5,5.5],0.5,  name="charge2")
        goal = MySquare([7.5,7.5],2.5,     name="goal")

        # \phi_gs
        gs_mu = Mu([goal,goal],["in","in"],["pos","pos"],[0,1],"OR")
        spec.add_pred(Pred("F",[0,20],  gs_mu))
        spec.add_pred(Pred("F",[20,40], gs_mu))

        # # \phi_ch,1
        # ch1_mu = Mu([charge1],["in"],["pos"])
        # spec.add_pred(Pred("G",[0,20], Pred("F",[0,20], Pred("G",[0,20], ch1_mu))))

        # \phi_ch,2
        ch2_mu = Mu([charge2],["in"],["pos"],[1])
        spec.add_pred(Pred("F",[0,20],  ch2_mu, add_sr=True))
        spec.add_pred(Pred("F",[40,55], ch2_mu, add_sr=True))     

        spec.set_discon("AND")

        robot = SingleIntegrator(spec.tf,spec.N,v_max=1.5,u_max=0.2,dim=4)

        world_size = MySquare([7.5,7.5],6.5)
        world = World()
        world.add_world_size(world_size)

    elif num == 7:
        # Case Study 2: Multi-agent surveillance
        if N is None:
            N = 8
        if order is None:
            order = 3

        spec.set_t0(0)
        spec.set_tf(60)
        spec.set_q0([0,0])
        spec.set_qf([0,0])
        spec.set_dq0([0,0])
        spec.set_dqf([0,0])

        # Beziers
        spec.set_N(N)
        spec.set_order(order)

        # Goals
        mu1 = MySquare([11.5,9.5],0.5, name="charge1")
        mu2 = MySquare([3.5,5.5], 0.5, name="charge2")

        # s1 = Pred("G",[0,20],[ Pred("F",[5,10],[ Pred("OR",[],[mu1,mu2]) ]) ])
        # s2 = Pred("F",[0,20],[ Pred("G",[5,10],[ Pred("AND",[],[mu1,mu2]) ]) ])

        # tot = Pred("OR",[0,60],[s1,s2])
        tot = Pred("F",[20,30],[mu1])
        spec.add_spec(tot)

        robot = SingleIntegrator(spec.tf,spec.N,v_max=1.5,u_max=0.2,dim=2)

        world_size = MySquare([7.5,7.5],6.5)
        world = World()
        world.add_world_size(world_size)

    return spec, robot, world



