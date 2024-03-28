from gurobipy import Model
import gurobipy as gp
from gurobipy import GRB
from helpers.helpers_classes import MyPolygon, Pred

def quant_parse_operator(obj,pred):
    # INPUT:  obj (MultiSplineOptMILPhr), contains the optimization program 'prog'
    #         pred (Pred), contains the area, the time bound, the type of predicate
    print("type: ", type(pred).__name__)
    if type(pred).__name__ == "Pred":
        # if pred.mus is another Pred, we do recursion
        print("mus type: ", type(pred.preds[0]).__name__)
        if type(pred.preds[0]).__name__ == "Pred":
            for mu in pred.preds:
                quant_parse_operator(obj,mu)
        # otherwise we have a list of Mu's 
        else:
            for mu in pred.preds:
                quant_mu(obj,mu)
    else:
        print("ERROR: quant_parse_operator: pred is not a Pred")

    print("pred: ", pred.get_string())
    quant_pred(obj,pred)


def qual_AND_z1_z2(obj,pred,z1,z2):
    z_and = obj.prog.addMVar((obj.N,),vtype=GRB.BINARY)
    for idx in range(obj.N):
        # c = b1 AND b2
        obj.prog.addConstr(z1[idx] + z2[idx] - 1 <= z_and[idx],
                           name=f"AND z_and sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(z_and[idx] <= z1[idx],
                           name=f"AND z_and z_space inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(z_and[idx] <= z2[idx],
                           name=f"AND z_and z_time inequality {pred.get_string()} {idx}")
    return z_and
    
def qual_OR_z1_z2(obj,pred,z1,z2):
    z_or = obj.prog.addMVar((obj.N,),vtype=GRB.BINARY)
    for idx in range(obj.N):
        # c = b1 OR b2
        obj.prog.addConstr(z_or[idx] <= z1[idx] + z2[idx],
                           name=f"OR z_or sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(z1[idx] <= z_or[idx],
                           name=f"OR z_or z_space inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(z2[idx] <= z_or[idx],
                           name=f"OR z_or z_time inequality {pred.get_string()} {idx}")
    return z_or

def qual_EXISTS_z1_z2(obj,pred,z1,z2):
    # boolean value that indicates whether the ALWAYS operator holds or not
    z_exists = obj.prog.addVar(vtype=GRB.BINARY)

    # now we have an array where at least one z[i] is true,
    # for that one, for EVENTUALLY satisfaction, we don't need to expand the bounds
    pred.cs = obj.prog.addMVar((obj.N,),vtype=GRB.BINARY)
    for idx in range(obj.N):
        # c = b1 AND b2
        obj.prog.addConstr(z1[idx] + z2[idx] - 1 <= pred.cs[idx],
                           name=f"OP cs sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(pred.cs[idx] <= z1[idx],
                           name=f"OP cs z_space inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(pred.cs[idx] <= z2[idx],
                           name=f"OP cs z_time inequality {pred.get_string()} {idx}")
        
    obj.prog.addConstr(z_exists <= gp.quicksum(pred.cs),
                       name=f"OP cs sum lower bound {pred.get_string()}")
    obj.prog.addConstr(gp.quicksum(pred.cs) <= obj.N*z_exists,
                       name=f"OP cs sum upper bound {pred.get_string()}")
    
    return z_exists

def qual_NOT_z(obj,pred,z):
    z_not = obj.prog.addMVar((obj.N,),vtype=GRB.BINARY)
    for idx in range(obj.N):
        # c = b1 AND b2
        obj.prog.addConstr(z[idx] + z_not[idx] == 1,
                           name=f"NOT z_not sum inequality {pred.get_string()} {idx}")
    return z_not

def parse_space(obj,mu):
    z_space = obj.prog.addMVar((obj.N,1),vtype=GRB.BINARY)
    # INPUT:  obj (MultiSplineOptMILPhr), contains the optimization program 'prog'
    #         pred (Pred), contains the area, the time bound, the type of predicate
    #
    # OUTPUT: z_var (Binarys), indicates whether the Bezier adheres to the area of the spec
    idx_dim_0 = 0*obj.dim
    idx_dim_f = idx_dim_0 + obj.dim

    io = mu.io
    pv = mu.pv

    b = mu.get_b_eta()
    n_faces = mu.n_faces

    # determine how much control points we need to consider
    # given the pred.pv of the predicate
    if pv == "pos":
        n_cp = obj.n_cp
    elif pv == "vel":
        n_cp = obj.n_cp-1

    for idx in range(obj.N):
        # b = obj.prog.addMVar((n_cp,n_faces),vtype=GRB.BINARY)
        for cp in range(n_cp):
            for face in range(n_faces):
                if pv == "vel":
                    ineqs = mu.H@obj.dr_var[idx_dim_0:idx_dim_f,cp,idx]
                elif pv == "pos":
                    ineqs = mu.H@obj.r_var[idx_dim_0:idx_dim_f,cp,idx]
                c = ineqs[face] - b[face]

                obj.prog.addConstr(c <= (obj.bigM)*(1-z_space[idx]),
                                    name=f"MU stay-in {idx} {cp} {face}")
    return z_space

def parse_time(obj,pred,I):
    # parse the time variables from the pred
    z_time = obj.prog.addMVar((obj.N,1),vtype=GRB.BINARY)

    ### first of the idx1 and last control point of the idx 1
    # so we decide to have at least Bezier curve for which the first control point is before
    # and the last control point is after
    for idx in range(obj.N):
        # # at zero there are some numerical issues, so we add a small epsilon
        b1 = obj.prog.addVar(vtype=GRB.BINARY)
        b2 = obj.prog.addVar(vtype=GRB.BINARY)
            
        # if obj.h_var[0,-1,idx] >= pred.range[1] then b1 = 1
        obj.prog.addConstr(obj.h_var[0,-1,idx] >= I[1] - obj.bigM*(1-b1),
                            name=f"conditional constraint {pred.get_string()} {idx} 1")
        obj.prog.addConstr(obj.h_var[0,-1,idx] <= I[1] + obj.bigM*b1,
                            name=f"conditional constraint {pred.get_string()} {idx} 1")
        
        # if obj.h_var[0,0,idx] <= pred.I[0] then b2 = 1
        obj.prog.addConstr(obj.h_var[0,0,idx] <= I[0] + obj.bigM*(1-b2),
                            name=f"conditional constraint {pred.get_string()} {idx} 2")
        obj.prog.addConstr(obj.h_var[0,0,idx] >= I[0] - obj.bigM*b2,
                            name=f"conditional constraint {pred.get_string()} {idx} 2")

        c = obj.prog.addVar(vtype=GRB.BINARY)
        # c = b1 AND b2
        obj.prog.addConstr(b1 + b2 - 1 <= c,
                            name=f"parse_time c sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(c <= b1,
                            name=f"parse_time c I[0] bound {pred.get_string()} {idx}")
        obj.prog.addConstr(c <= b2,
                            name=f"parse_time c I[1] bound {pred.get_string()} {idx}")

        # now if z_time[idx] == 1, the first and last control points are inside the time range
        obj.prog.addConstr(z_time[idx] == c,
                            name=f"parse_time z_time = c {pred.get_string()} {idx}")
    return z_time

def parse_hmin_hmax(obj,pred,key):
    # get the first index for which the variable z_time[idx] is equal to 1
    h_0_array = obj.prog.addMVar((obj.N,1),lb=obj.t0,ub=obj.tf+obj.bigM,vtype=GRB.CONTINUOUS)
    h_f_array = obj.prog.addMVar((obj.N,1),lb=-obj.t0-obj.bigM,ub=obj.tf,vtype=GRB.CONTINUOUS)
    for idx in range(obj.N):
        # CONSTRAINT: we now constrain that if z_space holds, then we add it to the timed constraint
        obj.prog.addConstr(h_0_array[idx] == obj.h_var[0,0,idx] + obj.bigM*(1-key[idx]),
                           name=f"temporal robustness h_0_array {pred.get_string()} {idx}")
        obj.prog.addConstr(h_f_array[idx] == obj.h_var[0,-1,idx] - obj.bigM*(1-key[idx]),
                           name=f"temporal robustness h_f_array {pred.get_string()} {idx}")

    h_min = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    h_max = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    pred.h_min = h_min
    pred.h_max = h_max
    h_0_array_ = [h_0_array[idx] for idx in range(obj.N)]
    h_f_array_ = [h_f_array[idx] for idx in range(obj.N)]

    # CONSTRAINT: h_min and h_max as the minimum and maximum time of the bezier curves that adhere to the timed constraints
    obj.prog.addConstr(h_min == gp.min_(h_0_array_),
                       name=f"temporal robustness h_min {pred.get_string()}")
    obj.prog.addConstr(h_max == gp.max_(h_f_array_),
                       name=f"temporal robustness h_max {pred.get_string()}")

    

def quant_mu(obj,mu,I=[]):
    mu.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    mu.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    if I == []:
        I = [obj.t0,obj.tf]

    mu.z_space = parse_space(obj,mu)
    mu.z_time = parse_time(obj,mu,I)
    zAND = qual_AND_z1_z2(obj,mu,mu.z_space,mu.z_time)
    mu.z = zAND
    parse_hmin_hmax(obj,mu,zAND)

    obj.prog.addConstr(mu.theta_p == mu.h_max - mu.h_min)
    obj.prog.addConstr(mu.theta_m == mu.h_max - mu.h_min)


def quant_AND(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    zAND = qual_AND_z1_z2(obj,pred,pred.preds[0].z,pred.preds[1].z)
    pred.z = zAND

    obj.prog.addConstr(pred.theta_p == gp.min_(theta_ps))
    obj.prog.addConstr(pred.theta_m == gp.min_(theta_ms))

def quant_OR(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    zOR = qual_OR_z1_z2(obj,pred,pred.preds[0].z,pred.preds[1].z)
    pred.z = zOR

    obj.prog.addConstr(pred.theta_p == gp.max_(theta_ps))
    obj.prog.addConstr(pred.theta_m == gp.max_(theta_ms))

def quant_NOT(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    zNOT = qual_NOT_z(obj,pred,pred.preds[0].z)
    pred.z = zNOT

    obj.prog.addConstr(pred.theta_p == -theta_ms[0])
    obj.prog.addConstr(pred.theta_m == -theta_ps[0])

def quant_UNTIL(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    # we now have an interval I
    t1 = obj.prog.addVar(lb=0,ub=obj.tf,vtype=GRB.CONTINUOUS)
    # the time is inbetween the interval
    obj.prog.addConstr(pred.I[0] <= t1)
    obj.prog.addConstr(t1 <= pred.I[1])
    # but should also be limited to the last control points of h
    b = obj.prog.addMVar((obj.N,1),lb=0,ub=1,vtype=GRB.BINARY)
    for idx in range(obj.N):
        # TODO: this can be inbetween h_var[0,0,idx] and h_var[0,-1,idx]
        # b = 1 if t1 is equal to obj.h_var[0,-1,idx]
        obj.prog.addConstr(t1 - obj.h_var[0,-1,idx] <= obj.bigM*(1-b[idx]))
        obj.prog.addConstr(obj.h_var[0,-1,idx] - t1 <= obj.bigM*(1-b[idx]))
    obj.prog.addConstr(gp.quicksum(b) == 1)

    z_time2 = parse_time(obj,pred,[pred.I[0],t1])
    z_AND2 = qual_AND_z1_z2(obj,pred,pred.preds[0].z,z_time2)
    theta_p_2 = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    theta_m_2 = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    obj.prog.addConstr(theta_p_2 == gp.min_(theta_ps[0]))
    obj.prog.addConstr(theta_m_2 == gp.min_(theta_ms[0]))

    z_AND1 = qual_AND_z1_z2(obj,pred,pred.preds[1].z,b)
    theta_p_1 = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    theta_m_1 = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    obj.prog.addConstr(theta_p_1 == gp.min_(theta_ps[1]))
    obj.prog.addConstr(theta_m_1 == gp.min_(theta_ms[1]))

    zUNTIL = qual_OR_z1_z2(obj,pred,z_AND1,z_AND2)
    pred.z = zUNTIL
    obj.prog.addConstr(pred.theta_p == gp.max_(theta_p_1,theta_p_2))
    obj.prog.addConstr(pred.theta_m == gp.max_(theta_m_1,theta_m_2))

def quant_EVENTUALLY(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    # we now have an interval I
    # push in a TRUE predicate
    pred.preds.insert(0,(Pred("TRUE")))
    quant_UNTIL(obj,pred,theta_ps,theta_ms)

def quant_ALWAYS(obj,pred,theta_ps,theta_ms):
    pred.theta_p = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-obj.dt,ub=obj.dt,vtype=GRB.CONTINUOUS)

    # take the NOT of preds[0]
    quant_NOT(obj,pred,pred.preds[0].theta_p,pred.preds[0].theta_m)
    # 
    


def quant_pred(obj,pred):
    theta_ps = []
    theta_ms = []
    for p in pred.preds:
        theta_ps.append(p.theta_p)
        theta_ms.append(p.theta_m)

    if pred.type == "AND":
        quant_AND(obj,pred,theta_ps,theta_ms)
    
    elif pred.type == "OR":
        quant_OR(obj,pred,theta_ps,theta_ms)

    elif pred.type == "NOT":
        quant_NOT(obj,pred,theta_ps,theta_ms)

    elif pred.type == "U":
        quant_UNTIL(obj,pred,theta_ps,theta_ms)

    elif pred.type == "F":
        quant_EVENTUALLY(obj,pred,theta_ps,theta_ms)
    
    elif pred.type == "G":
        quant_ALWAYS(obj,pred,theta_ps,theta_ms)
