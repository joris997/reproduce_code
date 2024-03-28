from gurobipy import Model
import gurobipy as gp
from gurobipy import GRB
from helpers.helpers_classes import MyPolygon


def qual_AND(obj,zs,theta_ms=None,theta_ps=None):
    # INPUT:  z ([Binaries]), a list that indicates whether underlying specs hold true or not
    #
    # OUTPUT: z_var (Binary), the AND of the robustness values, indicating all is true

    # z_var indicates true if all z_space and z_time are true
    z_var = obj.prog.addVar(vtype=GRB.BINARY)

    # TODO: this can be removed and lowers the constraints, it's just for printing afterwards
    for i in range(len(zs)):
        # constrain it to be true
        obj.prog.addConstr((z_var == 1) >> (zs[i] == 1),
                           name=f"AND_zs {i}")
    obj.prog.addConstr(z_var == 1,
                       name=f"AND_z_var")
    
    if theta_ms == None or theta_ps == None:
        return z_var

    theta_m_var = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    theta_p_var = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    # get the minimum value of the theta_ms
    obj.prog.addConstr(theta_m_var == gp.min_(theta_ms),
                       name=f"AND_theta_m_var")

    # get the minimum value of the theta_ps
    obj.prog.addConstr(theta_p_var == gp.min_(theta_ps),
                       name=f"AND_theta_p_var")
    
    return z_var,theta_m_var,theta_p_var

def qual_OR(obj,zs,theta_ms=None,theta_ps=None):
    # INPUT:  zs ([Binaries]), a list that indicates whether underlying specs hold true or not
    #         theta_ms ([Cont.]), a list that indicates the negative temporal robustness
    #         htetaps ([Cont.]), a list that indicates the positvie temporal robustness
    #
    # OUTPUT: z_var (Binary), the OR of the robustness values, indicating at least 1 is true

    # z_var indicates true if all z_space and z_time are true
    z_var = obj.prog.addMVar((len(zs),),vtype=GRB.BINARY)

    # OR so use bigM notation to constrain one if z_var is true
    # TODO: this can be removed and lowers the constraints, it's just for printing afterwards
    for i in range(len(zs)):
        obj.prog.addConstr(zs[i] == z_var[i],
                           name=f"OR_zs {i}")

    obj.prog.addConstr(gp.quicksum(z_var) == 1,
                       name=f"OR_z_var")
    
    if theta_ms == None or theta_ps == None:
        return z_var
    
    theta_m_var = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    theta_p_var = obj.prog.addVar(vtype=GRB.CONTINUOUS)
    # get the minimum value of the theta_ms
    obj.prog.addConstr(theta_m_var == gp.max_(theta_ms),
                       name=f"OR_theta_m_var")

    # get the minimum value of the theta_ps
    obj.prog.addConstr(theta_p_var == gp.max_(theta_ps),
                       name=f"OR_theta_p_var")

    return z_var,theta_m_var,theta_p_var


def qual_parse_operator(obj,pred):
    # INPUT:  obj (MultiSplineOptMILPhr), contains the optimization program 'prog'
    #         pred (Pred), contains the area, the time bound, the type of predicate
    if type(pred).__name__ == "Pred":
        qual_parse_operator(obj,pred.mu)
    elif type(pred).__name__ == "Mu":
        qual_MU(obj,pred)
        return

    parse_time(obj,pred)
    add_temporal_robustness(obj,pred)
    qual_space_and_time(obj,pred)


def qual_space_and_time(obj,pred):
    # boolean value that indicates whether the ALWAYS operator holds or not
    pred.z = obj.prog.addVar(vtype=GRB.BINARY)

    # now we have an array where at least one z[i] is true,
    # for that one, for EVENTUALLY satisfaction, we don't need to expand the bounds
    pred.cs = obj.prog.addMVar((obj.N,),vtype=GRB.BINARY)
    for idx in range(obj.N):
        # c = b1 AND b2
        obj.prog.addConstr(pred.mu.z_space[idx] + pred.z_time[idx] - 1 <= pred.cs[idx],
                           name=f"OP cs sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(pred.cs[idx] <= pred.mu.z_space[idx],
                           name=f"OP cs z_space inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(pred.cs[idx] <= pred.z_time[idx],
                           name=f"OP cs z_time inequality {pred.get_string()} {idx}")
        
    obj.prog.addConstr(pred.z <= gp.quicksum(pred.cs),
                       name=f"OP cs sum lower bound {pred.get_string()}")
    obj.prog.addConstr(gp.quicksum(pred.cs) <= obj.N*pred.z,
                       name=f"OP cs sum upper bound {pred.get_string()}")


def qual_MU(obj,mu):
    # INPUT:  obj (MultiSplineOptMILPhr), contains the optimization program 'prog'
    #         pred (Pred), contains the area, the time bound, the type of predicate
    #
    # OUTPUT: z_var (Binarys), indicates whether the Bezier adheres to the area of the spec
    for idx_area in range(len(mu.areas)):
        idx_dim_0 = mu.agents[idx_area]*obj.dim
        idx_dim_f = idx_dim_0 + obj.dim

        area = mu.areas[idx_area]
        io = mu.ios[idx_area]
        pv = mu.pvs[idx_area]

        b = area.get_b_eta(io)
        n_faces = area.n_faces

        area.z_space = obj.prog.addMVar((obj.N,1),vtype=GRB.BINARY)

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
                        ineqs = area.H@obj.dr_var[idx_dim_0:idx_dim_f,cp,idx]
                    elif pv == "pos":
                        ineqs = area.H@obj.r_var[idx_dim_0:idx_dim_f,cp,idx]
                    c = ineqs[face] - b[face]

            #         obj.prog.addConstr(c <= obj.bigM*(1-b[cp,face]))
            # print(gp.quicksum(gp.quicksum(1-b)))
            # obj.prog.addConstr(gp.quicksum(gp.quicksum(1-b)) <= obj.bigM*(1-area.z_space[idx]))
                    # if c < 0 then b1 = 1
                    # b1 = obj.prog.addVar(vtype=GRB.BINARY)
                    # obj.prog.addConstr(c <= 0 + obj.bigM*(1-b1),
                    #                     name=f"MU stay-in {idx} {cp} {face}")
                    # obj.prog.addConstr(c >= 0 - obj.bigM*b1,
                    #                     name=f"MU stay-in {idx} {cp} {face}")
                    # obj.prog.addConstr(b1 == area.z_space[idx])

                    obj.prog.addConstr(c <= (obj.bigM)*(1-area.z_space[idx]),
                                        name=f"MU stay-in {idx} {cp} {face}")
                        
    # now consider discon of mu to tie together the different area.z_space for a mu.z_space
    if len(mu.areas) == 1:
        mu.z_space = mu.areas[0].z_space
    else:
        mu.z_space = obj.prog.addMVar((obj.N,1),vtype=GRB.BINARY)
        if mu.discon == "AND":
            for idx in range(obj.N):
                obj.prog.addConstr(mu.z_space[idx] == gp.min_(area.z_space[idx] for area in mu.areas),
                                      name=f"MU AND {idx}")
        elif mu.discon == "OR":
            for idx in range(obj.N):
                # obj.prog.addConstr(mu.z_space[idx] == gp.quicksum(area.z_space[idx] for area in mu.areas),
                #                         name=f"MU OR {idx}")
                obj.prog.addConstr(mu.z_space[idx] == gp.max_(area.z_space[idx] for area in mu.areas),
                                      name=f"MU OR {idx}")     


def parse_time(obj,pred):
    # parse the time variables from the pred
    pred.z_time = obj.prog.addMVar((obj.N,1),vtype=GRB.BINARY)

    ### first of the idx1 and last control point of the idx 1
    # so we decide to have at least Bezier curve for which the first control point is before
    # and the last control point is after
    for idx in range(obj.N):
        # # at zero there are some numerical issues, so we add a small epsilon
        # eps = 1e-4
        # try:
        #     pred.I[0] -= eps if idx == 0 else 0
        # except:
        #     continue

        b1 = obj.prog.addVar(vtype=GRB.BINARY)
        b2 = obj.prog.addVar(vtype=GRB.BINARY)
            
        # if obj.h_var[0,-1,idx] > pred.range[0] then b1 = 1
        obj.prog.addConstr(obj.h_var[0,-1,idx] >= pred.I[0] - obj.bigM*(1-b1),
                            name=f"conditional constraint {pred.get_string()} {idx} 1")
        obj.prog.addConstr(obj.h_var[0,-1,idx] <= pred.I[0] + obj.bigM*b1,
                            name=f"conditional constraint {pred.get_string()} {idx} 1")
        
        # if obj.h_var[0,0,idx] < pred.I[1] then b2 = 1
        obj.prog.addConstr(obj.h_var[0,0,idx] <= pred.I[1] + obj.bigM*(1-b2),
                            name=f"conditional constraint {pred.get_string()} {idx} 2")
        obj.prog.addConstr(obj.h_var[0,0,idx] >= pred.I[1] - obj.bigM*b2,
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
        obj.prog.addConstr(pred.z_time[idx] == c,
                            name=f"parse_time z_time = c {pred.get_string()} {idx}")


def add_temporal_robustness(obj,pred):
    # add variables for the temporal robustness
    dt = obj.tf - obj.t0
    pred.theta_p = obj.prog.addVar(lb=-dt,ub=dt,vtype=GRB.CONTINUOUS)
    pred.theta_m = obj.prog.addVar(lb=-dt,ub=dt,vtype=GRB.CONTINUOUS)
    # at least one of the variables in z_time should be equal to 1 so that for 
    # at least two points, they are inside the time-range and we can satisfy
    # the pred in continuous time
    # obj.prog.addConstr(gp.quicksum(pred.z_time) >= 1)

    # get the first index for which the variable z_time[idx] is equal to 1
    h_0_array = obj.prog.addMVar((obj.N,1),lb=obj.t0,ub=obj.tf+obj.bigM,vtype=GRB.CONTINUOUS)
    h_f_array = obj.prog.addMVar((obj.N,1),lb=-obj.t0-obj.bigM,ub=obj.tf,vtype=GRB.CONTINUOUS)
    for idx in range(obj.N):
        # CONSTRAINT: c indicates whether z_space[idx] and z_time[idx] hold
        c = obj.prog.addVar(vtype=GRB.BINARY)
        obj.prog.addConstr(pred.z_time[idx] + pred.mu.z_space[idx] - 1 <= c,
                            name=f"temporal robustness c sum inequality {pred.get_string()} {idx}")
        obj.prog.addConstr(c <= pred.z_time[idx],
                            name=f"temporal robustness c z_time {pred.get_string()} {idx}")
        obj.prog.addConstr(c <= pred.mu.z_space[idx],
                            name=f"temporal robustness c z_space {pred.get_string()} {idx}")

        # CONSTRAINT: we now constrain that if c holds, then we add it to the timed constraint
        obj.prog.addConstr(h_0_array[idx] == obj.h_var[0,0,idx] + obj.bigM*(1-c),
                           name=f"temporal robustness h_0_array {pred.get_string()} {idx}")
        obj.prog.addConstr(h_f_array[idx] == obj.h_var[0,-1,idx] - obj.bigM*(1-c),
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

    # now we can use this variable to constrain the temporal robustness
    ###  TEMPORAL ROBUSTNESS
    if pred.type == "G":
        # for the always operator, we add temporal robustness around the time range
        # forward temporal robustness
        obj.prog.addConstr(h_max >= pred.I[1] + pred.theta_p,
                           name=f"temporal robustness h_max {pred.get_string()}")
        # backward temporal robustness
        obj.prog.addConstr(h_min <= pred.I[0] - pred.theta_m,
                           name=f"temporal robustness h_min {pred.get_string()}")
    elif pred.type == "F":
        # for the eventually operator, we add tmeporal robustness around the time point 
        t_star_p = obj.prog.addVar(lb=pred.I[0],ub=pred.I[1],vtype=GRB.CONTINUOUS)
        t_star_m = obj.prog.addVar(lb=pred.I[0],ub=pred.I[1],vtype=GRB.CONTINUOUS)

        # forward temporal robustness
        # also constrain that t_star is lower bounded by h_max
        # obj.prog.addConstr(t_star_p >= h_max,
        #                     name=f"temporal robustness t_star_p {pred.get_string()}")
        # obj.prog.addConstr(h_max >= t_star_p + pred.theta_p,
        #                    name=f"temporal robustness h_max {pred.get_string()}")
        h_max_I = obj.prog.addVar(lb=pred.I[0],ub=pred.I[1],vtype=GRB.CONTINUOUS)
        h_min_I = obj.prog.addVar(lb=pred.I[0],ub=pred.I[1],vtype=GRB.CONTINUOUS)
        obj.prog.addConstr(h_max_I == gp.min_(pred.I[1],h_max))
        obj.prog.addConstr(h_min_I == gp.max_(pred.I[0],h_min))

        obj.prog.addConstr(pred.theta_p == h_max - h_min_I)
        obj.prog.addConstr(pred.theta_m == h_max_I - h_min)
        
        # obj.prog.addConstr(pred.theta_m == h_max - h_min)
        
        # backward temporal robustness
        # also constrain that t_star is upper bounded by h_min
        # obj.prog.addConstr(t_star_m <= h_min,
        #                     name=f"temporal robustness t_star_m {pred.get_string()}")
        # obj.prog.addConstr(h_min <= t_star_m - pred.theta_m,
        #                    name=f"temporal robustness h_min {pred.get_string()}")

        # obj.prog.addConstr(pred.theta_p == h_max - pred.I[0])
        # obj.prog.addConstr(pred.theta_p == h_max - h_min)

        # obj.prog.addConstr(pred.I[0] <= h_max)
        # obj.prog.addConstr(h_max <= pred.I[1])

        # obj.prog.addConstr(pred.I[0] <= h_min)
        # obj.prog.addConstr(h_min <= pred.I[1])
        
        pred.t_star_p = t_star_p
        pred.t_star_m = t_star_m
    