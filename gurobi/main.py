import gurobipy as gp
from gurobipy import GRB


def bilinear():
    # Copyright 2022, Gurobi Optimization, LLC

    # This example formulates and solves the following simple bilinear model:
    #  maximize    x
    #  subject to  x + y + z <= 10
    #              x * y <= 2         (bilinear inequality)
    #              x * z + y * z = 1  (bilinear equality)
    #              x, y, z non-negative (x integral in second version)

    # Create a new model
    m = gp.Model("bilinear")

    # Create variables
    x = m.addVar(name="x")
    y = m.addVar(name="y")
    z = m.addVar(name="z")

    # Set objective: maximize x
    m.setObjective(1.0*x, GRB.MAXIMIZE)

    # Add linear constraint: x + y + z <= 10
    m.addConstr(x + y + z <= 10, "c0")

    # Add bilinear inequality constraint: x * y <= 2
    m.addConstr(x*y <= 2, "bilinear0")

    # Add bilinear equality constraint: x * z + y * z == 1
    m.addConstr(x*z + y*z == 1, "bilinear1")

    # First optimize() call will fail - need to set NonConvex to 2
    try:
        m.optimize()
    except gp.GurobiError:
        print("Optimize failed due to non-convexity")

    # Solve bilinear model
    m.Params.NonConvex = 2
    m.optimize()

    m.printAttr('x')

    # Constrain 'x' to be integral and solve again
    x.VType = GRB.INTEGER
    m.optimize()

    m.printAttr('x')


def linear():
    #!/usr/bin/env python3.7

    # Copyright 2022, Gurobi Optimization, LLC

    # This example formulates and solves the following simple MIP model:
    #  maximize
    #        x +   y + 2 z
    #  subject to
    #        x + 2 y + 3 z <= 4
    #        x +   y       >= 1
    #        x, y, z binary

    try:

        # Create a new model
        m = gp.Model("mip1")

        # Create variables
        x = m.addVar(vtype=GRB.BINARY, name="x")
        y = m.addVar(vtype=GRB.BINARY, name="y")
        z = m.addVar(vtype=GRB.BINARY, name="z")

        # Set objective
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")

        # Optimize model
        m.optimize()

        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X))

        print('Obj: %g' % m.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')




if __name__ == "__main__":
    # bilinear()
    linear()