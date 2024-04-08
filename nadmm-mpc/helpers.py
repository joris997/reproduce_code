import numpy as np
import casadi as ca

def stack_dict(dict):
    # stack all dictionary items in increasing order of key
    # also consider they might not exist
    stack = np.array([])
    for key in sorted(dict.keys()):
        stack = np.hstack((stack, dict[key]))
    return stack
    
def smooth_min(p_vars):
    eta = 5
    summation = 0
    for p in p_vars:
        summation += ca.exp(-eta*p)
    min = -ca.log(summation)/eta
    return min

def smooth_max(p_vars):
    eta = 5
    numerator = 0
    denominator = 0
    for p in p_vars:
        numerator += ca.exp(eta*p)*p
        denominator += ca.exp(eta*p)
    max = numerator/denominator
    return max