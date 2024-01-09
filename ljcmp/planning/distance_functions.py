
import numpy as np
from scipy.stats import norm

def distance_z(z1, z2):
    cdf_z1 = norm.cdf(z1)
    cdf_z2 = norm.cdf(z2)
    dist = np.linalg.norm(cdf_z1 - cdf_z2)
    return dist

def distance_q(q1, q2):
    if q1 is None or q2 is None:
        return np.inf
    
    dist = np.linalg.norm(q1 - q2)
    return dist

def distance_discrete_geodesic_q(q1, q2, k, constraint):
    q_ints = np.linspace(q1, q2, k)
    dist = 0

    for i in range(k-1):
        r = constraint.project(q_ints[i+1])
        if r is False:
            return np.inf

        dist += distance_q(q_ints[i], q_ints[i+1])

    return dist
