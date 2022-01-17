import torch
import numpy as np
import pdb

# currently none of these functions are being used

def row_constraint(a, diff=2):
    return np.array([a[0:diff].sum(), a[diff:].sum()])-1
def col_constraint(a, diff=2):
    return np.array([a[0::diff].sum(), a[1::diff].sum()])-1

def k_best_costs_nlp(k, c):
    phi_1, assignment = [], []
    for i in range(0,k):

        # func= lambda x: ((x*c.flatten()).sum()+ (list(np.round(x,2)) in assignment)*500)
        # bnds=((0,1), (0,1),(0,1),(0,1)) #cost_matrix is 2*2
        # cons=({'type': 'ineq', 'fun': lambda x: - x},
        #       {'type': 'eq', 'fun': lambda x: x[0]+x[1]-1},
        #       {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 1},
        #       {'type': 'eq', 'fun': lambda x: x[0] + x[2] - 1},
        #       {'type': 'eq', 'fun': lambda x: x[1] + x[3] - 1})

        #res = minimize(func, (1,1,0,0), method='SLSQP',bounds=bnds, constraints=cons, options={'maxiter': 10000, 'eps': 0.0001})# method either 'L-BFGS-B' or 'TNC'
        if len(assignment)>0:
            A_ub = -np.expand_dims(c.flatten(),0)
            b_ub=-np.round(phi_1[-1],2)-1

        else:
            A_ub=None
            b_ub=None
        A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
        res = linprog(c.flatten(), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones((len(c.flatten()))), bounds=(0,1))

        phi_1.append(res.fun)
        assignment.append(res.x)

    assignment_reshaped=[i.reshape(c.shape) for i in assignment]
    return phi_1, assignment_reshaped