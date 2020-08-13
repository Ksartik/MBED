import numpy as np
import sys
import time
from utils import add_edge, delete_edge, get_indicator_vector, update_res, is_balanced
sys.path.append("./finding-balanced-subgraphs")
import timbal

def initialize (A, S):
    """
    A - largest connected component in H
    S - set of vertices in the MBS
    will return the initial node-labelling and candidate set 
    """
    ind = get_indicator_vector(A[S,:][:,S])
    x_v = np.zeros(shape=A.shape[0])
    x_v[S] = ind
    # Candidate set to contain only the edges on the periphery of As
    C = []
    for v in S:
        for u in A[v, :].nonzero()[1]:
            if (x_v[u] == 0):
                C.append((v, u))
    return x_v, C

def node_out (x_v, e):
    if ((x_v[e[0]] != 0) and (x_v[e[1]] == 0)):
        return e[1]
    elif ((x_v[e[0]] == 0) and (x_v[e[1]] != 0)):
        return e[0]
    else:
        raise Exception('Edge ', e, ' that is passed not on the periphery')

def find_label (A, u, x_v):
    connected = np.abs(A[u,:]).dot(np.abs(x_v)) > 0
    if connected:
        x = A[u,:]
        ip = x.dot(x_v)
        nip = np.abs(x).dot(np.abs(x_v))
        agree = np.abs(ip)==nip
        if agree:
            return (1 if (ip >= 0) else -1)
        else:
            return 0
    else:
        raise Exception("vertex unconnected from MBS")

def count_compatible (u, A, x_v):
    cc = 1
    nbrs_u = np.nonzero(A[u,:])[1]
    for w in nbrs_u:
        if ((x_v[w] != 0) and (x_v[w] != A[u, w] * x_v[u])):
            # print((u, x_v[u]), " ", A[u, w], " ", (w, x_v[w]))
            return 0
    for w in nbrs_u:
        if (x_v[w] == 0):
            x_v[w] = A[u, w] * x_v[u]
            cc_w = count_compatible(w, A, x_v)
            if (cc_w == 0):
                x_v[w] = 0
                continue
            cc += cc_w
    return cc

def marginal_gain_ed (x_v, A, e): 
    """
    Calculating f_{S_{i-1}}(e) for each iteration 
    """
    ue = node_out(x_v, e)
    e_sign = A[e]
    A = delete_edge (A, e)
    try:
        label_ue = find_label (A, ue, x_v)
    except:
        # Edge making graph unconnected
        A = add_edge (A, e, e_sign)
        return -1
    if (label_ue == 0):
        A = add_edge (A, e, e_sign)
        return 0
    else:
        x_v[ue] = label_ue
        cc = count_compatible (ue, A, x_v.copy())
        A = add_edge (A, e, e_sign)
        x_v[ue] = 0
        # x_v = unset_all (ue, x_v, A, S)
        return (cc)

def find_optimal_balance (x_v, A, C, budget, verbose=True):
    if (budget == 0):
        return 0, []
    else:
        # if (verbose):
        #     print(budget, len(x_v.nonzero()[0]), len(C), A.shape[0])
        if (len(C) == 0):
            return 0, []
        opt_edges_removed = []
        opt_bal = 0
        for e in C:
            mag = marginal_gain_ed (x_v, A, e)
            if (mag >= 0):
                Ad = delete_edge(A.copy(), e)
                xd_v, Cd = initialize(Ad, np.nonzero(x_v)[0])
                bal_b1, edges_removed = find_optimal_balance(xd_v, Ad, Cd, budget-1, verbose=verbose)
                bal_e = mag + bal_b1
                if (bal_e > opt_bal):
                    opt_bal = bal_e
                    opt_edges_removed = edges_removed + [e]
        if (verbose):
            print(budget, len(x_v.nonzero()[0]), len(C), A.shape[0])
        return opt_bal, opt_edges_removed

def mbed_solve (A, budgets, S, verbose=True):
    """
    Maximize balance after deleting budget no. of edges from graph given initial MBS
    """
    start_time = time.time()
    x_v, C = initialize(A, S)
    delta, edges_removed = find_optimal_balance (x_v, A, C, max(budgets), verbose=verbose)
    time_taken = time.time() - start_time
    Ad = A.copy()
    for e in edges_removed:
        Ad = delete_edge(Ad, e)
    print(delta)
    results_info = [{"Budget": max(budgets), "Delta": delta, "RT": time_taken}]
    return results_info, timbal.process_only_second(Ad, S), Ad, edges_removed