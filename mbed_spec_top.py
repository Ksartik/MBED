import numpy as np
import sys
from utils import estimate_ep_ED, delete_edge, estimate_ep_CG, is_connected_postdel, update_res
sys.path.append("./finding-balanced-subgraphs")
import timbal
import time

"""
Maximizing the balance by minimizing an upper bound on the minimum eigenvalue after edge deletion.
"""

def node_out(S_curr, e):
    if ((e[0] in S_curr) and (e[1] not in S_curr)):
        return e[1]
    elif ((e[1] in S_curr) and (e[0] not in S_curr)):
        return e[0]
    else:
        raise Exception() 

def marginal_loss_ed (A, e, ev_Si):
    """
    ev_Si - eigenvector for the minimum eigenvalue in the ith iteration
    returns the marginal loss in the upper bound of min eigenvalue after deleting edge e
    """
    return ((ev_Si[e[0]] - A[e[0], e[1]] * ev_Si[e[1]])**2)


def greedy_solve (A, S, budgets, start_time, verbose=True):
    """
    Greedy will give an optimal set of edges to be deleted such that the devised upper bound
    on the minimum eigenvalue is minimized. 
    """
    A = A.copy()
    # Get the candidate set - here we will have to consider all the edges
    # As the algorithm does not have a way to define the periphery. 
    C = []
    for ui in range(A.shape[0]):
        for vi in A[ui, :].nonzero()[1]:
            if ((ui < vi) and ((ui not in S) or (vi not in S))):
                C.append((ui, vi))
    # us, vs = np.nonzero(A)
    # C = [(u, v) for u, v in zip(us, vs)]
    estimate_eigenpair = estimate_ep_ED if (A.shape[0] < 100) else estimate_ep_CG
    edges_removed = []
    marginal_ub_loss = np.zeros(shape=len(C))
    s, V = estimate_eigenpair (A)
    s, V = estimate_eigenpair (A, V)
    if (verbose):
        print("Eigenpair calculated")
    v = V[:, 0]
    for j, e in enumerate(C):
        marginal_ub_loss[j] = marginal_loss_ed (A, e, v)
    if (verbose):
        print("Marginal UB losses calculated")
    top_edges_inds = sorted(np.arange(len(C)), key=lambda x: marginal_ub_loss[x], reverse=True)
    edges_removed, nedges_removed = [], 0
    budget = np.max(budgets)
    select_time = 0
    results_info = []
    S_curr = S
    for i in top_edges_inds:
        e, ub_loss_i = C[i], marginal_ub_loss[i]
        if (verbose):
            print(e, ub_loss_i, nedges_removed)
        # if (is_connected_postdel(delete_edge(A.copy(), e), e)):
        Ad = delete_edge(A.copy(), e)
        try:
            ue = node_out(S_curr, e)
        except:
            continue
        if (len(Ad[S_curr, ue].nonzero()) > 0):
            A = delete_edge(A, e)
            S_curr = timbal.process_only_second(A, S_curr)
            nedges_removed += 1
            edges_removed.append(e)
            if (nedges_removed in budgets):
                select_time += time.time() - start_time
                # S_new = timbal.process_only_second(A, S)
                S_new = S_curr
                results_info.append({"Budget": nedges_removed, "RT": select_time, "Delta": len(S_new) - len(S)})
                start_time = time.time()
            if (nedges_removed == budget):
                # S_new = timbal.process_only_second(A, S)
                S_new = S_curr
                return results_info, S_new, A, edges_removed
    S_new = timbal.process_only_second(A, S)
    results_info = update_res(results_info, budgets, time.time() - start_time, len(S_new) - len(S))
    return results_info, S_new, A, edges_removed
    

def mbed_solve (A, budgets, S, verbose=True, **kwargs):
    start_time = time.time()
    results_info, S_new, Ad, edges_removed = greedy_solve(A, S, budgets, start_time, verbose=verbose)
    # select_time = time.time() - start_time
    # components = find_connected_components(Ad)
    # largest = components[np.argmax([len(i) for i in components])]
    # Ad = Ad[largest,:][:,largest]
    # print("Ad: ", Ad.shape, "\nA: ", A.shape)
    # S_max = []
    # for i in range(n_iters):
    #     S_new = timbal.process(Ad, max_removals=kwargs['max_removals'], samples=kwargs['samples'], 
    #                             avg_size=kwargs['avg_size'], subsample=kwargs['subsample'])
    #     if (len(S_new) > len(S_max)):
    #         S_max = S_new
    # S_new = S_max    
    # S_new = timbal.process_only_second(Ad, S)
    return results_info, S_new, Ad, edges_removed