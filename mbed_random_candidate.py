import numpy as np
import sys
import time
from utils import add_edge, delete_edge, is_connected_postdel, get_indicator_vector, update_res

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

def update_chosen (u, x_v, A, C):
    """
    Update the labelling and find new candidate set after deleting an edge 
    """
    nbrs_u = np.nonzero(A[u,:])[1]
    for w in nbrs_u:
        if ((x_v[w] != 0) and (x_v[w] != A[u, w] * x_v[u])):
            x_v[u] = 0
            return (C, [])
    C = [e for e in C if (u not in e)]
    new_candidates = []
    for w in nbrs_u:
        if (x_v[w] == 0):
            x_v[w] = A[u, w] * x_v[u]
            C, new_candidates_w = update_chosen (w, x_v, A, C)
            if (x_v[w] == 0):
                new_candidates.append((u, w))
            else:
                new_candidates += new_candidates_w
    return (C, new_candidates)


def random_choose_candidate_solve (x_v, C, A, S, budgets, start_time, verbose=True):
    """
    Random_choose_candidate Algorithm to maximize balance after deleting budget no. of edges
    """
    A = A.copy()
    edges_removed = []
    budget = np.max(budgets)
    results_info = []
    for i in range(budget):
        if (len(C) == 0):
            # Maximum balance achieved -> budget high.
            results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
            break
        while (True):
            try:
                e_chosen = C[np.random.choice(range(len(C)))]
            except:
                results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
                return results_info, np.nonzero(x_v)[0], A, edges_removed
            if (is_connected_postdel(delete_edge(A.copy(), e_chosen), e_chosen)):
                break
            else:
                C.remove(e_chosen)
        edges_removed.append(e_chosen)
        try:
            ue = node_out (x_v, e_chosen)
        except:
            print(e_chosen, " is not on the periphery")
            return
        A = delete_edge (A, e_chosen)
        x_v[ue] = find_label (A, ue, x_v)
        C.remove(e_chosen)
        if (verbose):
            print(e_chosen, " is chosen")
        if (x_v[ue] != 0):
            C, C_i = update_chosen(ue, x_v, A, C)
            # if (verbose):
            #     print("Edges added to C: ", C_i)
            C = C + C_i
        if (len(edges_removed) in budgets):
            select_time = time.time() - start_time
            results_info.append({"Budget": len(edges_removed), "RT": select_time, "Delta": len(np.nonzero(x_v)[0]) - len(S)})
        if (verbose):
            print("\n")
    return results_info, np.nonzero(x_v)[0], A, edges_removed


def mbed_solve (A, budgets, S, verbose=True):
    """
    Maximize balance after deleting budget no. of edges from graph given initial MBS
    """
    # print(S)
    start_time = time.time()
    x_v, C = initialize(A, S)
    if (verbose):
        print("Initialized")
        print("V1: ", np.sum(x_v == 1), " ,V2: ", np.sum(x_v == -1))
    results_info, S_new, Ad, edges_removed = random_choose_candidate_solve (x_v, C, A, S, budgets, start_time, verbose=verbose)
    return results_info, S_new, Ad, edges_removed