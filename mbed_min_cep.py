import numpy as np
import sys
import time
from utils import add_edge, delete_edge, get_indicator_vector, update_res
from pprint import pprint
sys.path.append("./finding-balanced-subgraphs")
import timbal
from scipy import sparse

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
        raise Exception("vertex disconnected from MBS")


def node_out (x_v, e):
    if ((x_v[e[0]] != 0) and (x_v[e[1]] == 0)):
        return e[1]
    elif ((x_v[e[0]] == 0) and (x_v[e[1]] != 0)):
        return e[0]
    else:
        raise Exception('Edge ', e, ' that is passed not on the periphery')

def find_cep (A, u, x_v):
    connected = np.abs(A[u,:]).dot(np.abs(x_v)) > 0
    if connected:
        x = A[u,:]
        ip = x.dot(x_v)
        nip = np.abs(x).dot(np.abs(x_v))
        return (nip - np.abs(ip))
    else:
        raise Exception("vertex disconnected from MBS")

def marginal_cep_ed (x_v, A, e):
    ue = node_out(x_v, e)
    e_sign = A[e]
    A = delete_edge(A, e)
    try:
        cep_ue = find_cep(A, ue, x_v)
    except:
        cep_ue = -1
    A = add_edge (A, e, e_sign)
    return cep_ue

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


def greedy_solve (x_v, C, A, S, budgets, start_time, randomize=True, verbose=True):
    """
    (Randomized) Greedy Algorithm to maximize balance after deleting budget no. of edges
    """
    A = A.copy()
    edges_removed = []
    results_info = []
    budget = np.max(budgets)
    added_nodes = []
    # marginal_benefits=sparse.csr_matrix(([], ([], [])), 
    #                         shape=(A.shape[0], A.shape[0])).astype(np.int)
    n_cep = {}
    for i in range(budget):
        if (verbose):
            print("Budget:", i)
        if (len(C) == 0):
            # Maximum balance achieved - budget high.
            results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
            break
        # S_previter = np.nonzero(x_v)[1]
        out_nodes = []
        for v in added_nodes:
            for u in A[added_nodes, :].nonzero()[1]:
                if (x_v[u] != 0):
                    out_nodes.append(u)
        new_count = 0
        for e in C:
            if ((e not in n_cep) or (e[0] in out_nodes) or (e[1] in out_nodes)):
                mag = marginal_cep_ed (x_v, A, e)
                new_count += 1
                if (mag > -1):
                    n_cep[e] = mag
                else:
                    C.remove(e)
        if (verbose):
            print("New count:", new_count, "out of", len(C))
        if (len(n_cep.keys()) == 0):
            # No more edges can be removed without making the graph disconnected
            results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
            return results_info, np.nonzero(x_v)[0], A, edges_removed
        Mi = sorted(C, key=lambda x: n_cep[x])
        top_ind = 0
        e_chosen = sorted(C, key=lambda x: n_cep[x])[0]
        edges_removed.append(e_chosen)
        try:
            ue = node_out (x_v, e_chosen)
        except:
            print(e_chosen, " is not on the periphery")
            return
        A = delete_edge (A, e_chosen)
        old_xv = x_v.copy()
        x_v[ue] = find_label (A, ue, x_v)
        C.remove(e_chosen)
        if (verbose):
            print(e_chosen, " is chosen and cep post del is ", n_cep[e_chosen]) #marginal_benefits_C[e_ind])
        if (x_v[ue] != 0):
            C, C_i = update_chosen(ue, x_v, A, C)
            # if (verbose):
            #     print("Edges added to C: ", C_i)
            C = C + C_i
        added_nodes = [u for u in x_v.nonzero()[0] if (old_xv[u] == 0)]
        if (len(edges_removed) in budgets):
            select_time = time.time() - start_time
            results_info.append({"Budget": len(edges_removed), "RT": select_time, "Delta": len(np.nonzero(x_v)[0]) - len(S)})
        if (verbose):
            print(len(timbal.process_only_second(A, S)), len(np.nonzero(x_v)[0]))
            print("\n")
    return results_info, np.nonzero(x_v)[0], A, edges_removed


def mbed_solve (A, budgets, S, randomize=True, verbose=True):
    """
    Maximize balance after deleting budget no. of edges from graph given initial MBS
    """
    # print(S)
    start_time = time.time()
    x_v, C = initialize(A, S)
    if (verbose):
        print("Initialized")
        print("V1: ", np.sum(x_v == 1), " ,V2: ", np.sum(x_v == -1))
    results_info, S_new, Ad, edges_removed = greedy_solve (x_v, C, A, S, budgets, start_time, randomize=randomize, verbose=verbose)
    # S_new = timbal.process_only_second(Ad, S)
    return results_info, S_new, Ad, edges_removed