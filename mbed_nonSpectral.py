import numpy as np
import sys
import time
from utils import add_edge, delete_edge, get_indicator_vector, update_res
from pprint import pprint
sys.path.append("./finding-balanced-subgraphs")
import timbal
from scipy import sparse

def get_periphery (A, S):
    # Candidate set to contain only the edges on the periphery of As
    C = []
    for v in S:
        for u in A[v, :].nonzero()[1]:
            if (u not in S):
                C.append((v, u))
    return C

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
    marginal_benefits = {}
    for i in range(budget):
        if (verbose):
            print("Budget:", i)
        if (len(C) == 0):
            # Maximum balance achieved - budget high.
            results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
            break
        # S_previter = np.nonzero(x_v)[1]
        rows, cols = [], []
        out_nodes = []
        for v in added_nodes:
            for u in A[added_nodes, :].nonzero()[1]:
                if (x_v[u] != 0):
                    out_nodes.append(u)
        while (True):
            new_count = 0
            for e in C:
                if ((e not in marginal_benefits) or (e[0] in out_nodes) or (e[1] in out_nodes)): #or (len(A[e[0],:][:,added_nodes].nonzero()[1]) > 1) or (len(A[e[1],:][:,added_nodes].nonzero()[1]) > 1)):#(e[0] in added_nodes) and (e[1] in added_nodes))):
                    new_count += 1
                    marginal_benefits[e] = marginal_gain_ed (x_v, A, e)
            if (verbose):
                print("New count:", new_count, "out of", len(C))
            # marginal_benefits=sparse.csr_matrix(marginal_benefits)
            # marginal_benefits_C = np.array(marginal_benefits[[e[0] for e in C], [e[1] for e in C]]).ravel()
            marginal_benefits_C = np.array(marginal_benefits.values())
            if (np.all(marginal_benefits_C == -1)):
                # No more edges can be removed without making the graph unconnected
                results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
                return results_info, np.nonzero(x_v)[0], A, edges_removed
            if (randomize):
                Mi = sorted(C, key=lambda x: marginal_benefits[x], 
                            reverse=True)[:budget]
                e_chosen = Mi[np.random.choice(np.arange(len(Mi)))]
                # Mi = sorted(range(len(C)), key=lambda x: marginal_benefits_C[x], 
                #             reverse=True)[:budget]
                # e_ind = np.random.choice(Mi)
            else:
                # e_ind = np.argmax(marginal_benefits_C)
                e_chosen = sorted(C, key=lambda x: marginal_benefits[x], reverse=True)[0]
            if (marginal_benefits[e_chosen] > -1):
                break
            # if (marginal_benefits_C[e_ind] > -1):
            #     break
        # e_chosen = C[e_ind]
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
            print(e_chosen, " is chosen and marginal gain is ", marginal_benefits[e_chosen]) #marginal_benefits_C[e_ind])
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
        print("V1: ", np.sum(x_v == 1), ", V2: ", np.sum(x_v == -1))
    results_info, S_new, Ad, edges_removed = greedy_solve (x_v, C, A, S, budgets, start_time, randomize=randomize, verbose=verbose)
    # S_new = timbal.process_only_second(Ad, S)
    return results_info, S_new, Ad, edges_removed