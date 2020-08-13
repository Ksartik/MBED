import numpy as np
import sys
from utils import estimate_ep_ED, delete_edge, get_indicator_vector, estimate_ep_CG, is_connected_postdel, update_res
import time
import timbal

"""
Maximizing the balance by minimizing an upper bound on the minimum eigenvalue after edge deletion.
"""

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


def marginal_loss_ed (A, e, ev_Si, considered_map):
    """
    ev_Si - eigenvector for the minimum eigenvalue in the ith iteration
    returns the marginal loss in the upper bound of min eigenvalue after deleting edge e
    """
    ex = (considered_map[e[0]], considered_map[e[1]])
    return ((ev_Si[ex[0]] - A[ex[0], ex[1]] * ev_Si[ex[1]])**2)


def greedy_solve (x_v, C, A, S, budgets, start_time, randomize=True, verbose=True):
    """
    Greedy will give an optimal set of edges to be deleted such that the devised upper bound
    on the minimum eigenvalue is minimized. 
    """
    A = A.copy()
    estimate_eigenpair = estimate_ep_ED if (A.shape[0] < 100) else lambda A,x=[]: estimate_ep_CG(A, x=x, maxiter=500)
    edges_removed = []
    select_time = 0
    budget = np.max(budgets)
    results_info = []
    for i in range(budget):
        if (verbose):
            print("Budget: ", i)
            print("C: ", len(C))
        if (len(C) == 0):
            # Maximum balance achieved -> budget high.
            results_info = update_res(results_info, budgets, time.time() - start_time, len(x_v.nonzero()[0]) - len(S))
            break
        to_consider = np.zeros(shape=len(x_v), dtype=bool)
        to_consider[S] = True
        for e in C:
            to_consider[node_out(x_v, e)] = True
        nodes_to_consider = np.where(to_consider)[0]
        considered_map = np.zeros(shape=len(x_v), dtype=int) - 1
        for j, n in enumerate(nodes_to_consider):
            considered_map[n] = j
        #Aps = A[nodes_to_consider, :][:, nodes_to_consider]
        Aps = A.copy()
        considered_map = np.arange(len(x_v))
        s, V = estimate_eigenpair (Aps)
        s, V = estimate_eigenpair (Aps, V)
        v = V[:, 0]
        if (verbose):
            print ("Eigenpair calculated: value:", s.item())
            print ("Eigenvector: ", np.min(v), "-",np.max(v),  np.matmul(v.T, v))
        marginal_ub_loss = np.zeros(shape=len(C))
        for j, e in enumerate(C):
            marginal_ub_loss[j] = marginal_loss_ed (Aps, e, v, considered_map)
            # print(e, marginal_ub_loss[j])
        top_losses = sorted(range(len(C)), key=lambda x: marginal_ub_loss[x], reverse=True)
        while (True):
            try:
                e_ind = np.random.choice(top_losses[:budget]) if (randomize) else top_losses[0]
            except: 
                results_info = update_res(results_info, budgets, time.time() - start_time, len(np.nonzero(x_v)[0]) - len(S))            
                return results_info, len(np.nonzero(x_v)[0]), A, edges_removed
            max_loss_e = C[e_ind]
            if (is_connected_postdel(delete_edge(A.copy(), max_loss_e), max_loss_e)):
                break
            else:
                C.remove(max_loss_e)
                top_losses.remove(e_ind)
                for j, ind in enumerate(top_losses):
                    if (ind > e_ind):
                        top_losses[j] -= 1
        edges_removed.append(max_loss_e)
        try:
            ue = node_out (x_v, max_loss_e)
        except:
            print(max_loss_e, " is not on the periphery")
            return
        A = delete_edge(A, max_loss_e)
        x_v[ue] = find_label (A, ue, x_v)
        max_loss = marginal_ub_loss[e_ind]
        C.remove(max_loss_e)
        if (x_v[ue] != 0):
            C, C_i = update_chosen(ue, x_v, A, C)
            if (verbose):
                print(len(C_i))
            C = C + C_i
        if (len(edges_removed) in budgets):
            select_time = time.time() - start_time
            #print(len(timbal.process_only_second(A, S)))
            results_info.append({"Budget": len(edges_removed), "RT": select_time, "Delta": len(np.nonzero(x_v)[0]) - len(S)})
        if (verbose):
            print("Edge ", max_loss_e, " is chosen with loss in UB: ", max_loss)
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
    return results_info, S_new, Ad, edges_removed
