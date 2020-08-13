import numpy as np
import scipy as sp
import random
import math
from scipy import sparse
import scipy.sparse.linalg as splinalg

DEBUG = 0

def delete_edge(A, e):
    A[e[0], e[1]] = 0
    A[e[1], e[0]] = 0
    # A[e[0], e[0]] = A[e[0], e[0]] - 1
    # A[e[1], e[1]] = A[e[1], e[1]] - 1
    return A

def add_edge(A, e, e_sign):
    A[e[0], e[1]] = e_sign
    A[e[1], e[0]] = e_sign
    # A[e[0], e[0]] = A[e[0], e[0]] + 1
    # A[e[1], e[1]] = A[e[1], e[1]] + 1
    return A

def compute_signed_laplacian(A):
    N = A.shape[0]
    At = -A
    for i in range(N):        
        x1 = A[i,:].T                
        At[i,i] = np.sum(np.abs(x1))
    return At


def compute_signed_laplacian_sparse(A):
    D = sparse.spdiags(np.sum(np.abs(A), axis=0), 0, A.shape[0], A.shape[1])
    return D-A

def update_res (results_info, budgets, time_taken, delta):
    res_info = []
    budgets = budgets.copy()
    for res in results_info:
        res_info.append(res)
        budgets.remove(res["Budget"])
    for b in budgets:
        res_info.append({"Budget": b, "RT": time_taken, "Delta": delta})
    return res_info

def remove_vertices(A, vertices):
    """
    Remove vertices indexed by given list
    """
    N = A.shape[0]
    r = np.arange(N)
    remaining = np.delete(r,vertices)
    return A[remaining,:][:,remaining]


def rank_vertices(A, L=None, s=None, v=None):
    if L == None:
        L = compute_signed_laplacian(A)
        s,V = splinalg.eigsh(L)
        v = V[:,0]**2
    l=s[0]
    L=-np.abs(L) + sparse.spdiags(2*np.sum(np.abs(A), axis=0), 0, L.shape[0], L.shape[1]) \
    - sparse.spdiags(np.ones(L.shape[0])*2*l, 0, L.shape[0], L.shape[1])
    df=-L.dot(v)
    return (l-df)/(1.-v)

def write_adj_sparse_matrix(filename, A, sep=None):  
    from scipy import sparse
    with open(filename, 'w+') as f:
        for ui in range(A.shape[0]):
            for vi in A[ui, :].nonzero()[1]:
                if (ui < vi):
                    f.write(sep.join([str(ui), str(vi), str(A[ui, vi])]))
                    f.write("\n")

def write_snodes(filename, S):  
    with open(filename, 'w+') as f:
        for v in S:
            f.write(str(v))
            f.write("\n")

def read_snodes (filename):
    with open(filename, 'r') as f:
        S = []
        for l in f:
            S.append(int(l.strip("\n")))
    return (np.array(S))

def write_deledges (filename, edges):
    with open(filename, 'w+') as f:
        for e in edges:
            f.write(','.join([str(e[0]), str(e[1])]))
            f.write("\n")

def read_deledges (filename):
    with open(filename, 'r') as f:
        deledges = []
        for l in f:
            deledges.append(tuple([int(x) for x in l.strip("\n").split(",")]))
    return (deledges)

def read_sparse_matrix(filename):
    from scipy import sparse
    with open(filename, 'r') as f:
        data, rows, cols = [], [], []
        size = int(f.readline().strip('#'))
        for l in f:
            s=l.strip('\n').split()
            data.append(int(s[2]))
            rows.append(int(s[0]))
            cols.append(int(s[1]))
        Ac=sparse.csr_matrix((data, (rows, cols)), shape=(size, size)).astype(np.float)
    # Redundant info is aggregated by scipy, so we turn non-1's into 1's
    Ac[Ac>1]=1
    Ac[Ac<-1]=-1
    return Ac+Ac.T

def read_adj_sparse_matrix(filename, comment='%', sep=None, min_node=0):  
    from scipy import sparse
    with open(filename, 'r') as f:
        data, rows, cols = [], [], []
        for l in f:
            if (l[0] == comment):
                # Ignore the comments
                continue
            s=l.strip('\n').split(sep=sep)
            u, v = int(s[0])-min_node, int(s[1])-min_node
            if (u == v):
                continue
            try:
                data.append(int(s[2]))
            except:
                data.append(int(float(s[2])))
            rows.append(u)
            cols.append(v)
        size = np.max(rows + cols) + 1
        Ac=sparse.csr_matrix((data, (rows, cols)), shape=(size, size)).astype(np.float)
    # Redundant info is aggregated by scipy, so we turn non-1's into 1's
    Ac = Ac+Ac.T
    Ac[Ac>1]=1
    Ac[Ac<-1]=-1
    return Ac

def read_soc_txt_sparse_matrix(filename):
    from scipy import sparse
    with open(filename, 'r') as f:
        line = f.readline()
        while (line[0] == '#'):
            if (line[2:].startswith("Nodes:")):
                size = int(line[9:].split(" ")[0])
            line = f.readline()
        data, rows, cols = [], [], []
        for l in f:
            s=l.strip('\n').split()
            data.append(int(s[2]))
            rows.append(int(s[0]))
            cols.append(int(s[1]))
        Ac=sparse.csr_matrix((data, (rows, cols)), shape=(size, size)).astype(np.float)
    # Redundant info is aggregated by scipy, so we turn non-1's into 1's
    Ac[Ac>1]=1
    Ac[Ac<-1]=-1
    return Ac+Ac.T

def is_connected (A):
    def traverse_set_vis (visited, u):
        visited[u] = True
        for v in A[u, :].nonzero()[1]:
            if (not(visited[v])):
                traverse_set_vis(visited, v)
    visited = np.zeros(A.shape[0], dtype=bool)
    traverse_set_vis(visited, 0)
    if (np.all(visited)):
        return True
    else:
        return False

def is_connected_postdel (A, e):
    # given A does not contain e => we need to check if e[0] and e[1] are still connected
    visited = np.zeros(A.shape[0], dtype=bool)
    visited[e[0]] = True
    vertices = [e[0]]
    while (len(vertices) > 0):
        new_vertices = []
        for u in vertices:
            for v in A[u, :].nonzero()[1]:
                if (not(visited[v])):
                    if (v == e[1]):
                        return True
                    else:
                        visited[v] = True
                        new_vertices.append(v)
        vertices = new_vertices
    return False

def find_connected_components(A):
    Ab = np.abs(A)
    A = Ab.copy()
    all_indices = np.arange(A.shape[0])
    remaining_indices = np.arange(A.shape[0])
    removed_indices = []
    components = []

    while (A.shape[0] > 0):
        # Start with the neighbours of the max-degree vertex
        maxdeg = np.argmax(np.sum(A, axis=0))
        component = np.hstack([[maxdeg],np.nonzero(A[maxdeg,:])[1]])
        # BFS
        prelen = 0
        while len(component) > prelen:
            prelen = len(component)
            B = np.sum(A[component,:], axis=0)
            component = np.unique(np.hstack([component, np.nonzero(B)[1]]))
        components.append(remaining_indices[component])
        remaining_indices = np.delete(remaining_indices, component)
        A=Ab[remaining_indices,:][:,remaining_indices]
    return components


def estimate_ep_ED(A, x=[], maxiter=None):
    N = A.shape[0]
    SHIFT = N
    L = compute_signed_laplacian_sparse(A)
    H = sparse.spdiags(np.ones(N)*SHIFT, 0, A.shape[0], A.shape[1])
    s,V = splinalg.eigsh(-L+H, k=1)
    s=N-s
    return s,V

def estimate_ep_CG(A, x=[], maxiter=100):
    if len(x) == 0:
        x=np.random.random((A.shape[0], 1))
    L = compute_signed_laplacian_sparse(A)
    s,V = splinalg.lobpcg(L, x, largest=False, tol=1e-12, maxiter=maxiter)
    return s,V


def is_balanced_(A, sv=0):

    #-1 means we switch this node, 1 means we keep its sign, 0 means we have not visited it yet
    sign_of_node=A.shape[0]*[0]
    #sign of first node
    sign_of_node[0]=1
    #add the first node to the current level
    priority_queue=np.array((A.shape[0]+1)*[-1])
    start_pointer=0
    end_pointer=0
    priority_queue[end_pointer]=sv
    end_pointer+=1

    while start_pointer!=end_pointer:
        curr_node=int(priority_queue[start_pointer])
        priority_queue[start_pointer]=-1
        start_pointer+=1
        #print("removed: "+str(curr_node))
        adjacency_vector=A[curr_node,:]
        #print("neighbours: "+str(np.nonzero(adjacency_vector)[1]))
        for neighbour in np.nonzero(adjacency_vector)[1]:
            #the other endpoint of this edge has not beed explored so we assign it a sign and add it to the queue
            if sign_of_node[neighbour]==0:
                #assign according to algorithm in zaslavsky paper
                sign_of_node[neighbour]=sign_of_node[curr_node]*A[curr_node,neighbour]
                priority_queue[end_pointer]=neighbour
                end_pointer+=1
                #print("this neighbour has not been visited so is being added to the list: "+str(neighbour))
            #check if this edge is negative after the switch
            if sign_of_node[neighbour]*A[curr_node,neighbour]*sign_of_node[curr_node] < 0 :
                return False
    return True


def is_balanced(A):
    N = A.shape[0]
    if N < 3:
        return True

    first = np.argmax(np.sum(np.abs(A), axis=0))
    switch = A[first, :]

    s = switch<=-1
    neg = np.zeros(N)
    negi = sparse.find(s)[1]
    neg[negi]=1

    s = switch>=1
    pos = np.zeros(N)
    posi = sparse.find(s)[1]
    pos[posi]=1

    finished = False
    while not finished:
        posi = sparse.find(pos)[1]
        rows,cols=sparse.find(A[posi, :]>=1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)
        rows,cols=sparse.find(A[posi, :]<=-1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)

        negi = sparse.find(neg)[1]
        rows,cols=sparse.find(A[negi, :]>=1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)
        rows,cols=sparse.find(A[negi, :]<=-1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)

        pos = pos[0]
        neg = neg[0]

        balanced = pos.T.dot(neg) == 0
        switch = pos-neg
        finished = switch.T.dot(switch)==A.shape[0]
        if not balanced:
            break
    return balanced


def get_indicator_vector(A):
    N = A.shape[0]
    first = np.argmax(np.sum(np.abs(A), axis=0))
    switch = A[first, :]

    s = switch==-1
    neg = np.zeros(N)
    negi = sparse.find(s)[1]
    neg[negi]=1

    s = switch==1
    pos = np.zeros(N)
    posi = sparse.find(s)[1]
    pos[posi]=1

    finished = False
    while not finished:
        posi = sparse.find(pos)[1]
        rows,cols=sparse.find(A[posi, :]==1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)
        rows,cols=sparse.find(A[posi, :]==-1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)

        negi = sparse.find(neg)[1]
        rows,cols=sparse.find(A[negi, :]==1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)
        rows,cols=sparse.find(A[negi, :]==-1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)

        pos = pos[0]
        neg = neg[0]

        balanced = pos.T.dot(neg) == 0
        switch = pos-neg
        finished = switch.T.dot(switch)==A.shape[0]
        if not balanced:
            break
    return switch


def create_graph(edgelist):
    graph = {}
    for e1, e2 in edgelist:
        graph.setdefault(e1, []).append(e2)
        graph.setdefault(e2, []).append(e1)
    return graph

# def DFSUtil(A, v,visited,vDegree,k): 
#     # Mark the current node as visited 
#     visited[v] = True
#     # Recur for all the vertices adjacent to this vertex 
#     i_dec = np.zeros(shape=len(A[v,:].nonzero()[1]), dtype=bool)
#     print("\n")
#     for n_i, i in enumerate(A[v, :].nonzero()[1]):
#         # vDegree of v is less than k, then vDegree of 
#         # adjacent must be reduced 
#         print((v, vDegree[v]), (i, vDegree[i]))
#         if vDegree[v] < k: 
#             i_dec[n_i] = True
#             vDegree[i] = vDegree[i] - 1
#         # If adjacent is not processed, process it 
#         if visited[i]==False: 
#             # If vDegree of adjacent after processing becomes 
#             # less than k, then reduce vDegree of v also 
#             # if (DFSUtil(A, i,visited,vDegree,k)): 
#                 # vDegree[v]-=1
#             print (DFSUtil(A, i,visited,vDegree,k))
#         print((v, vDegree[v]), (i, vDegree[i]))
#     for n_i, i in enumerate(A[v, :].nonzero()[1]):
#         if ((vDegree[v] < k) and (not(i_dec[n_i]))):
#             print("Not dec ?", i_dec[n_i])
#             i_dec[n_i] = True
#             vDegree[i] = vDegree[i] - 1
#     # Return true if vDegree of v is less than k 
#     return vDegree[v] < k 

# def find_kcore (A, k):
#     visited = np.zeros(A.shape[0], dtype=bool)
#     vdegree = np.zeros(A.shape[0])
#     for i in range(A.shape[0]):
#         vdegree[i] = len(A[i, :].nonzero()[1])
#     print(vdegree)
#     DFSUtil(A, 0, visited, vdegree, k)
#     for i in range(A.shape[0]):
#         if (not(visited[i])):
#             DFSUtil(A, i, visited, vdegree, k)
#     kcores = []
#     for i in range(A.shape[0]):
#         if (vdegree[i] >= k):
#             kcores.append(i)
#     Ak = A[kcores, :][:, kcores]
#     for i, j in zip(kcores, range(Ak.shape[0])):
#         print((i, vdegree[i]), (j, len(Ak[j, :].nonzero()[1])))
#     components = find_connected_components(Ak)
#     largest = components[np.argmax([len(i) for i in components])]
#     max_Ak = Ak[largest,:][:,largest]
#     return max_Ak

def find_kcore(A, kc):
    active = np.ones(shape=A.shape[0], dtype=bool)
    vdegree = np.zeros(shape=A.shape[0], dtype=int)
    for v in range(A.shape[0]):
        vdegree[v] = len(A[v,:].nonzero()[1])
    for k in range(1, kc+1):
        all_above_k = False
        while (not(all_above_k)):
            all_above_k = True
            for v in range(A.shape[0]):
                if ((active[v]) and (vdegree[v] < k)):
                    all_above_k = False
                    active[v] = False
                    for u in A[v,:].nonzero()[1]:
                        vdegree[u] -= 1
        if (len(np.where(active)[0]) == 0):
            raise Exception("Degeneracy exceeded")
    kcores = np.where(active)[0]
    print(A.dtype)
    Ak = A[kcores, :][:, kcores]
    print(Ak.dtype)
    # for i, j in zip(kcores, range(Ak.shape[0])):
    #     print((i, vdegree[i]), (j, len(Ak[j, :].nonzero()[1])))
    components = find_connected_components(Ak)
    largest = components[np.argmax([len(i) for i in components])]
    max_Ak = Ak[largest,:][:,largest]
    return max_Ak
                    

def find_degeneracy (A, perc=0.1):
    active = np.ones(shape=A.shape[0], dtype=bool)
    vdegree = np.zeros(shape=A.shape[0])
    for v in range(A.shape[0]):
        vdegree[v] = len(A[v,:].nonzero()[1])
    k = 1
    nnodes = A.shape[0]
    while(np.any(active)):
        all_above_k = False
        while (not(all_above_k)):
            all_above_k = True
            for v in range(A.shape[0]):
                if ((active[v]) and (vdegree[v] < k)):
                    all_above_k = False
                    active[v] = False
                    for u in A[v,:].nonzero()[1]:
                        vdegree[u] -= 1
        if (np.sum(active) < (perc * nnodes)):
            break
        k += 1
    return (k-1)

# Prim's
def mst(start, A, graph):
    signs_of_nodes={}
    signs_of_nodes[start]=1
    closed = set()
    rows = []
    cols = []
    edges = []
    q = [(start, start)]
    while q:
        v1, v2 = q.pop()
        if v2 in closed:
            continue
        closed.add(v2)
        if signs_of_nodes[v1]*A[v1,v2]<0:
            signs_of_nodes[v2]=-1
        else:
            signs_of_nodes[v2]=1

        edges.append((v1, v2))
        rows.append(v1)
        cols.append(v2)
        for v in graph[v2]:
            if v in graph:
                q.append((v2, v))
    del edges[0]
    assert len(edges) == len(graph)-1
    return edges,rows[1:], cols[1:],signs_of_nodes

def print_stats(A):
    d,W = splinalg.eigsh(A, k=1)
    v = W[:,0]
    x = np.sign(v)
    edges = np.sum(np.abs(A))/2.#float(sparse.spmatrix.dot(x.T, A).dot(x))/2
    degrees = np.array(np.sum(np.abs(A), axis=0))
    x1 = np.where(x==1, 1, 0)
    x2 = np.where(x==-1, 1, 0)
    edges1 = float(sparse.spmatrix.dot(x1.T, A).dot(x1))/2
    edges2 = float(sparse.spmatrix.dot(x2.T, A).dot(x2))/2
    edges_ac = -float(sparse.spmatrix.dot(x1.T, A).dot(x2))

    N = A.shape[0]
    M = np.sum(np.abs(A))/2.
    print('balanced: {}'.format(is_balanced(A)))
    print('size: {}'.format(A.shape[0]))
    print('edges: {}'.format(edges))
    print('edges_C1: {}'.format(edges1))
    print('edges_C2: {}'.format(edges2))
    print('edges_across: {}'.format(edges_ac))
    print('mean_degree: {}'.format(np.mean(degrees)))
    print('median_degree: {}'.format(np.median(degrees)))
    print('max_degree: {}'.format(np.max(degrees)))
    print('min_degree: {}'.format(np.min(degrees)))
    print('C1: {}'.format((x1.T.dot(x1))))
    print('C2: {}'.format((x2.T.dot(x2))))
    print('x_size: {}'.format(x.shape))
    print('x_nnz: {}'.format(x.dot(x)))
    print()
