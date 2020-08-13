import networkx as nx
import utils
import sys
sys.path.append("./finding-balanced-subgraphs")
import timbal
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle

def read_balance_file (dataset_name, dataset_info):
    dataset_res_dir = "Results/" + dataset_name + "/"
    A = utils.read_adj_sparse_matrix(dataset_res_dir + "adj_mat.txt", comment=dataset_info['comment'], 
                                    sep="\t", min_node=0)
    S = utils.read_snodes(dataset_res_dir + "s_nodes.txt")
    return A, S

datasets = json.load(open("datasets_info.json", "r"))

dataset_name = "BitcoinOTC"
budget = int(sys.argv[1])
# if (budget != 0):
#     chosen_edges = "deledges_rand_nonspec_mb" + str(budget) + "_H5872_s4487.txt"
# else:
#     chosen_edges = "deledges_rand_nonspec_mb50_H5872_s4487.txt"
chosen_edges = "deledges_nonspec_mb50_H5872_s4487.txt"

dataset_info = [y for x, y in datasets.items() if (x.split("/")[0] == dataset_name)][0]

A, S = read_balance_file(dataset_name, dataset_info)

G = nx.from_numpy_matrix(A.todense())

# 
ind = utils.get_indicator_vector(A[S, :][:, S])
x_v = np.zeros(shape=A.shape[0])
x_v[S] = ind
s_nodes_p = list(np.where(x_v == +1)[0])
s_nodes_m = list(np.where(x_v == -1)[0])

# 
all_edges = list(G.edges)
out_edges = [e for e in all_edges if ((e[0] not in S) or (e[1] not in S))]
del_edges = utils.read_deledges("Visualization/" + dataset_name + "/" + chosen_edges)[:budget]
print(len(del_edges))
del_edges_p = [e for e in del_edges if (A[e] == +1)]
del_edges_m = [e for e in del_edges if (A[e] == -1)]

# 
Ad = A.copy()
for e in del_edges:
    Ad = utils.delete_edge(Ad, e)

S_new = timbal.process_only_second (Ad, S)
s_nodes = S
added_nodes = list(set(S_new).difference(set(S)))
ind = utils.get_indicator_vector(Ad[S_new, :][:, S_new])
x_v = np.zeros(shape=Ad.shape[0])
x_v[S_new] = ind
added_nodes_p = [i for i in added_nodes if (x_v[i] == +1)]
added_nodes_m = [i for i in added_nodes if (x_v[i] == -1)]
other_nodes = set(G.nodes).difference(S_new)

# 
# 
Gr = nx.Graph()
Gr.add_nodes_from(G.nodes)
Gr.add_edges_from(out_edges)
# Gr.add_edges_from(del_edges_p + del_edges_m)

plt.figure(num=None, figsize=(2, 2), dpi=150)
plt.axis('off')
fig = plt.figure(1)

if (len(sys.argv) > 2):
    pos = pickle.load(open(sys.argv[2], 'rb'))
else:
    pos = {}
    Rs = 0.85
    for n in s_nodes_p:
        r = Rs * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        pos[n] = (-1 + r*np.cos(theta), r*np.sin(theta))

    for n in s_nodes_m:
        r = Rs * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        pos[n] = (1 + r*np.cos(theta), r*np.sin(theta))

    for n in (added_nodes):
        theta = np.random.uniform(60*np.pi/180, 120*np.pi/180)
        lower_r = 1/((np.cos(theta)**2/4 + np.sin(theta)**2/2)**0.5)
        # r = np.random.uniform(1, lower_r)
        r = np.random.uniform(0.75*lower_r, 1.25*lower_r)
        pos[n] = (r*np.cos(theta), r*np.sin(theta))

    for n in (other_nodes):
        lower_theta = 30 * np.pi/180
        theta = np.random.uniform(lower_theta, np.pi - lower_theta)
        # if (np.random.uniform() > 0.5):
        #     theta = np.random.uniform(lower_theta, np.pi - lower_theta)
        # else:
        #     theta = np.random.uniform(np.pi + lower_theta, 2*np.pi - lower_theta)
        lower_r = 1/((np.cos(theta)**2/4 + np.sin(theta)**2/2)**0.5)
        # r = np.random.uniform(lower_r, 2)
        r = np.random.uniform(0.75*lower_r, 1.25*lower_r)
        pos[n] = (r*np.cos(theta), r*np.sin(theta))

    pickle.dump(pos, open("Visualization/" + dataset_name + str(budget) + ".pkl", "wb"))

snodes_p_draw = nx.draw_networkx_nodes(Gr,pos,node_size=15,nodelist=s_nodes_p, node_color='green', 
                        node_shape='o')#, edgecolors='black')
snodes_m_draw = nx.draw_networkx_nodes(Gr,pos,node_size=15,nodelist=s_nodes_m, node_color='darkorange', 
                        node_shape='o')#, edgecolors='black')
added_nodes_p_draw = nx.draw_networkx_nodes(Gr,pos,node_size=15,nodelist=added_nodes_p, 
                        node_color='green', node_shape='s', edgecolors='black', linewidths=0.27)
added_nodes_m_draw = nx.draw_networkx_nodes(Gr,pos,node_size=15,nodelist=added_nodes_m, 
                        node_color='darkorange', node_shape='s', edgecolors='black', linewidths=0.27)
other_nodes_draw = nx.draw_networkx_nodes(Gr,pos,node_size=5,nodelist=other_nodes, 
                        node_color='grey', node_shape='o', edgecolors='grey')
deledges_p_draw = nx.draw_networkx_edges(Gr, pos, width=1.3, edgelist=del_edges_p, 
                        edge_color='red', style='dashed')
deledges_m_draw = nx.draw_networkx_edges(Gr, pos, width=1.3, edgelist=del_edges_m, 
                        edge_color='mediumblue', style='dashed')

other_edges = list(set(out_edges).difference(set(del_edges)))
nx.draw_networkx_edges(Gr, pos, width=1, edgelist=[e for e in other_edges if (A[e] == +1)], 
                        edge_color='orangered', alpha=0.1)
nx.draw_networkx_edges(Gr, pos, width=1, edgelist=[e for e in other_edges if (A[e] == -1)], 
                        edge_color='cornflowerblue', alpha=0.1)

try:
    snodes_p_draw.set_zorder(20)
    snodes_m_draw.set_zorder(20)
    added_nodes_p_draw.set_zorder(10)
    added_nodes_m_draw.set_zorder(10)
    other_nodes_draw.set_zorder(2)
    deledges_p_draw.set_zorder(20)
    deledges_m_draw.set_zorder(20)
except:
    pass

plt.axis('off')
plt.tight_layout()
# plt.subplots_adjust(wspace=-5,hspace=-5)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# plt.show()
plt.savefig("Visualization/BitcoinOTC" + str(budget) + ".png", bbox_inches = 'tight',
    pad_inches = 0)
