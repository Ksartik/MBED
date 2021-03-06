import sys
import utils
sys.path.append("./finding-balanced-subgraphs")
import timbal
import numpy as np
from mbed_nonSpectral import mbed_solve as mbed_nonSpectral
from mbed_spec_iter import mbed_solve as mbed_spec_iter
from mbed_spec_top import mbed_solve as mbed_spec_top
from mbed_random_candidate import mbed_solve as mbed_random_candidate
from mbed_min_cep import mbed_solve as mbed_min_cep
import time 
import json
import os
import csv
from argparse import ArgumentParser

parser = ArgumentParser(description="Maximizing balance by deleting edges in a given dataset using method given budget")
parser.add_argument("--datasets", nargs='+', help="Datasets name")
parser.add_argument("--method", type=str, help="Method name")
parser.add_argument("--kcores", nargs='+', type=int, help="kcores to be used")
parser.add_argument("--verbose", action='store_true', dest='verbose')
parser.add_argument("--replace", action='store_true', dest='replace')
parser.add_argument("--only_h0", action='store_true', dest='only_h0')
args = parser.parse_args(sys.argv[1:])

"""
First obtain the _h0 using TIMBAL algorithm. For this we exactly follow the steps in the 
code used to obtain results in the paper Ordozgoiti et al.
"""

def find_init_h0 (input_file, dataset_info, n_iters, kc):
    A = utils.read_adj_sparse_matrix(input_file, comment=dataset_info['comment'], 
                                    sep=dataset_info['sep'], min_node=dataset_info['min_node'])
    print("File read")
    A = utils.find_kcore(A, kc)
    S_max = []
    for i in range(n_iters):
        # TIMBAL
        # We discard all connected components but the largest
        # Subsample => Whether or not to sample random graphs. Recommended when the graph is large
        # Max_removals => Max number of removed vertices per iteration
        # samples => Randomly sampled graphs. Must be a factor of the number of threads used
        # avg_size => Approximate desired size of randomly sampled graph
        start_time = time.time()
        print("TIMBAL started")
        S = timbal.process(A, max_removals=dataset_info['max_removals'], samples=dataset_info['samples'], 
                            avg_size=dataset_info['avg_size'], subsample=dataset_info['subsample'])
        print("TIMBAL finished")
        print ("Time taken to find h0: ", str(time.time() - start_time) + " s")
        if ((len(S_max) < len(S)) and (utils.is_balanced(A[S,:][:,S]))):
            S_max = S
    As = A[S_max,:][:,S_max]
    print("\nH")
    utils.print_stats(A)
    print("\nS(H)")
    utils.print_stats(As)
    return (A, S_max)

def write_balance_file (dataset_name, A, S, dataset_info, kc):
    dataset_res_dir = "Results/" + dataset_name + "/"
    A = utils.write_adj_sparse_matrix(dataset_res_dir + str(kc) + "core_" + "adj_mat.txt", A, sep="\t")
    S = utils.write_snodes(dataset_res_dir + str(kc) + "core_" + "s_nodes.txt", S)

def read_balance_file (dataset_name, dataset_info, kc):
    dataset_res_dir = "Results/" + dataset_name + "/"
    A = utils.read_adj_sparse_matrix(dataset_res_dir + str(kc) + "core_" + "adj_mat.txt", comment=dataset_info['comment'], 
                                    sep="\t", min_node=0)
    S = utils.read_snodes(dataset_res_dir + str(kc) + "core_" + "s_nodes.txt")
    return A, S

def mbed_select (method, A, budgets, S, verbose, dataset_info):
    if (method == "rand_nonspec"):
        res_info, S_new, A_d, edges_removed = mbed_nonSpectral(A, budgets, S, randomize=True, verbose=verbose)
    elif (method == "nonspec"):
        res_info, S_new, A_d, edges_removed = mbed_nonSpectral(A, budgets, S, randomize=False, verbose=verbose)
    elif (method == "spec_iter"):
        res_info, S_new, A_d, edges_removed = mbed_spec_iter(A, budgets, S, verbose=verbose, randomize=False)
    elif (method == "spec_top"):
        res_info, S_new, A_d, edges_removed = mbed_spec_top(A, budgets, S, verbose=verbose)
    elif (method == "rand_cand"):
        res_info, S_new, A_d, edges_removed = mbed_random_candidate(A, budgets, S, verbose=verbose)
    elif (method == "pure_rand"):
        res_info, S_new, A_d, edges_removed = mbed_pure_random(A, budgets, S, verbose=verbose)
    elif (method == "min_cep"):
        res_info, S_new, A_d, edges_removed = mbed_min_cep (A, budgets, S, randomize=True, verbose=verbose)
    else:
        raise Exception("Unidentified method")
    return res_info, S_new, A_d, edges_removed
            

dataset_dir='Datasets/'
datasets= json.load(open("datasets_info.json", "r"))

all_datasets = ["BitcoinOTC", "BitcoinAlpha", "Chess", "Cloister", "Congress", "Epinions", 
            "HighlandTribes", "Slashdot", "WikiConflict", "WikipediaElections", "WikiPolitics"]

mb = 50
bp = 5
n_iters = 10
datasets_to_test = args.datasets if (args.datasets is not None) else all_datasets
method = args.method if (args.method is not None) else exit("""Method not provided: Select one of the following:
        1. pure_rand
        2. rand_cand
        3. spec_top
        4. spec_iter
        5. nonspec
        6. rand_nonspec
        7. min_cep""")
kcores = args.kcores
verbose = args.verbose
replace = args.replace
only_h0 = args.only_h0
budgets = [25]
# if (os.path.exists("Results/")):
#     os.mkdir("Results/")

print(kcores)

write_fields = ["Dataset", "method", "nS0", "nH", "Budget", "kcore", "RT", "Delta"]
with open ("results_kcores.csv", "a") as writef:
    writer = csv.DictWriter(writef, fieldnames=write_fields)
    for dataset, dataset_info in datasets.items():
        print(dataset)
        input_file=dataset_dir+dataset
        dataset_name=dataset.split('/')[0]
        if (dataset_name not in datasets_to_test):
            continue
        for kc in kcores:
            if (replace or (not(os.path.exists("Results/" + dataset_name + "/" + str(kc) + "core_" + "adj_mat.txt")))):
                A, S = find_init_h0(input_file, dataset_info, n_iters, kc)
                if (not(os.path.exists("Results/" + dataset_name))):
                    os.mkdir("Results/" + dataset_name)
                write_balance_file(dataset_name, A, S, dataset_info, kc)
            else:
                A, S = read_balance_file (dataset_name, dataset_info, kc)
            if (only_h0):
                continue
            # _h0 obtained now we do MBED
            # for budget in budgets:
            general_row = {"Dataset": dataset_name, "method": method, "nS0": len(S), "nH": A.shape[0], "kcore": kc}
            print("MBED started")
            start_time = time.time()
            res_info, S_new, A_d, edges_removed = mbed_select (method, A, budgets, S, verbose, dataset_info)
            print("MBED finished")
            print ("Total time taken to solve MBED: ", str(time.time() - start_time) + " s")
            added_nodes = len(S_new) - len(S)
            print ("Delta (Hm) - Delta(H) = " + str(len(S_new)) + " - " + str(len(S)) + " = " + str(added_nodes))
            print("\n")
            As_d = A_d[S_new,:][:,S_new]
            utils.print_stats(As_d)
            for row in res_info:
                row["Dataset"] = dataset_name
                for field in general_row:
                    row[field] = general_row[field]
                writer.writerow(row)
            print("\n")
