# Balance Maximization in Signed Networks via Edge Deletions
Supplementary code for the work by the same name submitted to WSDM 2021.

# Data & tools
We use 8 datasets (signed networks) in total all of which are available at <www.konect.cc>. To obtain the actual list of datasets used, please refer the paper. The folder **Datasets/** gives an example of two such datasets (Bitcoin graphs).

We use Python 3.6.9 (and 3.8.2) for this work. One can install the required libraries by running `pip install -r requirements.txt`. 

# Finding the initial MBS
We use the TIMBAL algorithm (Ordozgoiti et al. 2020) to find the initial maximum balanced subgraph (MBS). The code for the same is cloned from the [Github repository](https://github.com/justbruno/finding-balanced-subgraphs). We add one function for our purposes in the end of this which follows TIMBAL. We give this code along with the original TIMBAL in the folder **finding-balanced-graphs/**.

The folder **Results/** give 2 such examples (Bitcoin graphs) for the calculated initial MBS calculated as *s_nodes.txt* on the subgraph *adj_mat.txt* (in this case, the largest connected component). 

# Proposed baselines/algorithms to solve MBED
Our main algorithm is implemented in the file `mbed_nonSpectral.py` (both Greedy and Randomized-Greedy). The baselines are implemented in these files:

1. MIN-CEP: `mbed_min_cep.py`
2. Random: `mbed_random_candidate.py`
3. Iterated Spectral Algorithm(ISA): `mbed_spec_iter.py`
4. SPEC-TOP: `mbed_spec_top.py`

# Solving MBED 
## On the largest connected component(LCC)

> `python3 mbed_test.py --datasets <datasets> --method <method_name> --budgets <budgets>`

where 
- `<datasets>` denotes the list of datasets to solve on (passed as a space-separated list). 
- `<method_name>` denotes the *method* that is to be tested. This must be passed as: nonspec (for Greedy), rand_nonspec (for RG), min_cep (for MIN-CEP), rand_cand (for Random), spec_iter (for ISA), spec_top (for spec_top). 
- `<budgets>` denotes the list of budgets to solve on (passed as a space-separated list). This will solve on the highest of these budgets and also maintain the characteristics (running time, current balance) at the other budgets as it deleted those many edges as specified. 

The results will be appended to a file named **results.csv**. Note that this file will find the LCC for each dataset and the initial MBS on it if not already calculated (using MBS), and store it in the Results/. 

## On the k-cores 

It is similar as above:

> `python3 mbed_test.py --datasets <datasets> --method <method_name> --kcores <kcores>`

where 
- `<kcores>` is a list of values of k (again passed as a space-separated list) to be considered for finding the k-cores of different datasets. 

Now, the results will be appended to another file named **results_kcores.csv**. Note that budget is fixed now and maybe changed by changing the variable inside the code by that name. 

# Analysis 

The following files are used for making plots:

1. `analysis.py`: makes quality plots (and optionally time plots) when varying budget.
2. `analysis_kcore.py`: compares the increase in balance with different values of k. 
3. `analysis_largeB.py`: We use separate codes for large and small budgets. Here we use the percentage of edges deleted when a large budget was chosen. 

# Visualization

Finally, `visualize.py` visualizes a graph (here, BitcoinOTC) for a particular budget (which can be passed). It uses the saved log of the run of certain algorithms (here, Greedy algorithm). This log is saved in order to see which all edges were deleted during a particular run and what is the final MBS (in the files `mbed_test.py`). 
