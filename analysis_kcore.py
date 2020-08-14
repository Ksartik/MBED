import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

my_dpi = 150

def quality_plot (dfm, dataset):
    fig = plt.figure(figsize=(2.5, 2.3), dpi=my_dpi)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset]
    plt.xticks(np.unique(dfm_data["kcore"]), fontsize = 10)
    plt.yticks(fontsize = 10)
    indices = np.unique(dfm_data["kcore"])
    width = np.min(np.diff(indices))/(len(methods)+1)/1.2
    nmethod = 0
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        dfm_data_method = dfm_data.loc[dfm_data["method"] == method]
        # plt.plot(dfm_data_method["kcore"], dfm_data_method["Delta_perc"], shape, color=col, 
        #         linewidth=1, markersize=6, label=label)
        plt.bar(dfm_data_method["kcore"] - width*((len(methods))/2 - nmethod),
                dfm_data_method["Delta_perc"], align='edge', hatch=shape, width=width, 
                color=col, label=label, alpha=0.85)
        nmethod += 1
    plt.legend(loc='best', frameon=0, fontsize=8, ncol=2)
    plt.xlabel('k-core',fontsize=14)
    plt.ylabel("IB (%)",fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    fig.savefig('Plots/Parameter/Quality/' + dataset + '.png', bbox_inches = 'tight', 
                pad_inches = 0.07)
    plt.close()

def time_plot (dfm, dataset):
    fig = plt.figure(figsize=(3, 3), dpi=my_dpi)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset]
    plt.xticks(np.unique(dfm_data["kcore"]), fontsize = 10)
    plt.yticks(fontsize = 10)
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        dfm_data_method = dfm_data.loc[dfm_data["method"] == method]
        plt.plot(dfm_data_method["kcore"], dfm_data_method["RT"]/60, shape, color=col, 
                linewidth=1, markersize=7, label=label)
    plt.legend(loc='best', frameon=0, fontsize=6)
    plt.xlabel('k-core',fontsize=10)
    plt.ylabel('Running Time (min)',fontsize=10)
    fig.tight_layout()
    fig.savefig('Plots/Parameter/Time/' + dataset + '.png')
    plt.close()

df = pd.read_csv("results_kcores.csv")
df = df.loc[df["Budget"] == 50]
dfm = df.groupby(["Dataset", "method", "nS0", "nH", "kcore", "Budget"], as_index=False).mean()
dfm["dataset_info"] = df["Dataset"] + " (|H|=" + str(df["nH"]) + ", |S0|=" + str(df["nS0"]) + ")"
dfm["Delta_perc"] = dfm["Delta"]/(dfm["nH"] - dfm["nS0"]) * 100

methods = ["nonspec", "rand_nonspec", "spec_iter", "min_cep"]
colors = ["royalblue", "cornflowerblue", "hotpink", "green"]
shapes = ['OO', '++', '\\\\\\', '**']
labels = ['Greedy', 'RG', 'ISA', 'MIN-CEP']

datasets = ["BitcoinOTC", "BitcoinAlpha", "Chess", "WikipediaElections",
            "Epinions", "Slashdot", "WikiConflict", "WikiPolitics"]

for dataset in datasets:
    dfd = df.loc[df["Dataset"] == dataset]
    quality_plot(dfm, dataset)
    # time_plot(dfm, dataset)
