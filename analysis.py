import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

my_dpi = 150

def quality_plot (dfm, dataset, nH, nS0):
    fig = plt.figure(figsize=(2.5, 2.3), dpi=my_dpi)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset]
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        dfm_data_method = dfm_data.loc[dfm_data["method"] == method]
        plt.plot(dfm_data_method["Budget"], dfm_data_method["Delta_perc"], shape, color=col, 
                linewidth=1, markersize=6, label=label)
    plt.plot([], [], color='w', label=' ', alpha=0)
    ax = plt.gca()
    handles,leglabels = ax.get_legend_handles_labels()
    order = [0, 1, 5, 4, 3, 2, 6]
    handles = [handles[i] for i in order]
    leglabels = [leglabels[i] for i in order]
    plt.legend(handles, leglabels, loc='best', ncol=2, fontsize=8, frameon=False)
    plt.xlabel('Budget (b)',fontsize=14)
    plt.ylabel("IB (%)",fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    fig.savefig('Plots/Quality/' + dataset + '.png', bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

def time_plot (dfm, dataset, nH, nS0):
    fig = plt.figure(figsize=(2.5, 2.3), dpi=my_dpi)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset]
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        dfm_data_method = dfm_data.loc[dfm_data["method"] == method]
        plt.plot(dfm_data_method["Budget"], dfm_data_method["RT"]/60, shape, color=col, 
                linewidth=1, markersize=6, label=label)
    ax = plt.gca()
    handles,leglabels = ax.get_legend_handles_labels()
    order = [0, 1, 5, 4, 3, 2, 6]
    handles = [handles[i] for i in order]
    leglabels = [leglabels[i] for i in order]
    plt.legend(handles, leglabels, loc='best', ncol=2, fontsize=8, frameon=False)
    plt.xlabel('Budget (k)',fontsize=14)
    plt.ylabel('Running Time (min)',fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    fig.savefig('Plots/Time/' + dataset + '.png', bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

df = pd.read_csv("results.csv")
dfm = df.groupby(["Dataset", "method", "nS0", "nH", "Budget"], as_index=False).mean()
dfm["dataset_info"] = df["Dataset"] + " (|H|=" + str(df["nH"]) + ", |S0|=" + str(df["nS0"]) + ")"
dfm["Delta_perc"] = dfm["Delta"]/(dfm["nH"] - dfm["nS0"]) * 100

methods = ["nonspec", "rand_nonspec", "spec_iter", "spec_oneshot", "rand_cand", "min_cep"]
colors = ["royalblue", "cornflowerblue", "hotpink", "chocolate", "black", "green"]
shapes = ['-o', '-+', '->', '-s', '-x', '-*']
labels = ['Greedy', 'RG', 'ISA', 'SPEC-TOP', 'Random', 'MIN-CEP']

datasets = ["BitcoinOTC", "BitcoinAlpha", "Chess", "WikipediaElections",
            "Epinions", "Slashdot", "WikiConflict", "WikiPolitics"]

df = df.loc[df["method"].isin(methods)]
dfm = dfm.loc[dfm["method"].isin(methods)]

for dataset in datasets:
    dfd = df.loc[df["Dataset"] == dataset]
    nH = np.unique(dfd['nH']).item()
    nS0 = np.unique(dfd['nS0']).item()
    quality_plot(dfm, dataset, nH, nS0)
    # time_plot(dfm, dataset, nH, nS0)
