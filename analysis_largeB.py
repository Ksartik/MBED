import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

my_dpi = 150

chars_datasets = pd.DataFrame({
    "Dataset": ['BitcoinAlpha', 'BitcoinOTC', 'Chess', 'Cloister', 'Congress',
       'Epinions', 'HighlandTribes', 'Slashdot', 'WikiConflict',
       'WikiPolitics', 'WikipediaElections'],
    "nH": [3772, 5872, 6601, 18, 219, 119070, 16, 82052, 
        96243, 137713, 7066],
    "nS0": [2903, 4487, 3477, 10, 208, 81385, 13, 51486, 
        53542, 68037, 3857],
    "out_edges": [6641, 11130, 27891, 81, 69, 453890, 23, 360238, 1014132, 481773, 79407]
}).set_index("Dataset")

def quality_plot (dfm, dataset, out_edges, npoints=10):
    fig = plt.figure(figsize=(2.5, 2.3), dpi=my_dpi)
    eps = 0.005
    plt.xticks([0, 0.025, 0.05, 0.075, 0.1], fontsize = 10)
    plt.yticks(fontsize = 10)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset].reset_index()
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        try:
            dfm_data_method = dfm_data.loc[dfm_data["method"] == method].reset_index(drop=True)
            dfm_data_method["Budget perc"] = dfm_data_method["Budget"]/out_edges * 100
            dfm_data_method = dfm_data_method.loc[dfm_data_method["Budget perc"] <= 0.1].reset_index(drop=True)
            dfm_data_method = dfm_data_method.loc[[int(i * (len(dfm_data_method.index) - 1)/(npoints-1)) 
                                                        for i in range(npoints)]]
            plt.plot(dfm_data_method["Budget perc"], dfm_data_method["Delta_perc"], shape, color=col, 
                        linewidth=1, markersize=6, label=label)
        except:
            pass
    plt.plot([], [], color='w', label=' ', alpha=0)
    plt.plot([], [], color='w', label=' ', alpha=0)
    ax = plt.gca()
    handles,leglabels = ax.get_legend_handles_labels()
    order = [0, 5, 4, 3, 2, 1, 6, 7]
    handles = [handles[i] for i in order]
    leglabels = [leglabels[i] for i in order]
    plt.legend(handles, leglabels, loc='best', fontsize=8, frameon=False, ncol=2)
    plt.xlabel('ED (%)',fontsize=14)
    plt.ylabel("IB (%)",fontsize=14)
    plt.xlim((-0 - eps, 0.1 + eps))
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    fig.savefig('Plots/LargeB/Quality/' + dataset + '.png', bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

def time_plot (dfm, dataset, nH, nS0):
    fig = plt.figure(figsize=(3, 2), dpi=my_dpi)
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    dfm_data = dfm.loc[dfm["Dataset"] == dataset]
    for method, col, shape, label in zip(methods, colors, shapes, labels):
        dfm_data_method = dfm_data.loc[dfm_data["method"] == method]
        plt.plot(dfm_data_method["Budget"], dfm_data_method["RT"]/60, shape, color=col, 
                linewidth=1, markersize=6, label=label)
    plt.legend(loc='best', frameon=0, fontsize=6)
    plt.xlabel('Budget (k)',fontsize=8)
    plt.ylabel('Running Time (min)',fontsize=8)
    fig.tight_layout()
    fig.savefig('Plots/LargeB/Time/' + dataset + '.png')
    plt.close()

df = pd.read_csv("results_largeB.csv")
dfm = df.groupby(["Dataset", "method", "nS0", "nH", "Budget"], as_index=False).mean()
dfm["dataset_info"] = df["Dataset"] + " (|H|=" + str(df["nH"]) + ", |S0|=" + str(df["nS0"]) + ")"
dfm["Delta_perc"] = dfm["Delta"]/(dfm["nH"] - dfm["nS0"]) * 100

methods = ["nonspec", "rand_nonspec", "spec_iter", "spec_oneshot", "rand_cand", "min_cep"]
colors = ["royalblue", "cornflowerblue", "hotpink", "chocolate", "black", "green"]
shapes = ['-o', '-+', '->', '-s', '-x', '-*']
labels = ['Greedy', 'RG', 'ISA', 'SPEC-TOP', 'Random', 'MIN-CEP']

# datasets = ["BitcoinOTC", "BitcoinAlpha", "Chess", "WikipediaElections",
datasets = ["Epinions", "Slashdot", "WikiConflict", "WikiPolitics"]

df = df.loc[df["method"].isin(methods)]
dfm = dfm.loc[dfm["method"].isin(methods)]

for dataset in datasets:
    dfd = df.loc[df["Dataset"] == dataset]
    quality_plot(dfm, dataset, chars_datasets.loc[dataset]["out_edges"], npoints=5)
    # time_plot(dfm, dataset, nH, nS0)


