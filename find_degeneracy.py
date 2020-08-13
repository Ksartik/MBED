import utils
import json

dataset_dir='Datasets/'
datasets= json.load(open("datasets_info.json", "r"))

for dataset, dataset_info in datasets.items():
    input_file=dataset_dir+dataset
    dataset_name=dataset.split('/')[0]
    A = utils.read_adj_sparse_matrix(input_file, comment=dataset_info['comment'], 
                                sep=dataset_info['sep'], min_node=dataset_info['min_node'])
    num_pos_e = 0
    num_neg_e = 0
    edges = A.nonzero()
    for i, j in zip(edges[0], edges[1]):
        num_pos_e += 1 if (A[i, j] == +1) else 0
        num_neg_e += 1 if (A[i, j] == -1) else 0
    print(dataset, num_pos_e/2, num_neg_e/2)
    # print(dataset, utils.find_degeneracy(A, perc=0.1))

# BitcoinOTC/soc-sign-bitcoinotc.csv 21
# BitcoinAlpha/soc-sign-bitcoinalpha/out.soc-sign-bitcoinalpha 19
# Chess/chess/out.chess 15
# Cloister/moreno_sampson/out.moreno_sampson_sampson 9
# Congress/convote/out.convote 5
# Epinions/soc-sign-epinions.txt 121
# HighlandTribes/ucidata-gama/out.ucidata-gama 5
# Slashdot/soc-sign-Slashdot090221.txt 54
# WikiConflict/wikiconflict/out.wikiconflict 131
# WikipediaElections/elec/out.elec 53
# WikiPolitics/wikisigned-k2/out.wikisigned-k2 54

# BitcoinOTC/soc-sign-bitcoinotc.csv 9
# BitcoinAlpha/soc-sign-bitcoinalpha/out.soc-sign-bitcoinalpha 10
# Chess/chess/out.chess 11
# Cloister/moreno_sampson/out.moreno_sampson_sampson 9
# Congress/convote/out.convote 5
# Epinions/soc-sign-epinions.txt 11
# HighlandTribes/ucidata-gama/out.ucidata-gama 5
# Slashdot/soc-sign-Slashdot090221.txt 18
# WikiConflict/wikiconflict/out.wikiconflict 47
# WikipediaElections/elec/out.elec 48
# WikiPolitics/wikisigned-k2/out.wikisigned-k2 14

# BitcoinOTC/soc-sign-bitcoinotc.csv 18281.0 3153.0
# BitcoinAlpha/soc-sign-bitcoinalpha/out.soc-sign-bitcoinalpha 12769.0 1312.0
# Chess/chess/out.chess 19046.0 13604.0
# Cloister/moreno_sampson/out.moreno_sampson_sampson 56.0 54.0
# Congress/convote/out.convote 414.0 107.0
# Epinions/soc-sign-epinions.txt 589888.0 118619.0
# HighlandTribes/ucidata-gama/out.ucidata-gama 29.0 29.0
# Slashdot/soc-sign-Slashdot090221.txt 380933.0 117599.0
# WikiConflict/wikiconflict/out.wikiconflict 559094.0 901964.0
# WikipediaElections/elec/out.elec 78440.0 21915.0
# WikiPolitics/wikisigned-k2/out.wikisigned-k2 628000.0 84337.0