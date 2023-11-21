import collections
import json
from argparse import ArgumentParser
import numpy as np
import pymannkendall as mk
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from sklearn.metrics import cohen_kappa_score

import pandas as pd

p = ArgumentParser()
p.add_argument("--data", '-d', default='deuparl')
p.add_argument("--pos", action='store_true')
p.add_argument("--random", action='store_true')
p.add_argument("--balance", action='store_true')
p.add_argument("--parsers", default="stanza,biaffine,corenlp,stackpointer,towerparse")
args = p.parse_args()
data = args.data

# random

#df1 = pd.read_csv(f'tables/trend_test/{data}_random.csv')
df1 = pd.read_csv(f'tables/trend_test/{data}_random_no_aggregation.csv')

# balance

df2 = pd.read_csv(f'tables/trend_test/{data}_balance_no_aggregation_sorted.csv')
#df2 = pd.read_csv(f'tables/trend_test/{data}_balance.csv')


df = pd.concat([df1, df2], ignore_index=True)

results = collections.defaultdict(lambda : collections.defaultdict(list))
parsers = ['corenlp', 'stanza', 'biaffine', 'stackpointer', 'towerparse']

metrics = ['ndd', 'mdd', 'height', 'left_child_ratio', 'k_ary', 'num_leaves', 'degree_var',
                       'degree_mean',
                       'topo_edit_distance', 'tree_edit_distance', 'root_distance', 'longest_path', 'num_crossing',
                       'depth_var', 'depth_mean']
df = df[df.metric.isin(metrics)]
for i in range(len(parsers)):
#for parser1 in parsers:
    parser1 = parsers[i]
    p1 = df[df.parser == parser1]
    #for parser2 in parsers:
    #for j in range(i+1, len(parsers)):
    for j in range(len(parsers)):
        parser2 = parsers[j]
        #if parser1 == parser2:
        #    continue
        print(p1)
       # raise ValueError
        p2 = df[df.parser == parser2]
        assert len(p1) == len(p2)
        #p1 = p1[metrics]
        #p2 = p2[metrics]
        '''
        for metric in ['ndd', 'mdd', 'height', 'left_child_ratio', 'k_ary', 'num_leaves', 'degree_var',
                       'degree_mean',
                       'topo_edit_distance', 'tree_edit_distance', 'root_distance', 'longest_path', 'num_crossing',
                       'depth_var', 'depth_mean']:
        '''
        #if 'corenlp' in [parser1, parser2] and metric == 'num_crossing':
        #    continue
        #kappa = cohen_kappa_score(p1['OriMannKendall'].values, p2['OriMannKendall'].values)
        kappa = cohen_kappa_score(p1['PartialMannKendall'].values, p2['PartialMannKendall'].values)
        results[parser1][parser2].append(kappa)

        #print(f"{parser1} vs. {parser2}: {kappa}")

for i in range(len(parsers)):
#for parser1 in parsers:
    parser1 = parsers[i]
    p1 = df[df.parser == parser1]
    #for parser2 in parsers:
    #for j in range(i+1, len(parsers)):
    for j in range(len(parsers)):
        parser2 = parsers[j]
        #if parser1 == parser2:
        #    continue

        p2 = df[df.parser == parser2]
        assert len(p1) == len(p2)
        for metric in ['ndd', 'mdd', 'height', 'left_child_ratio', 'k_ary', 'num_leaves', 'degree_var',
                       'degree_mean',
                       'topo_edit_distance', 'tree_edit_distance', 'root_distance', 'longest_path', 'num_crossing',
                       'depth_var', 'depth_mean']:
            if 'corenlp' in [parser1, parser2] and metric == 'num_crossing':
                continue
            kappa = cohen_kappa_score(p1['OriMannKendall'].values, p2['OriMannKendall'].values)
            #kappa = cohen_kappa_score(p1['PartialMannKendall'].values, p2['PartialMannKendall'].values)
            results[parser1][parser2].append(kappa)

            print(f"{parser1} vs. {parser2} for {metric}: {kappa}")

averaged_results = collections.defaultdict(dict)
mat = np.zeros((len(parsers)+1, len(parsers)))
for i, parser1 in enumerate(results.keys()):
    for j, parser2 in enumerate(results[parser1].keys()):
        averaged_results[parser1][parser2] = np.mean(results[parser1][parser2])
        mat[i][j] = np.mean(results[parser1][parser2])
    averaged_results[parser1]['avg'] = np.mean(list(averaged_results[parser1].values()))
    #mat[i][j]

for i in range(len(parsers)):
    mat[-1, i] = np.mean([x for j, x in enumerate(mat[:-1, i]) if j != i])
#mat[-1:, :] = np.mean([mat[:-1, :], axis=1])
print(mat)
averaged_results = pd.DataFrame(averaged_results)
print(averaged_results)

import matplotlib.pyplot
import seaborn as sns
sns.set_style('dark')

#plt.figure()
fig, ax = plt.subplots()
cax = ax.matshow(mat, cmap=plt.cm.Reds)
fig.colorbar(cax)

for i in range(len(parsers)):
    for j in range(len(parsers)+1):
        ax.text(i, j, "%.2f" % mat[j, i], va='center', ha='center')

ax.set_yticks(range(len(parsers)+1))
ax.set_xticks(range(len(parsers)))
ax.set_yticklabels(parsers+['avg'])
ax.set_xticklabels(parsers, rotation=45)
plt.tight_layout()
plt.savefig(f'plots/{data}/{data}_parser_corr_all.png', dpi=300,bbox_inches='tight', pad_inches=0.0)
plt.show()


