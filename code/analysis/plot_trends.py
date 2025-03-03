import collections
from collections import defaultdict
import json
from argparse import ArgumentParser
import numpy as np
import pymannkendall as mk
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import sys
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import os 

sys.path.insert(0, '..')
from sklearn.metrics import cohen_kappa_score

import pandas as pd


#parsers = ['corenlp', 'stanza', 'biaffine', 'stackpointer', 'towerparse', 'crf2o_de_merged_proj,crf2o_en_merged_proj']
parsers = ['corenlp', 'stanza', 'biaffine_roberta', 'stackpointer', 'towerparse', 'crf2o_lstm']

p2name = {
    'corenlp': 'CoreNLP',
    'stanza': 'Stanza',
    'biaffine_roberta': 'Biaffine',
    'stackpointer': 'StackPointer',
    'towerparse': 'TowerParse',
    'crf2o_lstm': 'CRF2O',
}
lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]
plot_dir = "../../plots/analysis/trend/"
table_dir = "../../tables/analysis/"


with open("../../tables/analysis/trends_to_plot.json", 'r') as f:
    trends = json.load(f)
print(trends)

df = []
for data in ['hansard', 'deuparl']:
    trend_dir = f"../../data/{data}_final/parsed_v4_balanced_450_3/" # which would be the folder contains the parsing outputs from all parsers
    
    for parser in parsers:
        file = trend_dir + parser + '/measured.csv'
        if not os.path.exists(file):
            continue
        tmp = pd.read_csv(file)
        #tmp['parser'] = [p2name[parser if 'crf2o' not in parser else 'crf2o']] * len(tmp)
        tmp['parser'] = [p2name[parser]] * len(tmp)
        tmp['corpus'] = ['Hansard (en)' if data == 'hansard' else 'DeuParl (de)'] * len(tmp)
        df.append(tmp)

df = pd.concat(df, ignore_index=True)
df.rename(columns={'NDD': 'nDD', 'MDD': "mDD"}, inplace=True)
df['decade'] = [int(i.split("-")[0]) for i in df['id']]


def find_len_group(length, lengths):
        for i in range(len(lengths)):
            if i == len(lengths) - 1 and length >= lengths[-1]:
                return lengths[-1]
            if lengths[i] <= length < lengths[i + 1]:
                return lengths[i]
        return None

def find_decade_group(decade):
    if decade == 2020:
        return 2000
    elif decade % 20 == 0:
        return decade
    else:
        return decade - 10

df['len_group'] = [find_len_group(l, lengths) for l in df['len']]
df['decade_group'] = [find_decade_group(d) for d in df['decade']]
print(df)

decades = sorted(set(df['decade_group']))
for trend, group in trends.items():
    for g in group:
        print(g)
        metric, length = g.split(':')
        tmp = df[df.len_group == int(length)]
        ax = sns.lineplot(data=tmp, x='decade_group', y=metric, errorbar=('ci', 95), hue='corpus',
                          estimator="mean", legend=False)

        ax.set_title(f"Length: {length}", fontsize=40)
        ax.set_xlim((1860, 2000))
        ax.set_ylabel("")
        ax.set_xlabel("")

        plt.xticks(ticks=decades, labels=decades, rotation=30, fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        metric = metric.replace('#', "").replace('$', "").replace("{", '').replace("}", '')
        plt.savefig(os.path.join(plot_dir, f"{trend}-{metric}({length}).pdf"), dpi=300, bbox_inches='tight')
        plt.close()





