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

p = ArgumentParser()
#p.add_argument("--parsers", default="stanza,biaffine,corenlp,stackpointer,towerparse,crf2o_de_merged_proj,crf2o_en_merged_proj")
p.add_argument("--parsers", default="stanza,biaffine_roberta,corenlp,stackpointer,towerparse,crf2o_lstm")
args = p.parse_args()

parsers = args.parsers.split(',')
plot_dir = "../../plots/analysis/"
table_dir = "../../tables/analysis/"

df = []
for data in ['hansard', 'deuparl']:
    trend_dir = f"../../data/{data}_final/parsed_v4_balanced_450_3/"
    
    for parser in parsers:
        file = trend_dir + parser + '/trends.csv'
        if not os.path.exists(file):
            continue
        tmp = pd.read_csv(file)
        tmp['parser'] = [parser if 'crf2o' not in parser and 'biaffine' not in parser else parser.split('_')[0]] * len(tmp)
        tmp['corpus'] = ['Hansard (en)' if data == 'hansard' else 'DeuParl (de)'] * len(tmp)
        df.append(tmp)

df = pd.concat(df, ignore_index=True)

t2l = {
    'no trend': 0,
    'increasing': 1,
    'decreasing': 2
}
df['MannKendall'] = [t2l[l] for l in df['MannKendall']]
df['metric'] = [m if m not in ['MDD', 'NDD'] else m[0].lower()+m[1:] for m in list(df['metric'])]
assert 135 * 6 * 2 == len(df), print(df)




parsers = ['corenlp', 'stanza', 'biaffine', 'stackpointer', 'towerparse', 'crf2o']

p2name = {
    'corenlp': 'CoreNLP',
    'stanza': 'Stanza',
    'biaffine': 'Biaffine',
    'stackpointer': 'StackPointer',
    'towerparse': 'TowerParse',
    'crf2o': 'CRF2O'
}

m2m = {
    'root_distance': "d$_{root}$",
    'mdd': 'mDD',
    'ndd': "nDD",
    "num_crossing": '#Crossings',
    'num_leaves': '#Leaves',
    'height': 'Height',
    "longest_path": 'Height$_{dependency}$',
    "depth_var": "depthVar",
    'depth_mean': 'depthMean',
    'k_ary': 'treeDegree',
    'degree_var': "degreeVar",
    'degree_mean': 'degreeMean',
    'left_child_ratio': 'Ratio$_{head-final}$',
    'topo_edit_distance': 'd$_{head-final}$',
    'tree_edit_distance': 'd$_{randomTree}$'
}

metrics = [
    "d$_{root}$", 'mDD', "nDD", '#Crossings', '#Leaves', 'Height', 'Height$_{dependency}$',
    "depthVar", 'depthMean', 'treeDegree', "degreeVar", 'degreeMean', 'Ratio$_{head-final}$',
    'd$_{head-final}$', 'd$_{randomTree}$'
]

lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]
trend_tables = []

# majority vote
for corpus in ['DeuParl (de)', 'Hansard (en)']:
    results = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(int)))
    tmp = df[df.corpus==corpus]
    for length, len_group in tmp.groupby('len_group'):
        for parser, group in len_group.groupby('parser'):
            group = group[group.MannKendall!=0]
            #print(group)
            for _, row in group.iterrows():
                if row['MannKendall'] == 1:
                    results[length]['increasing'][row['metric']] += 1
                else:
                    results[length]['decreasing'][row['metric']] += 1

    final = collections.defaultdict(list)
    for length, trend_data in results.items():
        for state, metric_vote in trend_data.items():
            for metric, vote in metric_vote.items():
                threshold = 4 if metric != '#Crossings' else 3
                if vote >= threshold:
                    final['Len'].append(length)
                    final['Metric'].append(metric)
                    final['Trend'].append(state)
                    final['Vote'].append(vote)

    final = pd.DataFrame(final)

    final = final.sort_values(['Len', 'Trend', 'Vote'], ascending=False)
    #print(final)
    final.to_csv(table_dir + f'{corpus}_majority_vote_sep.csv', index=False)

    table = collections.defaultdict(list)
    table['Metric'] = metrics
    for length in lengths:
        measures = ['-'] * len(metrics)
        tmp = final[(final.Len == length)]

        for i, metric in enumerate(metrics):
            tmp_m = tmp[tmp.Metric == metric]
            #print(tmp_m)
            if len(tmp_m) > 0:
                assert len(tmp_m) == 1, tmp_m
                vote = tmp_m['Vote'].values[0]
                measures[i] = "+"+str(int(vote)) if tmp_m['Trend'].values[0] == 'increasing' else '-'+str(int(vote))

        table[length] = measures

    table = pd.DataFrame(table)
    trend_tables.append(table)
    
    table.to_csv(table_dir + f'{corpus}_majority_vote.csv', index=False)





same = 0
diff = 0
no = 0
df1 = trend_tables[0].values
df2 = trend_tables[1].values

trends_len = defaultdict(lambda: defaultdict(int))
trends_metric = defaultdict(lambda: defaultdict(int))
de_no_trend = 0
en_no_trend = 0
both_no_trend = 0

trends_to_plot = defaultdict(list)

for v1, v2 in zip(df1, df2):
    metric = v1[0]
    for i, (vv1, vv2) in enumerate(zip(v1, v2)):
        if i != 0:
            if vv1 == '-' or vv2 == '-':
                trends_len[lengths[i-1]]['Incomparable'] += 1
                trends_metric[metric]['Incomparable'] += 1
                no += 1
                if vv1 == vv2 == '-':
                    both_no_trend += 1
                else:
                    if vv1 == '-':
                        de_no_trend += 1
                    if vv2 == '-':
                        en_no_trend += 1
            elif int(vv1) * int(vv2) > 0:
                trends_len[lengths[i-1]]['Same'] += 1
                trends_metric[metric]['Same'] += 1
                same += 1
                trends_to_plot['same'].append(f"{metric}:{lengths[i-1]}")
                
            elif int(vv1) * int(vv2) < 0:
                trends_len[lengths[i-1]]['Different'] += 1
                trends_metric[metric]['Different'] += 1
                diff += 1
                trends_to_plot['diff'].append(f"{metric}:{lengths[i-1]}")
                #(lengths[i-1])
                #print(metric)
#raise ValueError

with open(table_dir+'trends_to_plot.json', 'w') as f:
    json.dump(trends_to_plot, f, indent=2)
#raise ValueError

labels = ['Same', 'Different', 'Incomparable']
sizes = [same, diff, no]
total = no+same+diff

def my_fmt(x):
    return '{:.1f}%\n({:.0f})'.format(x, total*x/100)

import matplotlib as mpl
mpl.rcParams['font.size'] = 15
plt.pie(sizes, labels=labels, autopct=my_fmt)

plt.tight_layout()
plt.savefig(plot_dir+'trend_dis_pie.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

age_ratios = [en_no_trend/no, de_no_trend/no, both_no_trend/no]
age_labels = ['No trend for English', 'No trend for German',  'No trend for both']
bottom = 1
width = .1

fig, ax2 = plt.subplots()
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    #print('h',height)
    bc = ax2.bar(0, height, width, bottom=bottom, color='C2', label=label,
                 alpha=0.1 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{round(age_ratios[j]*100, 1)}%\n({int(no*age_ratios[j])})"], label_type='center')

ax2.set_title('Incomparable')
ax2.legend(bbox_to_anchor=(1.2,1), fontsize=12, loc='upper right')
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)
plt.tight_layout()
plt.savefig(plot_dir+'trend_incomparable.pdf', dpi=300, bbox_inches='tight')
plt.show()

measure_df = defaultdict(list)
for measure, v in trends_metric.items():
    msame = v['Same']
    mdiff = v['Different']
    minc = v['Incomparable']

    measure_df['Metric'].append(measure)
    measure_df['Trends'].append('Same')
    measure_df['Count'].append(msame)

    measure_df['Metric'].append(measure)
    measure_df['Trends'].append('Different')
    measure_df['Count'].append(mdiff)

    measure_df['Metric'].append(measure)
    measure_df['Trends'].append('Incomparable')
    measure_df['Count'].append(minc)

measure_df = pd.DataFrame(measure_df)
plt.figure(figsize=(10,8))
sns.barplot(data=measure_df, y='Metric', x='Count', hue='Trends', orient='h', width=.8)
plt.grid(visible=True, which='major', axis='both')
plt.xlim(0,9)
plt.xticks(ticks=range(10))
plt.legend().set_visible(False)
plt.tight_layout()
plt.savefig(plot_dir+'/measure_diff.pdf', dpi=300, bbox_inches='tight')
#plt.xticks(rotation=30)
plt.show()
plt.close()
#raise ValueError

len_df = collections.defaultdict(list)
for l, v in trends_len.items():
    #print(v)
    msame = v['Same']
    mdiff = v['Different']
    minc = v['Incomparable']

    len_df['Length'].append(l)
    len_df['Trends'].append('Same')
    len_df['Count'].append(msame)

    len_df['Length'].append(l)
    len_df['Trends'].append('Different')
    len_df['Count'].append(mdiff)

    len_df['Length'].append(l)
    len_df['Trends'].append('Incomparable')
    len_df['Count'].append(minc)

len_df = pd.DataFrame(len_df)
plt.figure(figsize=(10,8))
sns.barplot(data=len_df, y='Length', x='Count', hue='Trends', orient='h', width=.5)
plt.grid(visible=True, which='major', axis='both')
plt.xlim(0,15)
plt.xticks(ticks=range(16))
plt.tight_layout()
plt.savefig(plot_dir+'len_diff.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#raise ValueError


# fleiss
#results = collections.defaultdict(lambda: collections.defaultdict(list))
preds = collections.defaultdict(lambda : collections.defaultdict(list))
r = collections.defaultdict(list)

for l, l_data in df.groupby('corpus'):
    #print(l)
    for index, tmp in l_data.groupby(['metric', 'len_group']):
        measure, length = index
        assert len(tmp) == 6
        preds[l][measure].append(tmp['MannKendall'].values)

assert len(preds) == 2
assert len(preds['Hansard (en)']) == 15
assert len(preds['Hansard (en)']['mDD']) == 9

# fleiss kappa
r = collections.defaultdict(list)
for l, l_preds in preds.items():
    for measure, rate in l_preds.items():
        data = []
        rate = np.array(rate)
        #print(rate)
        dats, cats = aggregate_raters(rate, n_cat=3)
        fleiss = fleiss_kappa(dats, method='fleiss')
        #print(f"{l}-{measure}-{fleiss}")
        r['corpus'].append(l)
        r['Metric'].append(measure)
        r["Fleiss' Kappa"].append(fleiss)

r = pd.DataFrame(r)
plt.figure(figsize=(10,4))
sns.barplot(data=r, x='Metric', y="Fleiss' Kappa", hue='corpus', hue_order=['Hansard (en)', 'DeuParl (de)'])
plt.xticks(rotation=45, fontsize=12)
plt.legend(loc='upper right')
plt.xlabel('')
plt.tight_layout()

plt.savefig(plot_dir+'fleiss_measure.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# cohen kappa
sns.set_style('white')
for corpus in ['Hansard (en)', 'DeuParl (de)']:
    matrix = np.zeros((len(parsers)+1, len(parsers)))

    tmp = df[df.corpus==corpus]
    for i in range(len(parsers)):
        
        p1 = tmp[tmp.parser == parsers[i]]
        for j in range(len(parsers)):
            p2 = tmp[tmp.parser == parsers[j]]
            assert len(p1) == len(p2) == 135
            a = cohen_kappa_score(list(p1['MannKendall']), list(p2['MannKendall']), labels=[0, 1, 2])
            matrix[i][j] = a 
            print(f"{parsers[i]} vs. {parsers[j]}: {a}")
        mask = np.ones(matrix[i].shape, bool)
        mask[i] = False
        matrix[-1][i] = np.mean(matrix[i][mask])


    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    mask = np.tri(matrix.shape[1], matrix.shape[0], k=0).T
    
    
    data = np.ma.array(matrix, mask=mask)
    data = data[1:,:]*100
   
    fig, ax = plt.subplots(figsize=(7,7))
    cax = ax.imshow(data, cmap='Reds')
    fig.colorbar(cax, fraction=0.045)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "%.1f" % data[i][j] if isinstance(data[i][j], float) else data[i][j], va='center', ha='center')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.set_xticks(range(6), list(p2name.values()), rotation=45)
    ax.set_yticks(range(6), list(p2name.values())[1:] + ['AVG'])

    plt.tight_layout()
    plt.show()
    plt.savefig(plot_dir+f'cohen_{corpus}.pdf', dpi=300, bbox_inches='tight')
    plt.close()



