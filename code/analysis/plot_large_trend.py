import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = 'deuparl'
dataset = 'balanced'
#dataset = '
df = pd.read_csv(f'tables/{data}/stanza_{dataset}.csv')
df['parser'] = ['stanza'] * len(df)
df2 = pd.read_csv(f'tables/{data}/corenlp_{dataset}.csv')
df2['parser'] = ['corenlp'] * len(df2)
df3 = pd.read_csv(f'tables/{data}/biaffine_{dataset}.csv')
df3['parser'] = ['biaffine'] * len(df3)
df = pd.concat([df, df2, df3], ignore_index=True)
df['year'] = [int(d[:4]) for d in df['date']]

df = df[df.len_group==20]
#df = df.groupby('year')
print(df)

metrics = ['len', 'ndd', 'mdd', 'height', 'left_child_ratio', 'k_ary', 'num_leaves', 'degree_var',
                           'degree_mean',
                           'topo_edit_distance', 'tree_edit_distance', 'root_distance', 'longest_path', 'num_crossing',
                           'depth_var', 'depth_mean']

m2m = {
    'len': 'len',
    'ndd': "NDD",
    'mdd': 'MDD',
    'height': 'Tree Height',
    'left_child_ratio': 'Left Children Ratio',
    'k_ary': 'Tree Degree',
    'num_leaves': '#Leaves',
    'degree_var': "Degree Variance",
    'degree_mean': 'Degree Mean',
    'topo_edit_distance': 'Topological Sorting Distance',
    'tree_edit_distance': 'Random Tree Distance',
    'root_distance': "Root Distance",
    "longest_path": 'Longest Path',
    "num_crossing": '#Crossings',
    "depth_var": "Depth Variance",
    'depth_mean': 'Depth Mean'
}

df = df.groupby(['year', 'parser'], as_index=False).mean()
print(df)
#sns.plot()

for metric in metrics:
    #sns.lineplot(data=df, x='decade_group', y=metric, estimator='mean', hue='parser')
    # lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
    #sns.lmplot(data=df, x='decade_group', y=metric, hue='parser')
    sns.lmplot(data=df, x='year', y=metric, hue='parser', scatter_kws={"s": 5},aspect=.8)
    plt.ylabel(m2m[metric])

    plt.savefig(f'plot/regression/{data}_{dataset}_{metric}_20.png', dpi=200)
    plt.show()
