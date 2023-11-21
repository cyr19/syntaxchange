import collections

import matplotlib.pyplot as plt
import numpy as np
import json
from argparse import ArgumentParser
import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

p = ArgumentParser()
p.add_argument("--data", '-d', default='deuparl')
p.add_argument("--pos", action='store_true')
p.add_argument("--random", action='store_true')
p.add_argument("--balance", action='store_true')
p.add_argument("--parsers", default="stanza,biaffine,corenlp,stackpointer,towerparse")
args = p.parse_args()
data = args.data

with open(f'measured/{data}/date.json', 'r') as f:
    dates = json.load(f)

with open(f"measured/{data}/id.json", 'r') as f:
    ids = json.load(f)

with open(f"measured/{data}/len.json", 'r') as f:
    lens = json.load(f)

with open(f"../../data/{data}_final/stanza_tokenized_v4/balanced_450_3.json", 'r') as f:
    balanced_ids = json.load(f)['data']

# random
with open(f"../../data/{data}_final/stanza_tokenized_v4/random_2000.json", 'r') as f:
    random_ids = json.load(f)['data']

df = collections.defaultdict(list)

if args.balance:
    valid_ids = [i for i, index in enumerate(ids['stanza']) if index in balanced_ids]
if args.random:
    valid_ids = [i for i, index in enumerate(ids['stanza']) if index in random_ids]

valid_dates = [d for i, d in enumerate(dates['stanza']) if i in valid_ids]
valid_lens = [int(l) for i, l in enumerate(lens['stanza']) if i in valid_ids]




#raise ValueError
#assert len(valid_ids) == len(valid_dates) == len(balanced_ids)

df['decade'] = [int(d[:3]+"0") if int(d[:4]) < 2020 else 2000 for d in valid_dates]
df['decade_group'] = [d if d % 20 == 0 else d-10 for d in df['decade']]
df['len'] = valid_lens

print(np.min(valid_lens))
print(np.max(valid_lens))
print(np.mean(valid_lens))
print(np.median(valid_lens))

print(len([l for l in valid_lens if l<=100])/len(valid_lens))

df = pd.DataFrame(df)
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['axes.facecolor'] = '#cc00ff'
#plt.rcParams['font.sans-serif'] = ['STKAITI']

#fig = plt.figure()

#axes3d = Axes3D(fig)
#plt.figure()
#sns.displot(data=df, x='len', hue='decade_group', kind='ecdf')
#sns.displot(df, x="len", hue='decade_group', kind="kde")
sns.displot(df, x="len", hue='decade_group', element='step', binwidth=3)
plt.xlim(0, 101)
#plt.legend()
plt.tight_layout()

if args.random:
    suffix = 'random'
if args.balance:
    suffix = 'balance'

plt.savefig(f'plots/{data}/len_dis_{suffix}.png', dpi=300)
plt.show()
