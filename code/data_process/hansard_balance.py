import collections

import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns

data_dir = "../../data/hansard_final/stanza_tokenized_v4/"

files = glob(data_dir+"*.csv")
print(files)

df = []
for file in files:
    tmp = pd.read_csv(file, sep='\t')
    tmp['decade'] = [int(d[:3]+"0") if int(d[:4]) < 2020 else 2000 for d in tmp['date']]
    tmp['decade_group'] = [d if d % 20 == 0 else d-10 for d in tmp['decade']]
    tmp = tmp[['date', 'len_wo_punct', 'decade', 'decade_group', 'len', 'index']]
    df.append(tmp)


df = pd.concat(df, ignore_index=True).sort_values("date", ascending=True)
df = df[df.len<=150]
#print(df)
#plt.figure()
#sns.histplot(data=df, x='len', binwidth=5)
#sns.histplot(data=df, x='decade', hue='length')
#plt.show()
#raise ValueError

#lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]
#lengths = [5, 10, 15, 20, 30, 40, 50, 60]
'''
plt.figure()
for decade, group in df.groupby('decade'):
    if int(decade) > 1960:
        continue
    nums = [len(group[(group.length < l + 5) & (group.length >= l)]) for l in lengths]
    plt.plot(lengths, nums, label=decade)

plt.xticks(ticks=lengths, labels=lengths, rotation=45)
plt.legend()
plt.savefig("analysis/deuparl_before_1960.png", dpi=200)
plt.show()


plt.figure()
for decade, group in df.groupby('decade'):
    if int(decade) < 1960:
        continue
    nums = [len(group[(group.length < l + 5) & (group.length >= l)]) for l in lengths]
    plt.plot(lengths, nums, label=decade)

plt.xticks(ticks=lengths, labels=lengths, rotation=45)
plt.legend()
plt.savefig("analysis/deuparl_after_1960.png", dpi=200)
plt.show()
'''

#balanced_data = collections.defaultdict(list)
balanced_data = []
lens = collections.defaultdict(list)

# 2000 for each length group
n = 450

for decade, group in df.groupby('decade_group'):
#for decade, group in df.groupby('decade'):
    #print(decade)
    for l in lengths:
        # check distribution

        tmp = group[(group.len < l + 3) & (group.len >= l)]
        num = len(tmp)

        if num < n:
            print(f"{decade}-{l} has {num} sents.")

        try:
            tmp = group[(group.len < l + 3) & (group.len >= l)].sample(n=n, random_state=42)
        except:
                tmp = group[(group.len_wo_punct < l + 3) & (group.len_wo_punct >= l)]

        balanced_data += [i['date'].split("-")[0][:3]+"0-"+str(i['index']) for _, i in tmp.iterrows()]
        lens[decade] += list(tmp['len'])
        #balanced_data[decade] += [i['date'].split("-")[0][:3]+"0-"+str(i['index']) for _, i in tmp.iterrows()]

import numpy as np
lens = {k: np.mean(v) for k, v in lens.items()}
print(lens)

print(len(balanced_data))
balanced_data = {
    "lens": lens,
    "data": balanced_data
}

import json
with open(data_dir+"balanced_450_3.json", 'w') as f:
    json.dump(balanced_data, f, indent=4)








