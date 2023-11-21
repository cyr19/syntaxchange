import collections
import json
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import seaborn as sns



def find_group(decade):
    if decade < 1860:
        return 1800
    elif decade < 1920:
        return 1860
    elif decade < 1970:
        return 1920
    else:
        return 1980

if __name__ == '__main__':

    data_dir = "../../data/hansard_final/stanza_tokenized_v4/"

    files = glob(data_dir + "*.csv")
    # print(files)

    df = []

    # postags = set()

    with open(data_dir + f'postags_17.json', 'r') as f:
        postags = json.load(f)

    for file in sorted(files):
        tmp = pd.read_csv(file, sep='\t')
        #tmp['decade'] = [int(d[:3] + "0") if int(d[:4]) < 2020 else 2000 for d in tmp['date']]
        tmp['decade'] = [int(d[:3] + "0") for d in tmp['date']]
        #tmp['decade_group'] = [d if d % 20 == 0 else d - 10 for d in tmp['decade']]
        tmp['decade_group'] = [find_group(d) for d in tmp['decade']]
        #tmp['decade_group'] = [1800 if d < 1920 else 1920 for d in tmp['decade']]
        tmp_tags = []

        for i, tags in enumerate(tmp['pos']):
            tags = tags.split()
            tags_vector = ['0'] * len(postags)
            #print(tags)
            c = collections.Counter(tags)
            for tag, count in c.items():
                tags_vector[postags.index(tag)] = str(count)
            #print(tags_vector)
            #raise ValueError
            tmp_tags.append('-'.join(tags_vector[1:]))
            #print(tmp_tags)
            #raise ValueError

        tmp['pos'] = tmp_tags
        '''
        for tags in tmp['pos']:
            tags = tags.split()
            postags.update(tags)
        '''
        tmp = tmp[['decade_group', 'decade', 'date', 'pos', 'index']]
        df.append(tmp)
        #print(file)
        #print(collections.Counter(tmp['pos']).most_common(10))

    #with open(data_dir+f'postags_{len(postags)}.json', 'w') as f:
    #    json.dump(list(postags), f, indent=2)
    #raise ValueError

    df = pd.concat(df, ignore_index=True).sort_values('date', ascending=True)
    #print(collections.Counter(df['pos']).most_common(10))
    c = Counter(df['pos'])#.most_common(30)
    #n = 200
    #most_common = len(c)
    most_common = 500
    #for decade, group in df.groupby('decade_group'):

    kept_pos = {}
    kept_data = []
    for tags, count in tqdm(c.most_common(most_common)):
        kept_tmp = []
        #if count < len(set(df['decade_group'])) * n:
        #    continue
        max_num = np.min([len(group[group.pos == tags]) for _, group in df.groupby('decade_group')])
        length = sum([int(s) for s in tags.split('-')])
        for decade, group in df.groupby('decade_group'):
            tmp = group[group.pos == tags]

            #print(tags)
            #print(length)
            '''
            if len(tmp) < n:
                print(f"{decade} - {tags} has {len(tmp)} sents.")
                break
            kept_tmp.append(tmp.sample(n=n, random_state=42))
            #if decade == 2000:
            if decade == 1950:
                tmp = pd.concat(kept_tmp, ignore_index=True)
                #tmp = tmp.sample(n=n, random_state=42)
                kept_pos[tags] = length
                kept_data.append([i['date'].split("-")[0][:3]+"0-"+str(i['index']) for _, i in tmp.iterrows()])
            '''
            tmp = tmp.sample(n=max_num, random_state=max_num)
            kept_tmp += [i['date'].split("-")[0][:3] + "0-" + str(i['index']) for _, i in tmp.iterrows()]

        kept_pos[tags] = f"len_{length}-num_{max_num}"
        kept_data.append(kept_tmp)

        #else:
        #    continue

    print(len(kept_pos))
    kept = {
        'postags': kept_pos,
        'data': kept_data
    }
    #with open(data_dir+f"same_pos_{n}_{most_common}.json", 'w') as f:
    with open(data_dir+f"same_pos_{most_common}_{len(set(df['decade_group']))}groups.json", 'w') as f:
        json.dump(kept, f, indent=4)

