from argparse import ArgumentParser
from glob import glob
from parsing_tree import Tree
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument("--data", '-d', default='hansard')
args = parser.parse_args()
corpus = args.data

valid_ids = set()
with open(f"../../data/{corpus}_final/stanza_tokenized_v4/balanced_450_3.json", 'r') as f:
    valid_balance = json.load(f)['data']
valid_ids.update(valid_balance)
print(len(valid_ids))
with open(f"../../data/{corpus}_final/stanza_tokenized_v4/same_pos_500_4groups.json", 'r') as f:
    data = json.load(f) #['data']
    postags = data['postags']
    data = data['data']
    valid_pos = []
    total_length = []
    kept_postags = []
    id2pos = {}
    for i, (tags, info) in enumerate(postags.items()):
        info = info.split('-')
        length = int(info[0].split('_')[-1])
        count = int(info[-1].split('_')[-1])
        #if count >= 100:
        #print('???')
        if corpus == 'hansard':
            #print('???')
            if length >= 10 and count >= 30:
                valid_pos += data[i]
                for index in data[i]:
                    id2pos[index] = tags
                total_length += [length] * len(data[i])
                kept_postags.append(tags)
            if 5 <= length < 10 and count >= 200:
                valid_pos += data[i]
                for index in data[i]:
                    id2pos[index] = tags
                total_length += [length] * len(data[i])
                kept_postags.append(tags)
            '''
            if length < 5 and count > 600:
                valid += data[i]
                total_length += [length] * len(data[i])
                kept_postags.append(tags)
            '''

        else:
            if length >= 10 and count >= 30:
                valid_pos += data[i]
                for index in data[i]:
                    id2pos[index] = tags
                total_length += [length] * len(data[i])
                kept_postags.append(tags)

            if 5 <= length < 10 and count >= 1000:
                valid_pos += data[i]
                for index in data[i]:
                    id2pos[index] = tags
                total_length += [length] * len(data[i])
                kept_postags.append(tags)

    '''
    print(np.mean(total_length))
    #print(np.max(total_length))
    print(len(kept_postags))
    print(len(total_length))
    print(len(valid_pos))
    #print(valid)
    '''
    print(len(kept_postags))
    raise ValueError
valid_ids.update(valid_pos)
'''
with open(f"../../data/{corpus}_final/stanza_tokenized_v4/postags.json", 'r') as f:
    postags = json.load(f)[1:]

id2pos = {k: '-'.join(postags)}
'''
with open(f"../../data/{corpus}_final/stanza_tokenized_v4/id2pos.json", 'w') as f:
    json.dump(id2pos, f, indent=4)

print(len(valid_ids))

with open(f"../../data/{corpus}_final/stanza_tokenized_v4/random_2000.json", 'r') as f:
    valid_random = json.load(f)['data']

valid_ids.update(valid_random)
print(len(valid_ids))

all_ids = defaultdict(list)
for ids in sorted(list(valid_ids)):
    decade, index = ids.split('-')
    print(decade)
    all_ids[decade].append(int(index))


print(all_ids.keys())
with open(f"../../data/{corpus}_final/stanza_tokenized_v4/all_ids_v2.json", 'w') as f:
    json.dump(all_ids, f, indent=4)