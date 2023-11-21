import collections
import json
from argparse import ArgumentParser
import numpy as np
import pymannkendall as mk
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')


import pandas as pd

p = ArgumentParser()
p.add_argument("--data", '-d', default='deuparl')
p.add_argument("--pos", action='store_true')
p.add_argument("--random", action='store_true')
p.add_argument("--balance", action='store_true')
p.add_argument("--parsers", default="stanza,biaffine,corenlp,stackpointer,towerparse")
args = p.parse_args()
data = args.data

if data == 'deuparl':
    from data_process.deuparl_same_pos import find_group
else:
    from data_process.hansard_same_pos import find_group

#data = 'hansard'
with open(f'measured/{data}/date.json', 'r') as f:
    dates = json.load(f)

with open(f"measured/{data}/id.json", 'r') as f:
    ids = json.load(f)

with open(f"measured/{data}/id_tree.json", 'r') as f:
    tree_ids = json.load(f)

with open(f"measured/{data}/len.json", 'r') as f:
    lens = json.load(f)

# random
with open(f"../../data/{data}_final/stanza_tokenized_v4/random_2000.json", 'r') as f:
    random_ids = json.load(f)['data']

# balanced
with open(f"../../data/{data}_final/stanza_tokenized_v4/balanced_450_3.json", 'r') as f:
    balanced_ids = json.load(f)['data']

measures = ['ndd', 'mdd', 'height', 'left_child_ratio', 'k_ary', 'num_leaves', 'n_num_crossing', 'degree_var', 'degree_mean', 'depth_var', 'depth_mean', 'topo_edit_distance', 'tree_edit_distance', 'root_distance', 'longest_path', 'num_crossing']
lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]

with open(f"../../data/{data}_final/stanza_tokenized_v4/id2pos.json", 'r') as f:
    id2pos = json.load(f)

postags = list(set(id2pos.values()))

with open(f"../../data/{data}_final/stanza_tokenized_v4/same_pos_500_4groups.json", 'r') as f:
    all_pos = json.load(f)

data_ids = [i for i, pos in enumerate(list(all_pos['postags'].keys())) if pos in postags]

pos_ids = []
for i, index in enumerate(list(all_pos['data'])):
    if i in data_ids:
        pos_ids += index


def find_len_group(length, lengths):
    for i in range(len(lengths)):
        if i == len(lengths) - 1 and length >= lengths[-1]:
            return lengths[-1]
        if lengths[i] <= length < lengths[i+1]:
            return lengths[i]
    return None


parsers = args.parsers.split(',')
print(parsers)

for parser in parsers:
    print(parser)
    if args.random:
        print("Processing random dataset..")
        # random
        valid_ids = [i for i, index in enumerate(ids[parser]) if index in random_ids]

        valid_dates = [d for i, d in enumerate(dates[parser]) if i in valid_ids]
        valid_lens = [d for i, d in enumerate(lens[parser]) if i in valid_ids]

        valid_data = collections.defaultdict(list)
        valid_data['id'] = [ids[parser][i] for i in valid_ids]
        valid_data['date'] = valid_dates
        valid_data['len'] = valid_lens
        for measure in measures:
            #if measure != 'tree_edit_distance':
            #    continue
            with open(f"measured/{data}/{measure}.json", 'r') as f:
                metrics = json.load(f)
            #print(measure)
            if measure == 'tree_edit_distance':
                ms = []
                assert len(tree_ids[parser]) == len(metrics[parser]), print(f"{len(tree_ids[parser])}-{len(metrics[parser])}")
                #assert len(metrics[parser]) == len(valid_tree_ids), print(f"{len(metrics[parser])}-{len(valid_tree_ids)}")
                tree_map = {index: distance for index, distance in zip(tree_ids[parser], metrics[parser])}
                assert len(tree_map) == len(tree_ids[parser])
                for i in valid_ids:
                    if ids[parser][i] in tree_ids[parser]:
                        ms.append(tree_map[ids[parser][i]])
                    else:
                        ms.append(None)
                valid_measures = ms
            else:
                assert len(ids[parser]) == len(metrics[parser]), print(f"{len(ids[parser])}-{len(metrics[parser])}")
                valid_measures = [m for i, m in enumerate(metrics[parser]) if i in valid_ids]
            valid_data[measure] = valid_measures

        #print(f"{parser}-random:", len(valid_measures))
        try:
            print(valid_data.keys())
            valid_data = pd.DataFrame(valid_data)
        except:
            print({k:len(v) for k, v in valid_data.items()})
            raise ValueError
        valid_data = valid_data.sort_values('date')

        print(f"{parser} - {len(valid_data)}")
        valid_data.to_csv(f"tables/{data}/{parser}.csv", index=False)

    if args.balance:
        print("Processing balance dataset..")
        # balanced
        valid_ids = [i for i, index in enumerate(ids[parser]) if index in balanced_ids]
        #valid_tree_ids = [i for i, index in enumerate(tree_ids[parser]) if index in balanced_ids]
        #valid_tree_index = [i for i, index in enumerate(tree_ids[parser]) if index in balanced_ids]

        valid_dates = [d for i, d in enumerate(dates[parser]) if i in valid_ids]
        valid_lens = [d for i, d in enumerate(lens[parser]) if i in valid_ids]

        valid_data = collections.defaultdict(list)
        valid_data['id'] = [ids[parser][i] for i in valid_ids]
        valid_data['date'] = valid_dates
        valid_data['len'] = valid_lens

        valid_data['decade'] = [int(d[:3] + "0") if int(d[:4]) < 2020 else 2000 for d in valid_data['date']]
        valid_data['decade_group'] = [d if d % 20 == 0 else d - 10 for d in valid_data['decade']]
        valid_data['len_group'] = [find_len_group(l, lengths) for l in valid_data['len']]

        for measure in measures:
            with open(f"measured/{data}/{measure}.json", 'r') as f:
                metrics = json.load(f)
            print(measure)
            if measure == 'tree_edit_distance':
                ms = []
                assert len(tree_ids[parser]) == len(metrics[parser]), print(f"{len(tree_ids[parser])}-{len(metrics[parser])}")
                # assert len(metrics[parser]) == len(valid_tree_ids), print(f"{len(metrics[parser])}-{len(valid_tree_ids)}")
                tree_map = {index: distance for index, distance in zip(tree_ids[parser], metrics[parser])}
                assert len(tree_map) == len(tree_ids[parser])
                for i in valid_ids:
                    if ids[parser][i] in tree_ids[parser]:
                        ms.append(tree_map[ids[parser][i]])
                    else:
                        ms.append(None)
                valid_measures = ms
            else:
                valid_measures = [m for i, m in enumerate(metrics[parser]) if i in valid_ids]
            valid_data[measure] = valid_measures

        # print(f"{parser}-random:", len(valid_measures))

        valid_data = pd.DataFrame(valid_data)
        valid_data = valid_data.sort_values('date')
        valid_data.drop(columns='decade', inplace=True)
        print(f"{parser} - balanced: {len(valid_data)}")
        valid_data.to_csv(f"tables/{data}/{parser}_balanced.csv", index=False)

    #continue
    if args.pos:
        print("Processing pos dataset..")
        # same_pos
        valid_ids = [i for i, index in enumerate(ids[parser]) if index in pos_ids]
        valid_dates = [d for i, d in enumerate(dates[parser]) if i in valid_ids]
        valid_lens = [d for i, d in enumerate(lens[parser]) if i in valid_ids]

        valid_data = collections.defaultdict(list)
        valid_data['id'] = [ids[parser][i] for i in valid_ids]
        valid_data['date'] = valid_dates
        valid_data['len'] = valid_lens
        valid_data['postags'] = [id2pos[i] for i in valid_data['id']]
        valid_data['decade_group'] = [find_group(int(d[:3]+"0")) for d in valid_data['date']]


        for measure in measures:
            with open(f"measured/{data}/{measure}.json", 'r') as f:
                metrics = json.load(f)
            valid_measures = [m for i, m in enumerate(metrics[parser]) if i in valid_ids]
            valid_data[measure] = valid_measures

        # print(f"{parser}-random:", len(valid_measures))

        valid_data = pd.DataFrame(valid_data)
        valid_data = valid_data.sort_values('date')


        print(f"{parser} - pos: {len(valid_data)}")
        valid_data.to_csv(f"tables/{data}/{parser}_pos.csv", index=False)








