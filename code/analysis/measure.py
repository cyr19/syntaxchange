import collections
from argparse import ArgumentParser
from glob import glob
from parsing_tree import Tree
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
from collections import defaultdict
import os
import pymannkendall as mk

def measure(sents, ori_data):
    #with open(path, 'r') as f:
    #    sents = f.read(f).strip().split("\n\n")
    print(f'===={len(sents)} sents are being measured...====')
    results = defaultdict(list)
    #for sent in sents:
    skipped = []
    #for i, sent in tqdm(enumerate(sents), total=len(sents), desc=f"{parser}-{decade}"):
    for i, sent in tqdm(enumerate(sents), total=len(sents)):
        if len(ori_data)>0:
            info = ori_data.iloc[i]
        tree = Tree(sent.strip())
        built = tree.build_tree()

        if built:
            if len(ori_data)>0:
                results['id'].append(f"{decade}-{info['index']}")
                assert len(tree.nodes) == info.len, print(f"\n{sent}\n\n{info.sent}")
                results['date'].append(info.date)
                results['len'].append(info.len)
            
            results['MDD'].append(float(tree.get_mdd()))
            results['NDD'].append(float(tree.get_ndd()))

            results['Height'].append(float(tree.get_tree_height_2()))
            results["Ratio$_{head-final}$"].append(float(tree.get_left_child_ratio()))
            results['treeDegree'].append(int(tree.get_k_ary()))
            results['#Leaves'].append(int(tree.get_num_leaves()))

            var, mean = tree.get_degree_variance()
            results['degreeVar'].append(float(var))
            results['degreeMean'].append(float(mean))

            var, mean = tree.get_depth_variance()
            results['depthVar'].append(float(var))
            results['depthMean'].append(float(mean))

            results["d$_{head-final}$"].append(int(tree.get_topo_distance()))
            #results['id_topo'][parser].append(f"{decade}-{info['index']}")

            results['#Crossings'].append(int(tree.get_num_crossing_edges()))
            results["Height$_{dependency}$"].append(int(tree.get_longest_path()))
            results["d$_{root}$"].append(int(tree.root))

            #if info['len'] < 80:
            #results['id_tree'][parser].append(f"{decade}-{info['index']}")
            results["d$_{randomTree}$"].append(tree.tree_edit_distance())
        else:
            #print(f'Sent {i} is not a tree.')
            #skipped += 1
            skipped.append(i)

    print(f"{len(skipped)} sents are not a tree.")
    results = pd.DataFrame(results)
    # skipped related things were added for corr_check so the  results for analysis sdon't have this output
    return results, skipped

def mannkendall(df):
    def find_len_group(length, lengths):
        for i in range(len(lengths)):
            if i == len(lengths) - 1 and length >= lengths[-1]:
                return lengths[-1]
            if lengths[i] <= length < lengths[i + 1]:
                return lengths[i]
        return None
    lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]
    metrics = df.columns[3:]
    df['len_group'] = [find_len_group(l, lengths) for l in df['len']]
    df = df.groupby('len_group')
    results = defaultdict(list)
    for length, group in df:
        for metric in metrics:
            ori_test = mk.original_test(list(group[metric]))#[0]
            #results['parser'].append()
            results['len_group'].append(length)
            results['metric'].append(metric)
            results['MannKendall'].append(ori_test[0])
            results['p'].append(ori_test[2])
            results['slope'].append(ori_test[-2])
    return results
    
        

if __name__ == '__main__':
    from glob import glob
    
    ap = ArgumentParser()
    ap.add_argument('-d', '--data', type=str)
    args = ap.parse_args()
    data = args.data
    #for data in ['deuparl', 'hansard']:
    #for parser in ['towerparse', 'corenlp', 'stanza', 'biaffine', 'stackpointer', 'crf2o_de_merged_proj', 'crf2o_en_merged_proj']:
    for parser in ['biaffine_de_merged_biaffine_lstm']:
        #if data == 'deuparl':
        #    continue
        #if data == 'deuparl' and parser != 'crf2o_de_merged_proj':
        #    continue
        #if data == 'hansard' and parser != 'crf2o_en_merged_proj':
        #    continue
        data_dir = f'../../data/{data}_final/parsed_v4_balanced_450_3/{parser}/'
        files = glob(data_dir + '*.conllu')
        #print(files)
        results = []

        id_json = f"../../data/{data}_final/stanza_tokenized_v4/balanced_450_3.json"
        with open(id_json, 'r') as f:
            ids = json.load(f)['data']

        for file in sorted(files):
            print(file)
            decade = int(file.split('/')[-1][:4])
            ori_path = glob(f"../../data/{data}_final/stanza_tokenized_v4/{decade}*.csv")[0]
            ori_data = pd.read_csv(ori_path, sep='\t')

            with open(file, 'r') as f:
                sents = f.read().strip().split("\n\n")
            
            decade_ids = [int(i.split('-')[-1]) for i in ids if i.split('-')[0]==str(decade)]

            if len(decade_ids) != len(sents):
                try:
                    with open(file.replace(".conllu", '_discarded.json'), 'r') as f:
                        discarded = json.load(f)

                    discarded = [int(d) for d in discarded]
                except:
                    discarded = []
                try:
                    assert len(discarded) + len(sents) == len(decade_ids), f"{decade}: {len(decade_ids)} - {len(sents)} - {len(discarded)}"
                    # re-index and remove the discarded sents
                    decade_ids = [i for i in decade_ids if i not in discarded]
                    
                    assert len(decade_ids) == len(sents)
                except:
                    print(f"{decade}: {len(decade_ids)} - {len(sents)} - {len(discarded)} - {len(ori_data)}")
                    raise ValueError('Indices not aligned.')
            
            ori_data = ori_data.iloc[decade_ids]
            assert len(sents) == len(ori_data)

            #decade_results = measure(sents[:10], ori_data[:10])
            decade_results, _ = measure(sents, ori_data)
            results.append(decade_results)
            #break
        results = pd.concat(results, ignore_index=True)
        results['date'] = pd.to_datetime(results['date'])
        results.sort_values('date', ascending=True, inplace=True)
        

        results.to_csv(data_dir + 'measured.csv', index=False)
        print(len(results))

        trends = pd.DataFrame(mannkendall(results))
        assert len(trends) == 135, print(len(trends))
        trends.to_csv(data_dir + 'trends.csv', index=False)