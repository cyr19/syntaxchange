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

parser = ArgumentParser()
parser.add_argument("--parsers", '-p', type=str, default=None)
parser.add_argument("--data", '-d', type=str)
#parser.add_argument("--balanced", action='store_true')
#parser.add_argument("--same_pos", action='store_true')
parser.add_argument("--sanity_check", action='store_true')
parser.add_argument("--example", action='store_true')
parser.add_argument("--chatgpt", action='store_true')
parser.add_argument("--corr", action='store_true')

args = parser.parse_args()
parsers = args.parsers.split(',') if args.parsers is not None else ['stanza', 'corenlp', 'stackpointer', 'biaffine', 'towerparse']

if args.example:
    file = "/home/ychen/projects/syntactic_change/data/ud_treebanks/ud-treebanks-v2.12/UD_German-GSD/de_gsd-ud-train.conllu"
    with open(file, 'r') as f:
        sents = f.read().strip().split("\n\n")
    for sent in sents:
        if len(sent.split('\n')) <= 10:
            tree = Tree(sent.strip())
            built = tree.build_tree()
            if built:
                if len(tree.get_crossing_edges()) > 0:
                    #print(info['sent'])
                    #print(info['date'])
                    print(" ".join([n.text for n in tree.nodes]))
                    print(tree.get_mdd())
                    print(tree.get_ndd())
                    print(tree.root)
                    print(tree.get_tree_height_2())
                    print(tree.get_left_child_ratio())
                    print(tree.get_k_ary())
                    #print(tree.get_nnum_crossing_edges())
                    print(tree.get_num_crossing_edges())
                    print(tree.get_num_leaves())
                    print(tree.topology_sort())
                    print(tree.get_topo_distance())

                    print(tree.tree_edit_distance())
                    print(tree.random_tree)
                    print(tree.get_longest_path())
                    print(tree.get_degrees())
                    print(tree.get_degree_variance())
                    print(tree.depths)
                    print(tree.get_depth_variance())
                    raise ValueError
    raise ValueError

elif not args.chatgpt and not args.corr:
    with open(f"../../data/{args.data}_final/stanza_tokenized_v4/all_ids.json", 'r') as f:
        valid_ids = json.load(f)

    all_discarded = 0

    results = defaultdict(lambda : defaultdict(list))

    for parser in parsers:
        #print()
        #print(parser)
        files = glob(f"../../data/{args.data}_final/parsed_v4/{args.data}_parsed_v4/{parser}/*.conllu")
        for file in sorted(files):
            decade = int(file.split('/')[-1][:4])
            #if decade < 1920:
            #    continue
            ori_path = glob(f"../../data/{args.data}_final/stanza_tokenized_v4/{decade}*.csv")[0]
            ori_data = pd.read_csv(ori_path, sep='\t')
            #print(decade)

            with open(file, 'r') as f:
                sents = f.read().strip().split("\n\n")

            #print(ori_data)
            if len(ori_data) != len(sents):
                try:
                    with open(file.replace(".conllu", '_discarded.json'), 'r') as f:
                        discarded = json.load(f)

                    discarded = [int(d) for d in discarded]
                except:
                    discarded = []

               # print(discarded)
                try:
                    assert len(discarded) + len(sents) == len(ori_data), f"{decade}: {len(ori_data)} - {len(sents)} - {len(discarded)}"
                    # re-index and remove the discarded sents
                    ori_data = ori_data.drop(index=discarded)
                    assert len(ori_data) == len(sents)
                except:
                    print(f"{decade}: {len(ori_data)} - {len(sents)} - {len(discarded)}")
                    pass

            valid = valid_ids[str(decade)]
            #print(len(valid))
            for i, sent in tqdm(enumerate(sents), total=len(sents), desc=f"{parser}-{decade}-{len(valid)}"):
                #if len(results['id'][parser]) > 3:
                #    break
                info = ori_data.iloc[i]
                if info['index'] not in valid:
                    continue

                tree = Tree(sent.strip())
                built = tree.build_tree()
                if built:

                    '''
                    results['id'][parser].append(f"{decade}-{info['index']}")
                    #results['decade'][parser].append(decade)
                    results['date'][parser].append(info.date)
                    results['len_wo_punct'][parser].append(int(info.len_wo_punct))
                    results['len'][parser].append(int(info.len))
                    
                    results['mdd'][parser].append(float(tree.get_mdd()))
                    results['ndd'][parser].append(float(tree.get_ndd()))
                    
                    results['height'][parser].append(float(tree.get_tree_height_2()))
                    results['left_child_ratio'][parser].append(float(tree.get_left_child_ratio()))
                    results['k_ary'][parser].append(int(tree.get_k_ary()))
                    results['num_leaves'][parser].append(int(tree.get_num_leaves()))
                    #results['topo_steps'].append(tree.get_sort_steps())
                    results['n_num_crossing'][parser].append(float(tree.get_nnum_crossing_edges()))
                    #results['num_crossing'][parser].append(float(np.mean(tree.get_crossing_edges())))
                    
                    var, mean = tree.get_degree_variance()
                    results['degree_var'][parser].append(float(var))
                    results['degree_mean'][parser].append(float(mean))
    
                    var, mean = tree.get_depth_variance()
                    results['depth_var'][parser].append(float(var))
                    results['depth_mean'][parser].append(float(mean))
    
    
    
                    # results['decade'][parser].append(decade)
                    #try:
                    results['topo_edit_distance'][parser].append(int(tree.get_topo_distance()))
                    results['id_topo'][parser].append(f"{decade}-{info['index']}")
                    
                    results['num_crossing'][parser].append(int(tree.get_num_crossing_edges()))
                    results['longest_path'][parser].append(int(tree.get_longest_path()))
                    #results['tree_edit_distance'][parser].append(tree.tree_edit_distance())
                    results['root_distance'][parser].append(int(tree.root))
                    
                    
                    if info['len'] < 80:
                        results['id_tree'][parser].append(f"{decade}-{info['index']}")
                        results['tree_edit_distance'][parser].append(tree.tree_edit_distance())
                    
                    results['mdd'][parser].append(float(tree.get_mdd()))
                    results['ndd'][parser].append(float(tree.get_ndd()))
                    '''
                    results['len_true'][parser].append(len(tree.nodes))

        for k, v in results.items():
            path = f'measured/{args.data}/{k}.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    stored = json.load(f)
                stored.update(v)
            else:
                stored = v
            with open(path, 'w') as f:
                json.dump(stored, f, indent=4)

elif args.chatgpt:

    for parser in parsers:
        results = defaultdict(lambda: defaultdict(list))
        files = glob(f'../../data/{args.data}_final/parsed_v4/chatgpt/*_sent_{parser}.conllu')

        for file in sorted(files):
            decade = int(file.split('/')[-1][:4])

            tmp = decade if decade < 2020 else 2000
            decade_group = tmp if tmp % 20 == 0 else tmp - 10

            with open(file, 'r') as f:
                sents = f.read().strip().split("\n\n")

            for i, sent in tqdm(enumerate(sents), total=len(sents), desc=f"{parser}-{decade}"):
                tree = Tree(sent.strip())
                built = tree.build_tree()
                if built:
                    results[parser]['decade'].append(decade)
                    results[parser]['decade_group'].append(decade_group)
                    results[parser]['len'].append(int(len(tree.nodes)))
                    results[parser]['mdd'].append(float(tree.get_mdd()))
                    results[parser]['ndd'].append(float(tree.get_ndd()))

                    results[parser]['height'].append(float(tree.get_tree_height_2()))
                    results[parser]['left_child_ratio'].append(float(tree.get_left_child_ratio()))
                    results[parser]['k_ary'].append(int(tree.get_k_ary()))
                    results[parser]['num_leaves'].append(int(tree.get_num_leaves()))
                    results[parser]['n_num_crossing'].append(float(tree.get_nnum_crossing_edges()))

                    var, mean = tree.get_degree_variance()
                    results[parser]['degree_var'].append(float(var))
                    results[parser]['degree_mean'].append(float(mean))

                    var, mean = tree.get_depth_variance()
                    results[parser]['depth_var'].append(float(var))
                    results[parser]['depth_mean'].append(float(mean))

                    results[parser]['topo_edit_distance'].append(int(tree.get_topo_distance()))

                    results[parser]['num_crossing'].append(int(tree.get_num_crossing_edges()))
                    results[parser]['longest_path'].append(int(tree.get_longest_path()))
                    results[parser]['root_distance'].append(int(tree.root))

                    if int(len(tree.nodes)) < 80:
                        results[parser]['tree_edit_distance'].append(tree.tree_edit_distance())
                    else:
                        results[parser]['tree_edit_distance'].append(None)

        tmp = pd.DataFrame(results[parser])
        tmp.to_csv(f"tables/chatgpt/{args.data}_{parser}.csv", index=False)

elif args.corr:

    results = defaultdict(lambda: defaultdict(list))
    for parser in parsers:

        # /home/ychen/projects/syntactic_change/code/parsers/parsing_outputs/gpt-3.5-turbo-0613_correction_3_edit_chatgpt_correction_biaffine.conllu
        # original
        files = glob(f'../parsers/parsing_outputs/gpt-3.5-turbo-0613_correction_*_edit_text_{parser}.conllu')
        #print(files)
        ori_sents = []
        for file in files:
            with open(file, 'r') as f:
                ori_sents += f.read().strip().split("\n\n")

        files = glob(f'../parsers/parsing_outputs/gpt-3.5-turbo-0613_correction_*_edit_correction_{parser}.conllu')
        #print(files)
        cor_sents = []
        for file in files:
            with open(file, 'r') as f:
                cor_sents += f.read().strip().split("\n\n")

        files = glob(f'../parsers/parsing_outputs/gpt-3.5-turbo-0613_correction_*_edit_chatgpt_correction_{parser}.conllu')
        #print(files)
        chatcor_sents = []
        for file in files:
            with open(file, 'r') as f:
                chatcor_sents += f.read().strip().split("\n\n")

        assert len(ori_sents) == len(chatcor_sents) == len(cor_sents)

        for i, sents in enumerate([ori_sents, cor_sents, chatcor_sents]):
            for j, sent in tqdm(enumerate(sents), total=len(sents)):
                tree = Tree(sent.strip())
                built = tree.build_tree()
                results[parser]['text_group'].append(i)
                results[parser]['tid'].append(j)
                if built:
                    #results[parser]['len'].append(int(len(tree.nodes)))
                    results[parser]['mdd'].append(float(tree.get_mdd()))
                    results[parser]['ndd'].append(float(tree.get_ndd()))

                    results[parser]['height'].append(float(tree.get_tree_height_2()))
                    results[parser]['left_child_ratio'].append(float(tree.get_left_child_ratio()))
                    results[parser]['k_ary'].append(int(tree.get_k_ary()))
                    results[parser]['num_leaves'].append(int(tree.get_num_leaves()))
                    #results[parser]['n_num_crossing'].append(float(tree.get_nnum_crossing_edges()))

                    var, mean = tree.get_degree_variance()
                    results[parser]['degree_var'].append(float(var))
                    results[parser]['degree_mean'].append(float(mean))

                    var, mean = tree.get_depth_variance()
                    results[parser]['depth_var'].append(float(var))
                    results[parser]['depth_mean'].append(float(mean))

                    results[parser]['topo_edit_distance'].append(int(tree.get_topo_distance()))

                    results[parser]['num_crossing'].append(int(tree.get_num_crossing_edges()))
                    results[parser]['longest_path'].append(int(tree.get_longest_path()))
                    results[parser]['root_distance'].append(int(tree.root))
                    results[parser]['tree_edit_distance'].append(tree.tree_edit_distance())
                else:
                    #results[parser]['text_group'].append(i)
                    # results[parser]['len'].append(int(len(tree.nodes)))
                    results[parser]['mdd'].append(None)
                    results[parser]['ndd'].append(None)

                    results[parser]['height'].append(None)
                    results[parser]['left_child_ratio'].append(None)
                    results[parser]['k_ary'].append(None)
                    results[parser]['num_leaves'].append(None)
                    # results[parser]['n_num_crossing'].append(float(tree.get_nnum_crossing_edges()))

                    #var, mean = tree.get_degree_variance()
                    results[parser]['degree_var'].append(None)
                    results[parser]['degree_mean'].append(None)

                    #var, mean = tree.get_depth_variance()
                    results[parser]['depth_var'].append(None)
                    results[parser]['depth_mean'].append(None)

                    results[parser]['topo_edit_distance'].append(None)

                    results[parser]['num_crossing'].append(None)
                    results[parser]['longest_path'].append(None)
                    results[parser]['root_distance'].append(None)
                    results[parser]['tree_edit_distance'].append(None)
            #break
    with open('tables/correction_correlation_measured.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open('tables/correction_correlation_measured.json', 'r') as f:
        results = json.load(f)
    from scipy.stats import pearsonr, spearmanr
    correlations = defaultdict(list)
    for parser in parsers:
        print(parser)
        tmp = pd.DataFrame(results[parser])
        tmp.to_csv(f"tables/{parser}.csv", index=False)
        #tmp.dropna()
        #for i in range(3):
        print(len(tmp))
        print(tmp)
        tmp.fillna(-1, inplace=True)
        #print(tmp.columns.values[0])
        #ori_measures = tmp[tmp[tmp.columns.values[0]]==0]
        ori_measures = tmp[tmp.text_group==0]
        cor_measures = tmp[tmp.text_group==1]
        chatcor_measures = tmp[tmp.text_group==2]

        assert len(ori_measures) == len(cor_measures) == len(chatcor_measures)
        correlations['parser'].append(parser)
        correlations['vs.'].append('ori vs. human corrected')
        # ori vs. human corr
        for measure in tmp.columns[2:]:
            #print(measure)
            #pc = pearsonr(ori_measures[measure], cor_measures[measure])[0]
            sc = spearmanr(ori_measures[measure], cor_measures[measure])[0]
            correlations[measure].append(sc)

        correlations['parser'].append(parser)
        correlations['vs.'].append('ori vs. chatgpt_corrected')
        # ori vs. chatgpt corr
        for measure in tmp.columns[2:]:
            # print(measure)
            # pc = pearsonr(ori_measures[measure], cor_measures[measure])[0]
            sc = spearmanr(ori_measures[measure], chatcor_measures[measure])[0]
            correlations[measure].append(sc)

        # human corr
        correlations['parser'].append(parser)
        correlations['vs.'].append('human_corrected vs. chatgpt_corrected')
        # ori vs. human corr
        for measure in tmp.columns[2:]:
            # print(measure)
            # pc = pearsonr(ori_measures[measure], cor_measures[measure])[0]
            sc = spearmanr(cor_measures[measure], chatcor_measures[measure])[0]
            correlations[measure].append(sc)

    correlations = pd.DataFrame(correlations)
    correlations.to_csv("tables/correction_correlations.csv", index=False)

