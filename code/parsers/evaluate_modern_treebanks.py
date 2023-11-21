import collections

import numpy as np

from conll18_ud_eval import *
import pandas as pd

from parse_all import *
from argparse import ArgumentParser
from glob import glob
import os
import re

def exists_cycle(heads):
    visited = [False] * len(heads)
    graph = collections.defaultdict(list)
    for i, head in enumerate(heads):
        graph[head].append(i+1)

    def dfs(graph, node, visited, start):
        if visited[node-1]:
            if node == start:
                #print('.............')
                return True

        visited[node-1] = True
        for child in graph[node]:
            dfs(graph, child, visited, start)
        visited[node-1] = False

    detected = 0
    for i in range(1, len(heads)+1):
        detected = dfs(graph, i, visited, i)
        if detected:
            break
    return detected


#def find_cycle(heads):

def read_conllu(path, discard=[]):
    if 'conllu' in path:
        with open(path, 'r', encoding='utf8') as f:
            data = f.read().strip()
    else:
        data = path.strip()

    instances = re.split('\n{2,}', string=data)
    #multi_roots_count, cycle_count = 0, 0
    tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents = collections.defaultdict(list), [], [], [], [] * len(instances)
    for i, instance in enumerate(instances):
        if i in discard:
            continue
        instance = instance.strip()
        sent, toknized_sent, upos, xpos, lemma = None, [], [], [], []
        gold_h, gold_r = [], []
        root = []
        mw = {}
        for l in instance.split('\n'):
            if l.startswith("#"):
                if l.startswith("# text = "):
                    sent = l.split("# text = ")[-1]
                continue
            tks = l.split('\t')
            if not re.match("^\d+$", string=tks[0]):
                if "-" in tks[0]:
                    pos = tks[0].split('-')
                    mw.update({j: tks[1] for j in range(int(pos[0])-1, int(pos[1]))})
                continue
            toknized_sent.append(tks[1])
            upos.append(tks[3])
            lemma.append(tks[2])
            xpos.append(tks[4])

            gold_h.append(int(tks[6]))
            gold_r.append(tks[7])
            if tks[6] == '0':
                root.append(int(tks[0]))
        '''
        if len(mw) > 4:
            print(instance)
            print(mw)
        '''
        #    raise ValueError
        assert sent is not None, instance
        tokenized_sentences['sentence'].append(sent)
        tokenized_sentences['tokenized'].append(" ".join(toknized_sent))
        tokenized_sentences['upos'].append(upos)
        tokenized_sentences['xpos'].append(xpos)
        tokenized_sentences['lemma'].append(lemma)

        gold_heads.append(gold_h)
        gold_rels.append(gold_r)
        gold_roots.append(root[0] if len(root) > 0 else -1)
        mw = collections.OrderedDict(sorted(mw.items()))
        mw_parents.append(mw)

        #if exists_cycle(gold_h):
        #    cycle_count += 1
           # print(instance)
            #break
        #break
        #print(tokenized_sentences)
        #raise ValueError
    #print(cycle_count)
    #print(multi_roots_count)
    #raise ValueError
    return tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents#, multi_roots_count, cycle_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--language', '-l', required=True)
    parser.add_argument('--mode', '-m', type=str, default='parse')
    parser.add_argument('--data_dir', type=str, default='../../data/ud_treebanks/ud-treebanks-v2.12')
    parser.add_argument('--treebanks', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--parser', '-p', type=str, default='stanza')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--leaderboard', type=str, default="evaluation/eval_2.csv")
    parser.add_argument("--checkpoint", '-c', type=str)
    parser.add_argument("--tokenized", action='store_true')
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--use_cache", action='store_true')

    #global args
    args = parser.parse_args()

    print(args)

    abb2lang = {
        'en': 'English',
        'de': 'German'
    }
    if not args.data_path:
        paths = glob(os.path.join(args.data_dir, f"UD_{abb2lang[args.language]}*/*test.conllu"))
        if args.treebanks:
            treebanks = args.treebanks.split(",")
        
    else:
        paths = [args.data_path]

    if not args.use_cache:
        parser = init_parser(args)

    final = collections.defaultdict(list)

    for path in sorted(paths):
        if args.data_path is None:
            dataset = path.split('/')[-2].split('-')[-1]
            print(dataset)
            if args.treebanks:
                if dataset not in treebanks:
                    print(f"skip {dataset}...")
                    continue
        else:
            dataset = '-'.join(path.split('/')[-2:])

        out_path = f"cache/{args.language}_{dataset}_{args.parser}{'_tokenized' if args.tokenized else ''}.conllu"
        discarded = []
        #if not os.path.exists(out_path) or args.parser in ['towerparse', 'stackpointer'] or not args.use_cache:
        if not args.use_cache:
            tokenized_sentences, gold_heads, gold_rels, gold_roots, mw_parents = read_conllu(path)
            if args.tokenized:
                results = parser.parse(sentences=tokenized_sentences['tokenized'], out='conllu', tokenized=True, mw_parents=mw_parents)
            else:
                results = parser.parse(sentences=tokenized_sentences['sentence'], out='conllu', tokenized=False)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(results)
            if args.parser in ['towerparse', 'stackpointer', 'biaffine', 'crf2o']:
                discarded = parser.discard
        #else:
        #    results =
        #if not args.tokenized:
        gold_ud, gold_cycle_count, gold_multi_roots_count = load_conllu_file(path, discarded=discarded)
        system_ud, system_cycle_count, system_multi_roots_count = load_conllu_file(out_path)
        evaluation = evaluate(gold_ud, system_ud)

        uas = evaluation["UAS"].f1
        las = evaluation["LAS"].f1

        #else:
        '''
        else:
            _, gold_heads, gold_rels, gold_roots = read_conllu(path, discard=discarded)
            _, system_heads, system_rels, system_roots = read_conllu(results if not args.use_cache else out_path)
            assert len(gold_heads) == len(system_heads)
            uas, las, total = 0, 0, 0
            for i in range(len(gold_heads)):
                heads1 = gold_heads[i]
                heads2 = system_heads[i]
                assert len(heads1) == len(heads2)
                uas += sum([h1 == h2 for h1, h2 in zip(heads1, heads2)])
                rels1 = gold_rels[i]
                rels2 = system_rels[i]
                assert len(rels1) == len(rels2)
                assert len(heads1) == len(rels2)
                las += sum([heads1[j] == heads2[j] and rels1[j] == rels2[j] for j in range(len(rels1))])
                total += len(heads1)

            uas /= total
            las /= total

            system_cycle_count = -1
            system_multi_roots_count = -1
        '''

        print("UAS F1 Score: {:.2f}".format(100 * uas))
        print("LAS F1 Score: {:.2f}".format(100 * las))
        print(f"{system_cycle_count} cycles detected.\n{system_multi_roots_count} multi roots detected.")

        final['language'].append(args.language)
        final['treebank'].append(dataset)
        final['tokenized'].append(args.tokenized)
        final['parser'].append(args.parser)
        #final['root_acc'].append(root_acc)
        final['uas'].append(uas)
        final['las'].append(las)
        final['cycle_count'].append(system_cycle_count)
        final['multi_roots_count'].append(system_multi_roots_count)
        final['skipped'].append(len(discarded))
        #break

    if len(paths) > 0:
        final['language'].append(args.language)
        final['treebank'].append('avg')
        final['tokenized'].append(args.tokenized)
        final['parser'].append(args.parser)
        final['uas'].append(np.average(final['uas']))
        final['las'].append(np.average(final['las']))
        final['cycle_count'].append(np.sum(final['cycle_count']))
        final['multi_roots_count'].append(np.sum(final['multi_roots_count']))
        final['skipped'].append(np.sum(final['skipped']))


    final = pd.DataFrame(final)

    print(final)
    if not os.path.exists(args.leaderboard):
        final.to_csv(args.leaderboard, mode='w', index=False)
    else:
        final.to_csv(args.leaderboard, mode='a', index=False, header=False)



