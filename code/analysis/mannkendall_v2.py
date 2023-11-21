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

def define_suffix():
    if args.random:
        return 'random'
    elif args.balance:
        return 'balance'
    elif args.pos:
        return 'pos'

parsers = args.parsers.split(',')
results = collections.defaultdict(list)
for parser in parsers:
    if args.random:
        df = pd.read_csv(f'tables/{data}/{parser}.csv')
        df['decade'] = [int(d[:3]+'0') for d in df['date']]
        #df = df.groupby('decade').mean()
        #print(df)
        #print(df.columns)
        #raise ValueError

        for metric in df.columns[3:]:
            print(f"{parser}-{metric}...")
            if 'decade' in metric or 'year' in metric or 'len' in metric:
                continue
            x = np.array([[m, d] for m, d in zip(df[metric], df['len'])])
            results['language'].append('de' if data == 'deuparl' else 'en')
            results['dataset'].append(define_suffix())
            results['parser'].append(parser)
            results['len'].append(np.mean(df['len']))
            results['metric'].append(metric)
            #results['aggregation'].append('mean_per_decade')
            results['aggregation'].append(False)
            ori_test = mk.original_test(list(df[metric]))#[0]

            results['OriMannKendall'].append(ori_test[0])
            results['Ori_p'].append(ori_test[2])
            results['Ori_slope'].append(ori_test[-2])

            partial_test = mk.partial_test(x)#[0]
            results['PartialMannKendall'].append(partial_test[0])
            results['partial_p'].append(partial_test[2])
            results['partial_slope'].append(partial_test[-2])

    if args.balance:
        def find_len_group(length, lengths):
            for i in range(len(lengths)):
                if i == len(lengths) - 1 and length >= lengths[-1]:
                    return lengths[-1]
                if lengths[i] <= length < lengths[i + 1]:
                    return lengths[i]
            return None



        df = pd.read_csv(f'tables/{data}/{parser}_balanced.csv')
        df['decade'] = [int(d[:3] + '0') for d in df['date']]

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True)
        print(df)
        for metric in df.columns[2:]:
            if 'decade' in metric or 'year' in metric or 'len' in metric:
                continue
            print(f"{parser}-{metric}...")
            x = np.array([[m, d] for m, d in zip(df[metric], df['len'])])
            results['language'].append('de' if data == 'deuparl' else 'en')
            results['dataset'].append(define_suffix())
            results['parser'].append(parser)
            results['len_group'].append(np.mean(df['len']))
            #results['len_group'].append('all')
            results['metric'].append(metric)
            # results['aggregation'].append('mean_per_decade')
            results['aggregation'].append(False)

            ori_test = mk.original_test(list(df[metric]))#[0]
            results['OriMannKendall'].append(ori_test[0])
            results['Ori_p'].append(ori_test[2])
            results['Ori_slope'].append(ori_test[-2])

            partial_test = mk.partial_test(x)#[0]
            results['PartialMannKendall'].append(partial_test[0])
            results['partial_p'].append(partial_test[1])
            results['partial_slope'].append(partial_test[-2])


        df = pd.read_csv(f'tables/{data}/{parser}_balanced.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True)
        print(df)
        lengths = [5, 10, 15, 20, 30, 40, 50, 60, 70]
        df['len_group'] = [find_len_group(l, lengths) for l in df['len']]
        #['year'] = [int(d[:4]) for d in df['date']]
        df = df.groupby('len_group')
        #print(len(df))
        for length, group in df:
            #print(group)
            #group = group.groupby('decade_group').mean()
            for metric in group.columns[2:]:
                if 'decade' in metric or 'year' in metric or 'len' in metric:
                    continue
                print(f"{parser}-{metric}-{length}...")
                x = np.array([[m, d] for m, d in zip(group[metric], group['len'])])
                results['language'].append('de' if data == 'deuparl' else 'en')
                results['dataset'].append(define_suffix())
                results['parser'].append(parser)
                results['len_group'].append(length)
                results['metric'].append(metric)
                #results['aggregation'].append('mean_per_len-decade_group')
                results['aggregation'].append(False)

                #print(df[metric])
                ori_test = mk.original_test(list(group[metric]))#[0]
                results['OriMannKendall'].append(ori_test[0])
                results['Ori_p'].append(ori_test[2])
                results['Ori_slope'].append(ori_test[-2])

                partial_test = mk.partial_test(x)#[0]
                results['PartialMannKendall'].append(partial_test[0])
                results['partial_p'].append(partial_test[2])
                results['partial_slope'].append(partial_test[-2])
                #results['OriMannKendall'].append(mk.original_test(list(group[metric]))[0])
                #results['PartialMannKendall'].append(mk.partial_test(x)[0])


                #for

    if args.pos:
        df = pd.read_csv(f'tables/{data}/{parser}_pos.csv')
        df = df.groupby('postags')
        for pos, group in df:
            for metric in group.columns[5:]:
                x = np.array([[m, d] for m, d in zip(group[metric], group['len'])])
                results['language'].append('de' if data == 'deuparl' else 'en')
                results['dataset'].append(define_suffix())
                results['parser'].append(parser)
                results['postags'].append(pos)
                results['len'].append(np.mean(group['len']))
                results['metric'].append(metric)
                results['aggregation'].append(False)
                results['OriMannKendall'].append(mk.original_test(list(group[metric]))[0])
                results['PartialMannKendall'].append(mk.partial_test(x)[0])

results = pd.DataFrame(results)
print(results)


#results.to_csv(f"tables/trend_test/{data}_{define_suffix()}_no_aggregation_sep.csv", index=False)
results.to_csv(f"tables/trend_test/{data}_{define_suffix()}_no_aggregation_sorted.csv", index=False)


