import collections
import os
from parse_all import *
from argparse import ArgumentParser
import pandas as pd
from types import SimpleNamespace

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--file_path', '-f', type=str)
    par.add_argument('--parser', '-p', type=str)
    par.add_argument('--lang', '-l', type=str, required=True)
    #par.add_argument("--batch_size", '-b', type=int)
    par.add_argument("--out_dir", type=str, default="parsing_outputs")
    par.add_argument("--columns", type=str)
    par.add_argument("--out_path", type=str, default=None)

    args = par.parse_args()

    import os
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.parser in ['stackpointer', 'towerparse', 'biaffine', 'crf2o']:
        batch_size = 64
    else:
        batch_size = 20000

    ckpt = {
        'stackpointer': f"/homes/ychen/syn_server/syntactic_change/code/parsers/NeuroNLP2/models/parsing/stackptr/v2.12_merged_{args.lang}",
        'biaffine': f"supar_github/checkpoints/supar_biaffine/v2.12_merged_{args.lang}",
        'crf2o': f"supar_github/checkpoints/supar_crf2o/v2.12_merged_{args.lang}",
        'corenlp': f"homes/ychen/stanza_corenlp"
    }

    conf = {
        'batch_size': batch_size,
        'port': 9000,
        'language': args.lang,
        'parser': args.parser,
        'checkpoint': ckpt[args.parser] if args.parser in ckpt.keys() else None
    }
    conf = SimpleNamespace(**conf)
    parser = init_parser(conf)

    if '.csv' in args.file_path:
        try:
            df = pd.read_csv(args.file_path)
        except:
            df = pd.read_csv(args.file_path, sep='\t')
        try:
            df = df[(df.correction != 'unknown') & (df.has_errors==True) & (df.chatgpt_has_errors==True)]
        except:
            pass
        #out = collections.defaultdict(list)
        #if args.columns is not None:
        columns = args.columns.split(',')
        #else:
        #    columns = ['sent']
        for c in columns:
            texts = list(df[c])
            results = parser.parse(sentences=texts, out='conllu', tokenized=False)

            #if args.out_path is None and args.out_dir is not None:
            with open(os.path.join(args.out_dir, args.file_path.split('/')[-1].replace('.csv', f'_{c}_{args.parser}.conllu')), 'w') as f:
                f.write(results)






