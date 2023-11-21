import json


#def init_parser(parser_name='stanza', language=None):
def init_parser(args):
    #print(args)
    if args.language:
        lang = args.language
    else:
        data2lang = {
            "deuparl": 'de',
            "hansard": 'en'
        }
        lang = data2lang[args.data]

    if args.parser == "towerparse":
        from tower_parser import TowParser
        parser = TowParser(batch_size=args.batch_size, language=lang, ckpt_dir=args.checkpoint)
    elif args.parser == "stackpointer":
        from stackpointer_parser import StackPointerParser
        parser = StackPointerParser(ckpt_dir=args.checkpoint, language=lang, batch_size=args.batch_size)
    elif args.parser == "stanza":
        from stanford_parser import StanfordParser
        parser = StanfordParser(language=lang, batch_size=args.batch_size)
    elif args.parser in ["biaffine", "crf2o"]:
        from biaffine_supar import SuparBiaffineParser
        parser = SuparBiaffineParser(batch_size=args.batch_size, language=lang, ckpt_dir=args.checkpoint)
    elif args.parser == 'corenlp':
        from corenlp_parser import StanfordParser
        parser = StanfordParser(language=lang, batch_size=args.batch_size, ckpt_dir=args.checkpoint, port=args.port)
    elif args.parser == 'mrc':
        from mrc_parser import MRCParser
        parser = MRCParser(language=lang, batch_size=args.batch_size, ckpt_dir=args.checkpoint)
    else:
        raise NotImplementedError("This parser is not implemented.")
    return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob
    from tqdm import tqdm
    import os
    import pandas as pd
    import json

    args_parser = ArgumentParser()
    args_parser.add_argument("--data", '-d', type=str, required=True)
    args_parser.add_argument("--language", '-l', type=str)
    args_parser.add_argument("--parser", "-p", type=str, required=True)
    args_parser.add_argument("--checkpoint", "-c", type=str)
    args_parser.add_argument("--version", "-v", type=int, default=4)
    args_parser.add_argument("--start", '-s', type=int, default=1800)
    args_parser.add_argument("--end", "-e", type=int, default=2020)
    args_parser.add_argument("--batch_size", '-b', type=int, default=4)
    args_parser.add_argument("--port", type=int, default=9000)
    args = args_parser.parse_args()

    out_dir = f"../../data/{args.data}_final/parsed_v{args.version}/{args.parser}/"
    data = f"../../data/{args.data}_final/stanza_tokenized_v{args.version}/*.csv"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Out Directory doesn't exist. Makedirs.")

    parser = init_parser(args)

    for file in tqdm(sorted(glob(data))):
        decade = int(file.split('/')[-1][:4])
        print()
        print(decade)
        if args.start <= decade <= args.end:
            df = pd.read_csv(file, delimiter='\t')#[:10]#[119190:119200]
            #if args.parser in ["stanza", "biaffine"]:
            sentences = list(df['tokenized'])
            #if parser != 'corenlp':
            results = parser.parse(sentences=sentences, out='conllu', tokenized=True, mw_parents=[[]] * len(sentences))
            #else:

            '''
            
            
            if args.parser in ["stanza"]:
                sentences = list(df['sent'])
                results = parser.parse(sentences=sentences, out='conllu', tokenized=False)
            elif args.parser == "corenlp":
                sentences = list(df['tokenized'])#[:5]
                results = parser.parse(sentences=sentences, out='conllu', tokenized=False)
                #print(results)
                #raise ValueError
            else:
                tokenized_sentences = list(df['tokenized']) #+ ["fa "*1000]
                results = parser.parse(sentences=tokenized_sentences, out='conllu', tokenized=True)
            '''
            #print(results)
            #print(results)
            #raise ValueError
            with open(os.path.join(out_dir, f"{decade}.conllu"), 'w', encoding='utf8') as f:
                f.write(results)
            if args.parser in ["towerparse", "stackpointer", "biaffine", "crf2o"]:
                with open(os.path.join(out_dir, f"{decade}_discarded.json"), 'w', encoding='utf8') as f:
                    json.dump(parser.discard, f, indent=2)
        else:
            print("skipped")

        #raise ValueError