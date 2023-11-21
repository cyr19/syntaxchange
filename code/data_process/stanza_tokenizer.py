import pandas as pd
import stanza
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="hansard")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    args = parser.parse_args()

    data = args.data

    lang = 'en' if data == 'hansard' else "de"
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,mwt,pos", tokenize_no_ssplit=True, tokenize_batch_size=args.batch_size, use_gpu=True)

    # results 1 - 2000 sents
    '''
    data_dir = os.path.join(os.getcwd(), f"data/{data}_final/chatgpt_is_sent_processed_v1/")
    out_dir = os.path.join(os.getcwd(), f"data/{data}_final/stanza_tokenized_v3/")
    '''
    # results 2 - 200,000 sents
    data_dir = os.path.join(os.getcwd(), f"../../data/{data}_final/spacy_processed_v3/")
    out_dir = os.path.join(os.getcwd(), f"../../data/{data}_final/stanza_tokenized_v4/")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = sorted(glob(data_dir+"*.csv"), reverse=False)


    for file in tqdm(files):
        decade = file.split('/')[-1][:4]
        #if int(decade) not in [1840, 1870, 2000]:
        #    continue
        print(file)
        df = pd.read_csv(file, delimiter="\t")

        # only need for chatgpt identified sents
        '''
        print(type(df['is_sent'].values[0]))
        
        if not (isinstance(df['is_sent'].values[0], np.bool_) or isinstance(df['is_sent'].values[0], bool)):
            print('...')
            df['is_sent'] = [True if t == "True" else False for t in df['is_sent']]
        df = df[df.is_sent == True]
        '''

        print(len(df))
        doc = nlp(list(df['sent']))
        tokens, postags = [], []
        for i, sentence in enumerate(doc.sentences):
            tokens.append("|".join([t.text for t in sentence.words]))
            postags.append("|".join([t.upos for t in sentence.words]))

        df['tokenized'] = tokens
        df['len'] = [len(t.split("|")) for t in tokens]
        df['pos'] = postags

        df.to_csv(out_dir+f"{decade}_{len(df)}.csv", sep='\t', index=False)




