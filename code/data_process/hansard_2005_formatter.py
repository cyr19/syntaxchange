import pandas as pd
import os
from hansard_xml_extractor import simple_preprocess
from tqdm import tqdm

if __name__ == '__main__':
    # download this file from https://zenodo.org/record/4843485#.ZCR7WexBz0p
    df = pd.read_csv('data/new_collected_hansard/hansard-speeches-v310.csv')#[:10]
    df = df[['date', 'speech', 'id']] #'major_heading', 'minor_heading']]
    df['year'] = [int(d[:4]) for d in df['date']]
    print(df.columns)
    df = df[df.year >= 2005]

    # chamber	date	section	text	zip_path
    df['chamber'] = [None] * len(df)
    df['section'] = [None] * len(df) #[row['major_heading']+'|'+row['minor_heading'] for _, row in df.iterrows()]
    df['decade'] = [d[:3]+'0s' for d in df['date']]
    df['month'] = [d.split('-')[1] for d in df['date']]

    for index, group in tqdm(df.groupby(['decade', 'year', 'month'])):
        #if index[1] < 2007: #and int(index[2]) < 3:
        #    continue
        out_dir = os.path.join('data/new_collected_hansard_csv', index[0], str(index[1]))
        #print(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        group.rename(columns={'speech': 'text', 'id': 'zip_path'}, inplace=True)
        group['text'] = [simple_preprocess(t) for t in group['text']]
        group = group[['chamber', 'date', 'section', 'text', 'zip_path']]
        out_path = os.path.join(out_dir, index[2]+'.csv')
        print(out_path)
        #raise ValueError
        if not os.path.exists(out_path) or os.stat(out_path).st_size == 0:
            group.to_csv(out_path, header=True, index=False, sep='\t', mode='w')
        else:
            group.to_csv(out_path, header=False, index=False, sep='\t', mode='a')

