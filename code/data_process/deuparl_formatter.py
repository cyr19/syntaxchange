import collections
import json
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
# https://github.com/SteffenEger/ocr_spelling_deuparl/blob/main/code_from_other_projects/tobiwalter_process_reichstag_data.py



def load_metadata(data='Reichstag'):
    data_dir = f'data/DeuParl-v2/{data} Data/'
    with open(data_dir+'details.json', 'r') as f:
        meta = json.load(f)
    #print(meta.keys())
    final = []
    size = []
    for file, info in tqdm(meta.items()):
        year = str(info['year'])
        decade = year[:3]+'0s'
        month = info['month']
        day = info['day']
        date = f"{year}-{month}-{day}"
        file_path = data_dir+f'{decade}/{file}'
        #if data == 'Bundestag':
        #    with open(file_path, 'r') as f:
        #        lines = f.read()
        #    if len(re.findall('\n\n')) >= 3:
        #        docs = lines.split('\n\n')
        #else:
        #if data == 'Reichstag':
        with open(file_path, 'r') as f:
            lines = f.readlines()
    #print(len(lines))

        docs = []
        lines = [l for l in lines if l!='\n']
        for i in range(0, len(lines), 50):
            docs.append(''.join(lines[i: i+50]))
        size.append(len(lines))
        if len(lines) == 0:
            print(file)
        #print(f"{file}: {len(docs)}")
        #texts = text.split('\n\n\n')

        n = len(docs)
        if data == 'Bundestag':
            info['type'] = None
        final.append(pd.DataFrame({'era': [info['era']]*n, 'period': [info['period']]*n, 'type': [info['type']]*n, 'text': docs, 'date': [date]*n, 'file_path': [file_path]*n}))
        '''
        if data == 'Bundestag':
            info['type'] = None
        final.append(pd.DataFrame(
            {'era': [info['era']], 'period': [info['period']], 'type': [info['type']], 'text': text,
             'date': [date], 'file_path': [file_path]}))
        '''
        #raise ValueError
    #print(np.mean(size))
    #print(np.min(size))
    #print(np.max(size))
    final = pd.concat(final, ignore_index=True)
    return final


if __name__ == '__main__':
    df1 = load_metadata('Reichstag')
    df2 = load_metadata('Bundestag')
    df = pd.concat([df1, df2], ignore_index=True)
    #df.to_csv('data/DeuParl-v2/merged.csv', sep='\t', index=False)
    df['decade'] = [year[:3]+'0' for year in df['date']]
    for decade, group in tqdm(df.groupby('decade')):
        #outpath = f'data/DeuParl-v2/formatted_fix_nlines_100/{decade}.csv'
        #outpath = f'data/DeuParl-v2/formatted_fixnlines30+doublelinebreak/{decade}.csv'
        outpath = f'data/DeuParl-v2/formatted_fixnlines50/{decade}.csv'
        group.to_csv(outpath, sep='\t', index=False)
    print(collections.Counter(df['decade']))

