import re
import string
from spacy.language import Language
import pandas as pd
import spacy
from glob import glob
from tqdm import tqdm
import os

def paragraph_level_preprocess(text):
    #print(text)
    #raise ValueError
    text = text.strip()
    text = re.sub(r'\-\n', '', text)
    text = re.sub(r'\—+', "—", text)
    text = re.sub(r"\-\-+", "—", text)
    text = text.replace('—', ' — ')
    text = text.replace(' -', '-') # theilweise -fortbestehen
    text = re.sub(r'\s+', ' ', string=text)
    text = re.sub(r"\'{2}|\"{2}", "\"", string=text)
    text = re.sub(r" \"", "\" ", string=text) # space issue like: „Gemeindebehörde ''

    return text


def sentence_level_preprocess(text, tags, tokens):

    #text = text.lstrip('*')
    text = text.strip()#.strip('\n')
    # done: TODO: move to the beginning of this function and process tokens and tags as well!!!!
    if len(text.split('; ')) > 1:  # Präsident: Der Abgeordnete Stumm hat das Wort.
        sub_sents = text.split('; ')
        try:
            if not any(['VV' in p or 'VA' in p for p in tags[: tokens.index(';')]]):
                # if sub_sents[1][0].isupper():
                text = '; '.join(sub_sents[1:])
                tokens = tokens[tokens.index(";") + 1:]
                tags = tags[tokens.index(";") + 1:]
        except:
            pass

    if len(text.split(': ')) > 1:  # Präsident: Der Abgeordnete Stumm hat das Wort.
        sub_sents = text.split(': ')
        # print(text)
        # print(tokens)
        try:
            if not any(['VV' in p or 'VA' in p for p in tags[: tokens.index(
                    ':')]]):  # Dr. Klejdzinski (SPD): "):" would be one token, so the this line is not fully correct
                text = ': '.join(sub_sents[1:])
                tokens = tokens[tokens.index(":") + 1:]
                tags = tags[tokens.index(":") + 1:]
        except:
            pass

    text = text.strip()

    if len(re.findall('„', text)) != len(re.findall("\"", text)):
        if text.startswith('„'):
            text = text[1:]
            tokens = tokens[1: ]
            tags = tags[1: ]
        elif text.endswith('\"'):
            text = text[:-1]
            tokens = tokens[:-1]
            tags = tags[:-1]

    text = text.strip()

    #if text.startswith('—'):
    #    text = text[1:]
    #text = text.strip()

    tokenized = tokens
    if len(tokenized) == 0:
        return text
    try:
        if tokenized[0].isdigit() and tokenized[1][0].isupper():
            text = ' '.join(tokenized[1:])
            tokens = tokens[1:]
            tags = tags[1:]
    except:
        pass
    text = text.strip()

    tokenized = tokens
    try:
        if tokenized[0] in string.punctuation and tokenized[1][0].isupper():
            text = ' '.join(tokenized[1:])
            tokens = tokens[1:]
            tags = tags[1:]
    except:
        pass
    text = text.strip()

    #text = text.strip()
    text = re.sub(r'\s{2,}', ' ', text)

    return text, tokens, tags


def sents_postprocsess(sents, tags, tokens, dates):
    assert len(dates) == len(sents), f'{len(sents)} -- {len(dates)}'
    new_sents, new_tags, new_tokens, new_dates = [], [], [], []
    i = 0
    #for i in range(len(sents)-1):
    #print(len(sents))
    while i < len(sents):
        new_dates.append(dates[i])
        #print('i: ', i)
        new_sent, new_tag, new_token = [sents[i]], tags[i], tokens[i]
        for j in range(i+1, len(sents)):
            if (sents[i][-1] in [';', ':', ',', '\"', '—', '-']) and (sents[j][0].isalpha() and sents[j][0].islower()):
                new_sent.append(sents[j])
                new_tag += '|'+tags[j]
                new_token += '|'+tokens[j]
            else:
                break
        i += len(new_sent)
        new_sents.append(' '.join(new_sent))
        new_tags.append(new_tag)
        new_tokens.append(new_token)

    return new_sents, new_tags, new_tokens, new_dates


def is_sent(text, pos, tokens):
    pos_list = pos.split('|')
    token_list = tokens.split('|')
    #print(pos)
    if len(text) == 0:
        return False

    pattern = r"[A-ZÄÖÜ]"
    if not re.match(pattern, text[0]):
        #print(text)
        return False
    if text[-1] not in ['.', '?', '!']:
        return False
    if not any(['VV' in p or 'VA' in p for p in pos_list]):
        return False
    if (len(re.findall('\"', text)) != len(re.findall('„', text))) or (len(re.findall('\)', text)) != len(re.findall('\(', text))):
        return False
    if len(re.findall('\'', text)) % 2 == 1:
        return False
    try:
        if ';' in pos:
            if not any(['VV' in p or 'VA' for p in pos_list[:token_list.index(';')]]):
                return False
        if ':' in pos:
            if not any(['VV' in p or 'VA' for p in pos_list[:token_list.index(':')]]):
                return False
    except:
        pass
    return True


def spacy_tokenize(df, model, threshold=20000):
    #texts = df['text'].tolist()
    dfs = []
    #nn = 1
    num_sent = 0
    pbar = tqdm(total=threshold)
    #for i, (text, date) in tqdm(enumerate(zip(df['text'], df['date'])), total=len(df)):
    for i, (text, date) in enumerate(zip(df['text'], df['date'])):
        all_sents, tokens, simple_pos_tags, pos_tags, ids, dates = [], [], [], [], [], []
        if len(text.split('\n')) > 200:
            lines = text.split('\n')
            texts = [''.join(lines[pi: pi+200]) for pi in range(0, len(lines), 200)]
            print('\nmore than 200 lines.\n')
        else:
            texts = [text]
        for text in texts:
            if len(text) > 1:
                text = paragraph_level_preprocess(text)
                doc = model(text)
                sents = [s.text for s in doc.sents]

                for sent in sents:
                    if len(sent.split()) > 1:
                        ts = model(sent)
                        tags = [t.tag_ for t in ts]
                        tks = [t.text for t in ts]
                        try:
                            sent, tks, tags = sentence_level_preprocess(sent, tags, tks)

                            pos_tags.append('|'.join(tags))
                            tokens.append('|'.join(tks))
                            all_sents.append(sent)
                            dates.append(date)

                        except:
                            pass

        all_sents, pos_tags, tokens, dates = sents_postprocsess(all_sents, pos_tags, tokens, dates)
        df = pd.DataFrame({'sent': all_sents, 'pos': pos_tags, 'tokenized': tokens, 'date': dates})

        df['is_sent'] = [is_sent(s, t, tk) for s, t, tk in zip(df['sent'], df['pos'], df['tokenized'])]
        # df = df[['id', 'is_sent', 'sent', 'pos']]
        #df.to_csv('data/tmp.csv')

        df = df[df.is_sent == True]
        df = df[['is_sent', 'sent', 'pos', 'date', 'tokenized']]
        #print(df)
        dfs.append(df)
        num_sent += len(df)
        #print(num_sent)
        #raise ValueError
        pbar.update(len(df))
        if num_sent >= threshold:
            break
    pbar.close()
    df = pd.concat(dfs, ignore_index=True)

    #df['is_sent'] = [is_sent(s, t) for s, t in zip(df['sent'], df['simple_pos'])]


    #print(df)
    return df


if __name__ == '__main__':

    #extract_paragraphs(1940)
    #import thinc_gpu_ops
    #print(thinc_gpu_ops.AVAILABLE)
    import torch

    print(torch.cuda.is_available())

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--start", "-s", type=int)
    parser.add_argument("--end", "-e", type=int)
    parser.add_argument("--limit", "-l", type=int, default=1000000)
    args = parser.parse_args()

    #raise ValueError
    spacy.require_gpu()
    nlp = spacy.load('de_dep_news_trf')
    #csv_files = glob('data/deuparl_validation/*.csv')
    #csv_files = sorted(glob('data/DeuParl-v2/formatted_fixnlines50+doublelinebreak/*.csv'))
    csv_files = sorted(glob('../../data/DeuParl-v2/formatted_fixnlines50/*.csv'))
    #out_dir = 'data/deuparl_validation/spacy/test_v1_extended/'
    out_dir = '../../data/deuparl_final/spacy_processed_v3/'
    #import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for csv in tqdm(csv_files):
        print(csv)

        decade = csv.split('/')[-1][:4]
        #if int(decade) < 1960:
        #    continue
        #if decade not in ['1860', '1950', '2000']:
        #if decade not in ['2000']:
           # continue
        print()
        print(decade)
        if int(decade) < args.start or int(decade) > args.end:
            continue
        df = pd.read_csv(csv, sep='\t').sample(frac=1, random_state=320)

        #if len(df) > 1000:
        #    df = df.sample(n=1000, random_state=320)#.iloc[398:403]
        print(len(df))
        #print(df.columns)
        #raise ValueError
        #tmp = spacy_tokenize(df, nlp, threshold=float("inf"))
        #tmp = spacy_tokenize(df, nlp, threshold=1000000)
        tmp = spacy_tokenize(df, nlp, threshold=args.limit)
        #print(tmp.columns)
        #raise ValueError
        #tmp.to_csv(out_dir + f'{decade}.csv', index=False, sep='\t')
        tmp = tmp[tmp.is_sent == True]#.sample(frac=1, random_state=1)
        tmp['index'] = range(len(tmp))
        tmp = tmp[['index', 'sent', 'pos', 'date']]
        #tmp = tmp[['index', 'sent', 'pos', 'date']]
        print(tmp)
        #print(out_dir + f'{decade}_sents_{len(tmp)}.csv')
        #raise ValueError
        tmp.to_csv(out_dir + f'{decade}_sents_{len(tmp)}.csv', index=False, sep='\t')
        #print('..wrote..')
        tmp = tmp[['index', 'sent', 'date']]
        #except:
        #    pass

        '''
        sents_path = out_dir + f'sents.xlsx'
        if os.path.exists(sents_path):
            with pd.ExcelWriter(sents_path, mode='a') as writer:
                tmp.to_excel(writer, sheet_name=f'{decade}', index=False)
        else:
            with pd.ExcelWriter(sents_path, mode='w') as writer:
                tmp.to_excel(writer, sheet_name=f'{decade}', index=False)
        
        
        '''




