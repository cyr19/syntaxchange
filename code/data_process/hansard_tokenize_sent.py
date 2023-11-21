import re
import string
#from spacy.language import Language
import pandas as pd
import spacy
from glob import glob
from tqdm import tqdm


def paragraph_level_preprocess(text):
    text = text.strip()

    text = re.sub(r'\n\[\d+\n', ' ', string=text) # 数字加换行
    text = re.sub(r'\n\d+\n', ' ', string=text) # 数字加换行

    text = re.sub(r'\-\n', '', text)
    text = re.sub(r'\—+', "—", text)
    text = re.sub(r"\-\-+", "—", text)
    text = text.replace('—', ' — ')
    text = text.replace(' -', '-')  # theilweise -fortbestehen
    text = re.sub(r'\s+', ' ', string=text)
    text = re.sub(r"\'{2}|\"{2}", "\"", string=text)

    return text


def sentence_level_preprocess(text, tags, tokens):
    text = text.strip()
    if len(text.split('; ')) > 1:  # Präsident: Der Abgeordnete Stumm hat das Wort.
        sub_sents = text.split('; ')
        try:
            if not any(['VB' in p for p in tags[: tokens.index(';')]]):
                # if sub_sents[1][0].isupper():
                text = '; '.join(sub_sents[1:])
                #print(tokens)
                #print(tags)
                tokens = tokens[tokens.index(";") + 1:]
                tags = tags[tokens.index(";") + 1:]
        except:
            pass
    text = text.strip()

    if len(text.split(': ')) > 1: # Präsident: Der Abgeordnete Stumm hat das Wort.
        sub_sents = text.split(': ')
        #print(text)
        #print(tokens)
        try:
            if not any(['VB' in p for p in tags[: tokens.index(':')]]): # Dr. Klejdzinski (SPD): "):" would be one token, so the this line is not fully correct
                text = ': '.join(sub_sents[1:])
                tokens = tokens[tokens.index(":") + 1:]
                tags = tags[tokens.index(":") + 1:]
        except:
            pass

    text = text.strip()#.strip('\n')

    if len(re.findall('\"', text)) % 2 == 1:
        if text.endswith('\"'):
            text = text[:-1]
            tokens = tokens[:-1]
            tags = tags[:-1]
        elif text.startswith('\"'):
            text = text[1:]
            tokens = tokens[1:]
            tags = tags[1:]

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

    text = re.sub(r'\s{2,}', ' ', text)
    #print(text, tokens, tags)
    return text, tokens, tags


def sents_postprocsess(sents, tags, tokens, dates):
    assert len(dates) == len(sents), f'{len(sents)} -- {len(dates)}'
    new_sents, new_tags, new_tokens, new_dates = [], [], [], []
    i = 0
    while i < len(sents):
        new_dates.append(dates[i])
        new_sent, new_tag, new_token = [sents[i]], tags[i], tokens[i]
        for j in range(i+1, len(sents)):
            if len(sents[j]) == 0:
                new_sent.append('')
                continue
            if ((sents[i][-1] in [';', ':', ',', '\"', '—', '-']) and (sents[j][0].isalpha() and sents[j][0].islower()))\
                    or (len(sents[i]) >= 4 and sents[i][-4:] == 'hon.'):
                new_sent.append(sents[j])
                new_tag += '|'+tags[j]
                new_token += '|'+tokens[j]
            else:
                break
        i += len(new_sent)
        new_sent = paragraph_level_preprocess(' '.join(new_sent))
        new_sents.append(new_sent)
        new_tags.append(new_tag)
        new_tokens.append(new_token)
    return new_sents, new_tags, new_tokens, new_dates


def is_sent(text, pos, tokens):
    pos_list = pos.split('|')
    token_list = tokens.split('|')
    if len(text) == 0:
        return False

    pattern = r"[A-ZÄÖÜ]"
    if not re.match(pattern, text[0]):
        return False
    if text[-1] not in ['.', '?', '!']:
        return False
    if not any(['VB' in p for p in pos_list]):
        return False
    if len(re.findall('\"', text)) % 2 == 1 or (len(re.findall('\)', text)) != len(re.findall('\(', text))):
        return False
    if len(re.findall('\'', text)) % 2 == 1:
        return False
    try:
        if ';' in pos:
            if not any(['VB' in p for p in pos[:token_list.index(';')]]):
                return False
        if ':' in pos:
            if not any(['VB' in p for p in pos[:token_list.index(':')]]):
                return False
    except:
        pass

    return True


def spacy_tokenize(df, model, threshold=20000):
    #texts = df['text'].tolist()
    dfs = []
    num_sent = 0
    pbar = tqdm(total=threshold)

    #docs = model.pipe(texts=df['text'],)
    for i, (text, date) in enumerate(zip(df['text'], df['date'])):
        all_sents, tokens, simple_pos_tags, pos_tags, ids, dates = [], [], [], [], [], []

        if len(text.split('\n')) > 50:
            lines = text.split('\n')
            texts = [''.join(lines[pi: pi+50]) for pi in range(0, len(lines), 50)]
            print('\nmore than 50 lines.\n')
        else:
            texts = [text]

        #texts = [text]

        for text in texts:
            if len(text.split()) > 1:
                text = paragraph_level_preprocess(text)
                #print(text)
                doc = model(text)
                #for d in docs:
                 #   print(dir(d))
                #raise ValueError
                sents = [s.text for s in doc.sents]
                #print(len(sents))
                #num_sent = 0
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
        df = df[['is_sent', 'sent', 'pos', 'date', 'tokenized']]
        df = df[df.is_sent == True]
        dfs.append(df)
        num_sent += len(df)
        pbar.update(len(df))
        if num_sent >= threshold:
            break
    pbar.close()
    df = pd.concat(dfs, ignore_index=True)
    #df['is_sent'] = [is_sent(s, t, tk) for s, t, tk in zip(df['sent'], df['pos'], df['tokenized'])]
    #new_df['is_sent'] = [is_sent(s, t) for s, t in zip(new_df['sent'], new_df['pos'])]
    #df = df[['is_sent', 'sent', 'pos', 'date', 'tokenized']]

    return df


if __name__ == '__main__':
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_trf')

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--start", "-s", type=int)
    parser.add_argument("--end", "-e", type=int)
    parser.add_argument("--limit", "-l", type=int, default=1000000)
    parser.add_argument("--batch_size", "-b", type=int, default=10000)
    args = parser.parse_args()


    #out_dir = 'data/hansard_final/spacy_processed/'
    out_dir = '../../data/hansard_final/spacy_processed_v3/'
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #csv_files = sorted(csv_files)
    data_dir = '../../data/new_collected_hansard_csv'
    for decade in range(1800, 2021, 10):
        #if decade < 1980:
        #    continue
        print()
        print(decade)
        if int(decade) < args.start or int(decade) > args.end:
            continue

        decade_dir = os.path.join(data_dir, str(decade) + 's')
        csv_files = glob(os.path.join(decade_dir, '*/*.csv'))
        dfs = []
        for csv in csv_files:
            #print(csv)
            if os.stat(csv).st_size > 1:
                #print(os.stat(csv).st_size)
                df = pd.read_csv(csv, delimiter='\t')
                if len(df) != 0:
                    if df.columns[0] != 'chamber':
                        df.columns = ['chamber', 'date', 'section', 'text', 'zip_path']
                    dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna(subset='text').sample(frac=1, random_state=320)
        print(len(df))
        #raise ValueError
        #if len(df) > 8000:
            #df = df.sample(n=8000, random_state=320)  # .iloc[398:403]
        print(len(df))

        tmp = spacy_tokenize(df, nlp, threshold=args.limit)
        tmp = tmp[tmp.is_sent == True]
        tmp['index'] = range(len(tmp))
        tmp = tmp[['index', 'sent', 'pos', 'date']]

        tmp.to_csv(out_dir + f'{decade}_sents_{len(tmp)}.csv', index=False, sep='\t')


