from abc import abstractmethod
import torch
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
import stanza
import gdown
from zipfile import ZipFile
import os
import wget
import tarfile
import json
import string

class UniversalParser(object):
    def __init__(self, parser_type, ckpt_dir=None, batch_size=128, language='de'):
        self.parser_type = parser_type
        self.batch_size = batch_size
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if ckpt_dir or parser_type=='stanza':
            self.ckpt_dir = ckpt_dir
        else:
            output = f'./checkpoints/{parser_type}_{language}'

            if not os.path.exists(output):
                self.download_checkpoint(output)

            if self.parser_type in ['biaffine', 'crf2o', 'stackpointer']:
                self.ckpt_dir = output + f'/v2.12_merged_{language}'
            elif self.parser_type == 'towerparse':
                self.ckpt_dir = output + ('/UD_English-EWT' if language == 'en' else "/UD_German-HDT")

    def download_checkpoint(self, output):
        urls = {
            'biaffine': {
                'en': "17rWkylDxDeSFWbU404wQQYA7XW8xP5fV",
                'de': "1MGTI4UIPmc-n8CBHXG3aTtsOPay59_xs"
            },
            'stackpointer': {
                'en': "1B4IItwuN5TQzWtSXEeEidtbrQZVW-AUu",
                'de': "13ISEZvAeAWX-7f6sJE4c-CRVkmaBd-GU"
            },
            'crf2o': {
                'en': "1GkGR20cyHNwTsQYohaGuclEVgzd6hco4",
                'de': "1bAgJoWOycShtjATbSZgE65PJjgIChPJ8"
            },
            'towerparse': {
                'en': "http://data.dws.informatik.uni-mannheim.de/tower/UD_English-EWT.tar.gz",
                'de': "http://data.dws.informatik.uni-mannheim.de/tower/UD_German-HDT.tar.gz"
            }
        }
        
        url = urls[self.parser_type][self.language]

        if self.parser_type in ['biaffine', 'crf2o', 'stackpointer']:
            gdown.download("https://drive.google.com/uc?id="+url, output+'.zip')
            with ZipFile(output+'.zip', 'r') as zip_ref:
                zip_ref.extractall(output)
            

        elif self.parser_type == 'towerparse':
            wget.download(url, output+".tar.gz")
            file = tarfile.open(output+".tar.gz")
            file.extractall(output)
            file.close()

        

    @abstractmethod
    def init_parser(self):
        pass

    @abstractmethod
    def parse(self, sentences):
        pass

    @abstractmethod
    def load_data(self, sentences):
        pass

    def align(self, sentence_result, sentence):
        spans = []
        span = []
        for i, tmp_r in enumerate(sentence_result):
            if tmp_r['token'].lower() not in sentence.lower():
                span.append(i)
            else:
                if len(span) > 1:
                    spans.append(span)
                span = [i]

        return spans

    def convert_to_conull(self, results, sentences, mw_parents):
        s = ''
        for i, result in enumerate(results):
            s += f"# text_id = {result[0]['tid']}\n# text = {sentences[i]}\n"
            mw_spans = []
            if len(mw_parents[i]) > 0:
                start = None
                end = None
                for k, v in mw_parents[i].items():
                    if start is None:
                        start = k
                        end = k
                    else:
                        if k == end + 1 and k != list(mw_parents[i].keys())[-1] and mw_parents[i][k] == mw_parents[i][start]:
                            end = k
                        else:
                            if k == list(mw_parents[i].keys())[-1]:
                                mw_spans.append((start, k))
                            else:
                                mw_spans.append((start, end))
                                start = k
                                end = k
            mw_starts = [s[0] for s in mw_spans]
            for j, tmp_r in enumerate(result):
                if j in mw_starts:
                    s += f"{j+1}-{[s[1] for s in mw_spans if s[0] == j][0]+1}\t{mw_parents[i][j]}\t_\t_\t_\t_\t_\t_\t_\t_\n"
                s += f"{tmp_r['id']}\t{tmp_r['token']}\t_\t_\t{tmp_r['pos']}\t_\t{tmp_r['head_id']}\t{tmp_r['deprel']}\t_\t_\n"
            s += "\n"

        return s

    def tokenize(self, sentences, postag=False):
        tokenizer = stanza.Pipeline(self.language, processors='tokenize,mwt,pos', mwt_batch_size=self.batch_size,
                                    tokenize_batch_size=self.batch_size, tokenize_no_ssplit=True,
                                    use_gpu=True if self.device == 'cuda' else False, verbose=False)
        doc = tokenizer(sentences)
        tokenized_sentences = []
        postags = []
        mw_parents = []
        for i, sentence in enumerate(doc.sentences):
            tokenized_sentences.append([word.text for word in sentence.words])
            postags.append([word.upos for word in sentence.words])
            mw_parents.append({j: word.parent.text for j, word in enumerate(sentence.words) if word.text != word.parent.text})


        assert len(sentences) == len(tokenized_sentences)
        assert len(mw_parents) == len(sentences)
        
        if postag:
            return tokenized_sentences, postags, mw_parents
        else:
            return tokenized_sentences, mw_parents


