from abc import abstractmethod
import torch
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
import stanza

class UniversalParser(object):
    def __init__(self, parser_type, ckpt_dir=None, batch_size=128, language='de'):
        self.parser_type = parser_type
        self.batch_size = batch_size
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_dir = ckpt_dir

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

    def convert_to_conull_mw(self, results, sentences):
        s = ''
        for i, result in enumerate(results):
            #s += f"# text_id = {result[0]['tid']}\n# sent_id = {i}\n# text = {sentences[i]}\n"
            s += f"# text_id = {result[0]['tid']}\n# text = {sentences[i]}\n"
            mw_spans = self.align(sentence_result=result, sentence=sentences[i])
            mw_starts = [s[0] for s in mw_spans]
            for j, tmp_r in enumerate(result):
                if j in mw_starts:
                    span = [sp for sp in mw_spans if sp[0] == j][0]
                    start = j+1
                    end = start + len(span)-1
                    s += f"{start}-{end}\t\t\t\t\t\t\t\t\t\t"
                s += f"{tmp_r['id']}\t{tmp_r['token']}\t_\t_\t{tmp_r['pos']}\t_\t{tmp_r['head_id']}\t{tmp_r['deprel']}\t_\t_\n"
            s += "\n"
        return s

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
            #if len(mw_parents[i])>4:
           #if "in dem selben" in sentences[i]:
               # print(mw_parents[i])
               # print(mw_spans)
                #raise ValueError
            mw_starts = [s[0] for s in mw_spans]
            for j, tmp_r in enumerate(result):
                if j in mw_starts:
                    s += f"{j+1}-{[s[1] for s in mw_spans if s[0] == j][0]+1}\t{mw_parents[i][j]}\t_\t_\t_\t_\t_\t_\t_\t_\n"
                s += f"{tmp_r['id']}\t{tmp_r['token']}\t_\t_\t{tmp_r['pos']}\t_\t{tmp_r['head_id']}\t{tmp_r['deprel']}\t_\t_\n"
            s += "\n"
            '''
            if "in dem selben" in sentences[i]:
                print(s)
                print(mw_parents[i])
                print(mw_spans)
                raise ValueError
            '''
        return s

    def tokenize_unused(self, sentences):
        tokenizer = stanza.Pipeline(self.language, processors='tokenize,mwt', mwt_batch_size=self.batch_size,
                                    tokenize_batch_size=self.batch_size, tokenize_no_ssplit=True,
                                    use_gpu=True if self.device == 'cuda' else False, verbose=False)
        doc = tokenizer(sentences)
        tokenized_sentences = []
        for i, sentence in enumerate(doc.sentences):
            tokenized_sentences.append([word.text for word in sentence.tokens])

        assert len(sentences) == len(tokenized_sentences)
        return tokenized_sentences

    def tokenize(self, sentences, postag=False):
        #raise ValueError
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


            #if len(mw_parents[-1]) > 1:
            #if i == 1:
            #    print(sentences[i])
            #    print(mw_parents[i])
            #    raise ValueError

        #print(tokenized_sentences[1])
        #print(sentences[1])

        assert len(sentences) == len(tokenized_sentences)
        assert len(mw_parents) == len(sentences)
        #raise ValueError
        if postag:
            return tokenized_sentences, postags, mw_parents
        else:
            return tokenized_sentences, mw_parents

'''
class UniversalDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
'''
