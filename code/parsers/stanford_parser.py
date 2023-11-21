import stanza
from basic_parser import *
from stanza.models.common.doc import Document
import re
import string

class StanfordParser(UniversalParser):
    def __init__(self, ckpt_dir=None, batch_size=1024, language='de'):
        super(StanfordParser, self).__init__(parser_type='stanza', language=language, batch_size=batch_size, ckpt_dir=ckpt_dir)
        #self.parser, self.tokenizer = self.init_parser()
        self.parser = self.init_parser()

    def init_parser(self):
        #parser = stanza.Pipeline(lang=self.language, processors='tokenize,pos,lemma,depparse')
        #parser = stanza.Pipeline(lang=self.language, processors='depparse', depparse_pretagged=True, use_gpu=True if self.device == 'cuda' else False, depparse_batch_size=self.batch_size)
        #parser = stanza.Pipeline(lang=self.language, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, depparse_pretagged=True,
        #                         use_gpu=True if self.device == 'cuda' else False, tokenize_batch_size=self.batch_size, depparse_batch_size=self.batch_size)
        parser = stanza.Pipeline(lang=self.language, processors='tokenize,pos,lemma,depparse',
                                 tokenize_pretokenized=True,
                                 use_gpu=True if self.device == 'cuda' else False,
                                 depparse_batch_size=self.batch_size)
        #tokenizer = stanza.Pipeline(self.language, processors='tokenize', tokenize_batch_size=self.batch_size, tokenize_no_ssplit=True, use_gpu=True if self.device == 'cuda' else False)
        #tokenizer = stanza.Pipeline(self.language, processors='tokenize,mwt,pos', tokenize_no_ssplit=True, tokenize_batch_size=self.batch_size, use_gpu=True if self.device=='cuda' else False)
        #return parser, tokenizer
        return parser

    def parse(self, sentences, tokenized=True, out='conllu', mw_parents=[]):
        results = []
        if not tokenized:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)
        else:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = [s.split() for s in sentences]
            #mw_parents = [[]] * len(tokenized_sentences)
        #print(tokenized_sentences[0])
        #print(mw_parents)
        #raise ValueError
        doc = self.parser(tokenized_sentences)

        for i, sentence in enumerate(doc.sentences):
            r = []
            #tid =
            for word in sentence.words:
                r.append({'tid': i, 'id': word.id, "token": word.text, "head_id": word.head,
                          "head": sentence.words[word.head - 1].text if word.head > 0 else "root",
                          "deprel": word.deprel, "pos": '_'})

            results.append(r)

        assert len(results) == len(sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results

    def load_data(self, sentences):
        """

        Args:
            sentences: list of tokens separated by white spaces; pre-tokenized.

        Returns: pretagged_doc format required by Stanza

        """
        #tokenized = sentences['tokenized'], sentences['upos']
       # print(tokenized)
        #print(upos)
        #print(sentences)
        #print("??", sentences[0])
        #raise ValueError

        pretagged_doc = []
        #doc = self.tokenizer(sentences)
        for i, sentence in enumerate(sentences['tokenized']):
            #print(sentences[0])
            #print(sentences[1])
            #print(sentence)
            #raise ValueError
            sent_data = []
            for j, token in enumerate(sentence.split()):
                tmp = {'id': j+1, 'text': token, 'lemma': sentences['lemma'][i][j], 'upos': sentences['upos'][i][j], 'xpos': sentences['xpos'][i][j],
                       'feats': '_'}
                sent_data.append(tmp)
            #print(sent_data)
            #raise ValueError
            pretagged_doc.append(sent_data)
        #print(pretagged_doc)
        #raise ValueError
        assert len(pretagged_doc) == len(sentences['tokenized']), f"{len(pretagged_doc)} -- {len(sentences)}"
        pretagged_doc = Document(pretagged_doc)

        return pretagged_doc

    def evaluate(self, data_path, punct=False):
        with open(data_path, 'r') as f:
            text = f.read()
        instances = re.split('\n{2,}', string=text)#[:10]
        toknized_sents, postags = [], []
        gold_heads, gold_rels = [], []
        gold_root = []
        pretagged_doc = []
        for i, instance in enumerate(instances):
            toknized_sent, postag = [], []
            gold_h, gold_r = [], []
            root = []
            sent_data = []
            for l in instance.strip().split('\n'):
                if l.startswith("#"):
                    continue
                tks = l.split('\t')
                if not re.match("^\d+$", string=tks[0]):
                    continue
                toknized_sent.append(tks[1])
                postag.append(tks[3])
                gold_h.append(int(tks[6]))
                gold_r.append(tks[7])
                if tks[6] == '0':
                    #gold_root.append()
                    root.append(int(tks[0]))
                tmp = {'id': int(tks[0]), 'text': tks[1], 'lemma': tks[2], 'upos': tks[3], 'xpos': tks[4],
                       'feats': tks[5]}
                sent_data.append(tmp)

            if len(root) != 1:
                continue
            #print('....')
            gold_root.append(root[0])
            toknized_sents.append(toknized_sent)
            postags.append(postag)
            gold_heads.append(gold_h)
            gold_rels.append(gold_r)

            pretagged_doc.append(sent_data)

        assert len(pretagged_doc) == len(gold_root), f"{len(pretagged_doc)} -- {len(gold_root)}"
        pretagged_doc = Document(pretagged_doc)

        results = self.parse(pretagged_doc)
        #print(results)

        correct_heads, correct_rels, correct_root = 0, 0, 0
        num_punct = 0
        for i in range(len(gold_root)):
            result = results[i]
            for j, r in enumerate(result):
                if not punct and (r['token'] in string.punctuation or r['head'] in string.punctuation):
                    num_punct += 1
                    continue
                if r['head_id'] == gold_heads[i][j]:
                    correct_heads += 1
                if r['deprel'] == gold_rels[i][j]:
                    correct_rels += 1
                if r['head'] == 'root' and r["id"] == gold_root[i]:
                    correct_root += 1
        total_heads = sum([len(h) for h in gold_heads])
        print(data_path)
        print("correct root: ", correct_root / len(results))
        print("UAS: ", correct_heads / (total_heads - num_punct))
        print("LAS: ", correct_rels / (total_heads - num_punct))
        print()

if __name__ == '__main__':
    #parser = StanfordParser(language='en', batch_size=10000)
    parser = StanfordParser(language='de', batch_size=10000)
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/en_test.conllu')
    '''
    parser.evaluate(data_path='adversarial_treebank/historic_spelling_ori_1971.conllu')
    parser.evaluate(data_path='adversarial_treebank/historic_spelling_attack_1971.conllu')

    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_ori_0.2_2000.conllu')
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.2_2000.conllu')
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.5_2000.conllu')
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.8_2000.conllu')
    '''
    import json
    import glob
    import pandas as pd

    #proj_dir = "/home/ychen/projects/syntactic_change/"
    #out_dir = "/home/ychen/projects/syntactic_change/data/hansard_final/parsed_v2/stanford/"
    out_dir = "../../data/deuparl_final/parsed_v3/stanford/"
    #out_dir = "../../data/hansard_final/parsed_v3/stanford/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Directory doesn't exist. Makedirs.")

    #for file in sorted(glob.glob("../../data/hansard_final/stanza_tokenized_v3/*.csv"),
    for file in tqdm(sorted(glob.glob("../../data/deuparl_final/stanza_tokenized_v3/*.csv"),
    #for file in tqdm(sorted(glob.glob("../../data/hansard_final/stanza_tokenized_v3/*.csv"),
                       reverse=False)):
        df = pd.read_csv(file, delimiter='\t')
        print(file)
        sentences = list(df['sent'])#[:1]
        #sentences = ['I have a dream.']
        results = parser.parse(sentences=sentences)
        print(results[0])
        #raise ValueError
        decade = file.split('/')[-1][:4]
        with open(os.path.join(out_dir, f"{decade}.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

