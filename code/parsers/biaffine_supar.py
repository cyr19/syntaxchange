import os

from basic_parser import *
from supar import Parser


class SuparBiaffineParser(UniversalParser):
    def __init__(self, ckpt_dir="dep-biaffine-xlmr", batch_size=128, language='en'):
        super(SuparBiaffineParser, self).__init__(ckpt_dir=ckpt_dir, parser_type='biaffine_supar', batch_size=batch_size, language=language)
        self.parser = self.init_parser()

    def init_parser(self):
        parser = Parser.load(self.ckpt_dir)
        return parser

    def parse(self, sentences, out='conllu', tokenized=True, mw_parents=[]):
        results = []
        if tokenized:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = [s.split() for s in sentences]
        else:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)

        self.discard = [i for i, tokens in enumerate(tokenized_sentences) if len(tokens) >= 200]
        print(f"\n{len(self.discard)} out of {len(sentences)} were discarded.\n")
        tokenized_sentences = [tokens for i, tokens in enumerate(tokenized_sentences) if i not in self.discard]
        sentences = [s for i, s in enumerate(sentences) if i not in self.discard]
        mw_parents = [mw for i, mw in enumerate(mw_parents) if i not in self.discard]

        parsed = self.parser.predict(tokenized_sentences, batch_size=self.batch_size, prob=False, verbose=True)

        assert len(parsed) == len(tokenized_sentences)
        for j, p in enumerate(parsed):
            r = []
            heads = p.arcs
            tokens = p.texts
            deprel = p.rels
            for i in range(len(heads)):
                tmp_r = {'tid': j, 'id': i+1, "token": tokens[i], "head_id": heads[i], "head": tokens[heads[i]-1] if heads[i] > 0 else "root", "deprel": deprel[i], "pos": "_"}
                r.append(tmp_r)
            results.append(r)

        assert len(results) == len(tokenized_sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results

    def evaluate(self, data_path):
        self.parser.evaluate(data_path, verbose=True)


if __name__ == '__main__':
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="biaffine-dep-en")
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="biaffine-dep-xlmr")
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="dep-biaffine-xlmr")
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="biaffine-xlmr-dep")
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="dep-xlmr-biaffine")
    #parser = SuparBiaffineParser(batch_size=256, language='en', ckpt_dir="xlmr-dep-biaffine")

    '''
    import glob
    import re
    files = glob.glob("/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/*.conllu")
    for file in files:
        print(file)
        with open(file, 'r') as f:
            text = f.read()
    #
        text = re.sub(pattern=r"\n{2,}", string=text, repl="\n\n")

        with open(file.replace(".conllu", "_reform.conllu"), 'w', encoding='utf8') as f:
            f.write(text)
    '''
    #import torch
    #print(torch.cuda.is_available())
    #parser = SuparBiaffineParser(batch_size=256, language='de', ckpt_dir="dep-biaffine-xlmr")
    #parser = SuparBiaffineParser(batch_size=1, language='de', ckpt_dir="/home/ychen/projects/checkpoints/supar_crf2o/v2.12_merged_en")
    parser = SuparBiaffineParser(batch_size=1, language='en', ckpt_dir="/home/ychen/projects/checkpoints/supar_biaffine/v2.12_merged_en")
    #parser.parser.predict('I saw Sarah with a telescope.', lang='en')
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/en_test_reform.conllu')
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/en_test_reform.conllu')
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/ud2.2/UD_German-GSD/de_gsd-ud-test.conllu')

    '''
    parser.evaluate(data_path='adversarial_treebank/historic_spelling_ori_1971.conllu')
    
    print("\n\n\n\n")
    parser.evaluate(data_path='adversarial_treebank/historic_spelling_attack_1971.conllu')
    print("\n\n\n\n")
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_ori_0.2_2000.conllu')
    print("\n\n\n\n")
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.2_2000.conllu')
    print("\n\n\n\n")
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.5_2000.conllu')
    print("\n\n\n\n")
    parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.8_2000.conllu')
    print("\n\n\n\n")
    '''

    import json
    import glob
    import pandas as pd
    out_dir = "/home/ychen/projects/syntactic_change/data/hansard_final/parsed_v2/biaffine/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Directory doesn't exist. Makedirs.")

    for file in sorted(glob.glob("/home/ychen/projects/syntactic_change/data/hansard_final/stanza_tokenized_v2/*.csv"),
                       reverse=False):
        df = pd.read_csv(file, delimiter='\t')
        print(file)
        sentences = list(df['sent'])
        results = parser.parse(sentences=sentences)
        decade = file.split('/')[-1][:4]
        with open(os.path.join(out_dir, f"{decade}.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
