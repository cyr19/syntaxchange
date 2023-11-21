import re
import string
from basic_parser import *
from towerparse.tower import TowerParser
import stanza

class TowParser(UniversalParser):
    def __init__(self, batch_size=4, language='de', ckpt_dir=None):
        super(TowParser, self).__init__(parser_type='tower', batch_size=batch_size, language=language, ckpt_dir=ckpt_dir)
        print(f'device: {self.device}')
        #self.parser, self.tokenizer = self.init_parser()
        self.parser = self.init_parser()
        self.discard = []

    def init_parser(self):
        if self.ckpt_dir:
            path = f"towerparse/checkpoints/UD_{'German' if self.language == 'de' else 'English'}-{self.ckpt_dir}"
        else:
            language2ckpt_dir = {
                #'de': "towerparse/checkpoints/UD_German-GSD",
                'de': "towerparse/checkpoints/UD_German-HDT",
                #'en': "towerparse/checkpoints/UD_English-PUD"
                'en': "towerparse/checkpoints/UD_English-EWT"
            }
            path = language2ckpt_dir[self.language]
        parser = TowerParser(path, device=self.device)
        stanza.download(self.language)
        #tokenizer = stanza.Pipeline(self.language, processors='tokenize,pos')
        return parser#, tokenizer

    def parse(self, sentences, out='conllu', tokenized=True, mw_parents=[]):
        results = []
        if not tokenized:
            print('not tokenized...')
            print(sentences[0])
            tokenized_sentences, mw_parents = self.tokenize(sentences)
        else:
            print('tokenized...')
            print(sentences[0])
            tokenized_sentences = [s.split() for s in sentences]

        print(tokenized_sentences[0])

        parsed, discarded = self.parser.parse(lang=self.language, sentences=tokenized_sentences, batch_size=self.batch_size)
        self.discard = discarded
        assert len(parsed) == len(tokenized_sentences) - len(discarded), f"{len(parsed)} -- {len(tokenized_sentences)}\n{tokenized_sentences}\n{parsed}"

        tokenized_sentences = [s for i, s in enumerate(tokenized_sentences) if i not in discarded]
        mw_parents = [mw for i, mw in enumerate(mw_parents) if i not in self.discard]
        sentences = [s for i, s in enumerate(sentences) if i not in self.discard]

        #s = ''
        #tid = 1
        for i, p in enumerate(parsed):
            r = []
            #s += f"# text_id = {tid}\n# sent_id = {i}\n# sent = {' '.join(tokenized_sents[i])}\n"
            tokens = tokenized_sentences[i]
            #s += f"# text_id = {i}\n# text = {' '.join(tokens)}\n"
            #tokens = remaining_sents[i].split()
            #print(p)
            for j, t in enumerate(p):
                if t[2] is None or t[-1] is None:
                    raise ValueError
                #print(t)
                #raise ValueError
                r.append({'tid': i, 'id': t[0], "token": tokens[j], "head_id": t[2],
                          "head": tokens[t[2]-1] if t[2] > 0 else "root",
                          "deprel": t[-1], "pos": '_'})

                #s += f"{t[0]}\t{t[1]}\t_\t_\t_\t_\t{t[2]}\t{t[-1]}\t_\t_\n"

                #s += f"{t[0]}\t{tokens[j]}\t_\t_\t_\t_\t{t[2]}\t{t[-1]}\t_\t_\n"

            results.append(r)

        if out == 'conllu':
            print(results[-1])
            return self.convert_to_conull(results, sentences, mw_parents)
        else:
            return results

    def load_data(self, sentences):
        tokenized_sents = []
        postags = []
        for sent in sentences:
            doc = self.tokenizer(sent)
            tokens, tags = [], []
            for sentence in doc.sentences:
                for word in sentence.words:
                    tokens.append(word.text)
                    tags.append(word.upos)
            tokenized_sents.append(tokens)
            postags.append(tags)
        assert len(tokenized_sents) == len(sentences), f"{len(tokenized_sents)} -- {len(sentences)}"
        return tokenized_sents, postags

    def evaluate(self, data_path, punct=False):
        print('\nnhere!!!!\n')
        with open(data_path, 'r') as f:
            text = f.read()
        instances = re.split('\n{2,}', string=text)#[:10]
        toknized_sents, postags = [], []
        gold_heads, gold_rels = [], []
        gold_root = []
        for i, instance in enumerate(instances):
            toknized_sent, postag = [], []
            gold_h, gold_r = [], []
            root = []
            for l in instance.strip().split('\n'):
                if l.startswith("#"):
                    continue
                tks = l.split('\t')
                if not re.match("^\d+$", string=tks[0]):
                    continue
                #if not punct and tks[1] in string.punctuation:
                #    continue
                toknized_sent.append(tks[1])
                postag.append(tks[3])
                gold_h.append(int(tks[6]))
                gold_r.append(tks[7])
                if tks[6] == '0':
                    #gold_root.append()
                    root.append(int(tks[0]))
            if len(root) != 1:
                continue
            gold_root.append(root[0])
            toknized_sents.append(toknized_sent)
            postags.append(postag)
            gold_heads.append(gold_h)
            gold_rels.append(gold_r)

        results = self.parse((toknized_sents, postags))
        #print(results)
        print('\n...................!!!!\n')
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
        print("correct root: ", correct_root/len(results))
        print("UAS: ", correct_heads/(total_heads-num_punct))
        print("LAS: ", correct_rels /(total_heads-num_punct))
        print("complete!!!")



if __name__ == '__main__':
   # parser = TowParser(batch_size=32)
   # parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/de_test.conllu')
    args_parser = ArgumentParser()
    args_parser.add_argument("--data", '-d', type=str, required=True)
    args_parser.add_argument("--start", '-s', type=int, default=1860)
    args_parser.add_argument("--end", "-e", type=int, default=2020)
    args_parser.add_argument("--batch_size", '-b', type=int, default=4)
    args = args_parser.parse_args()
    DATA = args.data

    lang = 'de' if DATA == 'deuparl' else "en"
    parser = TowParser(batch_size=4, language=lang)
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/de_test_reform.conllu')
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/ud-treebanks-v2.12/UD_English-EWT/en_ewt-ud-test.conllu')
    #parser.evaluate(data_path='adversarial_treebank/historic_spelling_ori_1971.conllu')
    #parser.evaluate(data_path='adversarial_treebank/historic_spelling_attack_1971.conllu')


    #parser.evaluate(data_path='adversarial_treebank/ocr_spelling_ori_0.2_2000.conllu')
    #parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.2_2000.conllu')
    #parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.5_2000.conllu')
    #parser.evaluate(data_path='adversarial_treebank/ocr_spelling_attack_0.8_2000.conllu')
    import json
    import glob
    import pandas as pd
    
    out_dir = f"../../data/{DATA}_final/parsed_v4/towerparse/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Directory doesn't exist. Makedirs.")
    
    for file in sorted(glob.glob(f"../../data/{DATA}_final/stanza_tokenized_v4/*.csv"), reverse=False):
        decade = file.split('/')[-1][:4]
        print(decade)
        df = pd.read_csv(file, delimiter='\t')[:5]
        sentences = list(df['sent'])
        results = parser.parse(sentences=sentences)
        print(results)
        raise ValueError

        #with open(os.path.join(out_dir, f"{decade}.json"), 'w', encoding='utf-8') as f:
        #   json.dump(results, f, indent=2)

        with open(os.path.join(out_dir, f"{decade}.conllu"), 'w', encoding='utf8') as f:
            f.write(results)