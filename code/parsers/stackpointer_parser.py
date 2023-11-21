import json
import os.path
import string

from basic_parser import *
from NeuroNLP2.neuronlp2.io import conllx_data, conllx_stacked_data, iterate_data
from NeuroNLP2.neuronlp2.models import StackPtrNet
from NeuroNLP2.neuronlp2 import utils
from NeuroNLP2.neuronlp2.io import CoNLLXWriter
from NeuroNLP2.neuronlp2.tasks import parser
from NeuroNLP2.neuronlp2.nn.utils import freeze_embedding
from NeuroNLP2.experiments.parsing import eval
import torch


class StackPointerParser(UniversalParser):
    def __init__(self, ckpt_dir='/home/ychen/projects/syntactic_change/code/parsers/NeuroNLP2_old/experiments/models/parsing/stackptr', batch_size=128, language='de'):
        super(StackPointerParser, self).__init__(ckpt_dir=ckpt_dir, parser_type='stackpointer', language=language, batch_size=batch_size)
        self.ckpt_dir = ckpt_dir
        self.parser, self.config = self.init_parser()
        self.discard = []

    def init_parser(self):
        config = {}
        model_path = self.ckpt_dir
        model_name = os.path.join(model_path, 'model.pt')
        punctuation = ['.', '``', "''", ':', ',']
        alphabet_path = os.path.join(model_path, 'alphabets')
        assert os.path.exists(alphabet_path)
        word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, None)
        config['word_alphabet'] = word_alphabet
        config['char_alphabet'] = char_alphabet
        config['pos_alphabet'] = pos_alphabet
        config['type_alphabet'] = type_alphabet
        config['beam'] = 10

        num_words = word_alphabet.size()
        num_chars = char_alphabet.size()
        num_pos = pos_alphabet.size()
        num_types = type_alphabet.size()

        result_path = os.path.join(model_path, 'tmp')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        punct_set = set(string.punctuation)
        if punctuation is not None:
            punct_set = set(punctuation).union(punct_set)
        config['punct_set'] = punct_set
        hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_type = hyps['model']
        assert model_type in ['DeepBiAffine', 'NeuroMST', 'StackPtr']
        word_dim = hyps['word_dim']
        char_dim = hyps['char_dim']
        use_pos = hyps['pos']
        pos_dim = hyps['pos_dim']
        mode = hyps['rnn_mode']
        hidden_size = hyps['hidden_size']
        arc_space = hyps['arc_space']
        type_space = hyps['type_space']
        p_in = hyps['p_in']
        p_out = hyps['p_out']
        p_rnn = hyps['p_rnn']
        activation = hyps['activation']
        alg = 'transition' if model_type == 'StackPtr' else 'graph'
        encoder_layers = hyps['encoder_layers']
        decoder_layers = hyps['decoder_layers']
        num_layers = (encoder_layers, decoder_layers)
        prior_order = hyps['prior_order']
        grandPar = hyps['grandPar']
        sibling = hyps['sibling']
        network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                              mode, hidden_size, encoder_layers, decoder_layers, num_types, arc_space, type_space,
                              prior_order=prior_order, activation=activation, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                              pos=use_pos, grandPar=grandPar, sibling=sibling)

        network = network.to(self.device)
        network.load_state_dict(torch.load(model_name, map_location=self.device))
        config['prior_order'] = prior_order
        config['alg'] = alg
        network.eval()
        return network, config

    def load_data(self, tokenized_sentences):
        s = ""
        for i, tokens in enumerate(tokenized_sentences):
            for j, token in enumerate(tokens):
                s += f"{j+1}\t{token}\t_\t_\t_\t_\t-1\t_\t_\t_\n"
           # line += "\n"
            s += '\n'
        data = conllx_stacked_data.read_data(s.strip(), self.config['word_alphabet'], self.config['char_alphabet'], self.config['pos_alphabet'], self.config['type_alphabet'],
                                             prior_order=self.config['prior_order'])
        return data

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

        # discard long texts that trigger memory issues
        self.discard = [i for i, tokens in enumerate(tokenized_sentences) if len(tokens) >= 500]
        print(f"\n{len(self.discard)} out of {len(sentences)} were discarded.\n")
        tokenized_sentences = [tokens for i, tokens in enumerate(tokenized_sentences) if i not in self.discard]
        sentences = [s for i, s in enumerate(sentences) if i not in self.discard]
        mw_parents = [mw for i, mw in enumerate(mw_parents) if i not in self.discard]
        loaded_data = self.load_data(tokenized_sentences)

        pbar = tqdm(total=len(sentences))
        sent_count = 0
        for data in iterate_data(loaded_data, self.batch_size):
            words = data['WORD'].to(self.device)
            chars = data['CHAR'].to(self.device)
            postags = data['POS'].to(self.device)
            lengths = data['LENGTH'].numpy()
            masks = data['MASK_ENC'].to(self.device)
            heads_pred, types_pred = self.parser.decode(words, chars, postags, mask=masks, beam=self.config['beam'], leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            words = words.cpu().numpy()

            start = 1
            end = 0
            bs = words.shape[0]
            for i in range(bs):
                r = []
                for j in range(start, lengths[i]-end):
                    w = self.config['word_alphabet'].get_instance(words[i, j])
                    if w == "<_UNK>" or w == "<UNK>":
                        w = tokenized_sentences[sent_count][j-1]
                    t = self.config['type_alphabet'].get_instance(types_pred[i, j])
                    h = heads_pred[i, j]
                    r.append({'tid': sent_count, 'id': j, 'token': w, 'head_id': h, 'head': self.config['word_alphabet'].get_instance(words[i, h]) if h > 0 else 'root',
                              'deprel': t, "pos": '_'})
                sent_count += 1
                results.append(r)
                pbar.update(1)

        pbar.close()
        assert len(results) == len(tokenized_sentences)
        if out == 'conllu':
            return self.convert_to_conull(results, tokenized_sentences, mw_parents)
        else:
            return results

    def evaluate(self, data_path, punct=False):
        data_dev = conllx_stacked_data.read_data(data_path, self.config['word_alphabet'], self.config['char_alphabet'], self.config['pos_alphabet'], self.config['type_alphabet'],
                                                 prior_order=self.config['prior_order'])

        pred_writer = CoNLLXWriter(self.config['word_alphabet'], self.config['char_alphabet'], self.config['pos_alphabet'], self.config['type_alphabet'])
        gold_writer = CoNLLXWriter(self.config['word_alphabet'], self.config['char_alphabet'], self.config['pos_alphabet'], self.config['type_alphabet'])

        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        pred_filename = os.path.join('tmp', 'pred')
        pred_writer.start(pred_filename)
        gold_filename = os.path.join('tmp', 'gold')
        gold_writer.start(gold_filename)

        test_stats, test_stats_nopunct, test_stats_root = eval(self.config['alg'], data_dev, self.parser, pred_writer, gold_writer, self.config['punct_set'],
                                                            self.config['word_alphabet'], self.config['pos_alphabet'], self.device, beam=self.config['beam'], batch_size=self.batch_size)

        test_ucorrect, test_lcorrect, test_ucomlpete, test_lcomplete, test_total = test_stats
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_ucomlpete_nopunc, test_lcomplete_nopunc, test_total_nopunc = test_stats_nopunct
        test_root_correct, test_total_root, test_total_inst = test_stats_root

        pred_writer.close()
        gold_writer.close()

        if punct:
            uas = test_ucorrect / test_total
            las = test_lcorrect / test_total


            print(f'\nWith punct:\n UAS --- {uas}; LAS --- {las}')
        else:
            uas_wo = test_ucorrect_nopunc / test_total_nopunc
            las_wo = test_lcorrect_nopunc / test_total_nopunc

            #print(f'\nW/o punct:\n UAS --- {uas_wo}; LAS --- {las_wo}')

            root_correct = test_root_correct / test_total_root
            #print(f"\nroot correct: {root_correct}")

            print(data_path)
            print("correct root: ", root_correct)
            print("UAS: ", uas_wo)
            print("LAS: ", las_wo)
            print()

        #return uas, las, uas_wo, uas_wo, root_correct


if __name__ == '__main__':

    DATA = 'deuparl'
    ckpt = 'NeuroNLP2/models/parsing/stackptr/v2.12_merged_de/' if DATA == 'deuparl' else 'NeuroNLP2/models/parsing/stackptr/v2.12_merged_en'
    #parser = StackPointerParser(ckpt_dir='/home/ychen/projects/syntactic_change/code/parsers/NeuroNLP2/models/parsing/stackptr/v2.12_merged_en', language='en', batch_size=32)
    parser = StackPointerParser(ckpt_dir=ckpt, language='de', batch_size=80)
    #parser.evaluate(data_path='/home/ychen/projects/syntactic_change/data/parsers/merged_v2.12/de_test.conllu')

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

    #out_dir = "/home/ychen/projects/syntactic_change/data/hansard_final/parsed_v4/stackpointer/"
    out_dir = f"../../data/{DATA}_final/parsed_v4/stackpointer/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Directory doesn't exist. Makedirs.")

    from tqdm import tqdm
    for file in tqdm(sorted(glob.glob(f"../../data/{DATA}_final/stanza_tokenized_v4/*.csv"),
                       reverse=False)):
        decade = file.split('/')[-1][:4]
        if int(decade) == 1860:
            continue
        df = pd.read_csv(file, delimiter='\t')#[:10000]
        print(file)
        #sentences = list(df['sent'])
        tokens, tags = list(df['tokenized']), list(df['pos'])
        #results = parser.parse(sentences=sentences)
        results = parser.parse(sentences=(tokens, tags))


        #with open(os.path.join(out_dir, f"{decade}.json"), 'w', encoding='utf-8') as f:
        #    json.dump(results, f, indent=2)

        with open(os.path.join(out_dir, f"{decade}.conllu"), 'w', encoding='utf8') as f:
            f.write(results)
