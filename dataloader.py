import json
import math
import os

import numpy as np
from gensim.models import word2vec
from data_tools import *


class DataMath23k():
    def __init__(self):
        filename = "./data/math23k_0108.json"
        self.data_dict = read_json_data(filename)


class Word2Vec():
    def __init__(self, data, flag=False):
        self.data = data
        if flag:
            self.embedding_vec = np.load("./data/new_emb.npy")
        else:
            self.train_word2vec()

    def train_word2vec(self):
        """
        训练word2vec模型，将所有的词和几个特殊符号编码，得到embedding向量
        :return:
        """
        new_data = {}
        sentences = []
        for k,v in self.data.data_dict.items():
            sentence = v['text'].strip().split(' ')
            sentences.append(sentence)
            for elem in sentence:
                new_data[elem] = new_data.get(elem, 0) + 1
                # new_data.get(elem, 0)字典key不存在时返回默认值0

        model = word2vec.Word2Vec(sentences, vector_size=128, min_count=1) # 训练
        token = ['PAD_token', 'SOS_token', 'END_token', 'UNK_token']
        emb_vectors = []

        # 这是给上面这几个token的embedding向量
        emb_vectors.append(np.zeros((128)))
        emb_vectors.append(np.random.rand((128)) / 1000.0)
        emb_vectors.append(np.random.rand((128)) / 1000.0)
        emb_vectors.append(np.random.rand((128)) / 1000.0)

        op_list = [u'+', u'*', u'-', u'/', u'1', u'PI', u'temp_m', u'temp_l', u'temp_o', u'temp_n', u'temp_i', u'temp_h', u'temp_k', u'temp_j', u'temp_e', u'temp_d', u'temp_g', u'temp_f', u'temp_a', u'temp_c', u'temp_b', u'^']
        for k,v in new_data.items():
            token.append(k)
            # model.wv[k]获取对应的字符的embedding
            emb_vectors.append(np.array(model.wv[k]))  # np.array(model.wv[k]) = model.wv[k]
        for elem in op_list:
            if elem in token:
                continue
            else:
                token.append(elem)
                emb_vectors.append(np.random.rand((128))/1000.0)
        emb_vectors = np.array(emb_vectors)
        print("----------完成词表生成----------")

        emb_filename = "./data/new_emb.npy"
        token_filename = "./data/new_token_list.json"
        np.save(emb_filename, emb_vectors)
        with open(token_filename, 'w') as f:
            json.dump(token, f)
        self.embedding_vec = emb_vectors
        print("---------词表保存成功-----------")


class DataLoader():
    def __init__(self, args=None):
        self.args = args
        self.data_23k = DataMath23k()
        print("---------math23k数据加载完成---------")
        self.word2vec = Word2Vec(self.data_23k, True) # True则直接加载
        self.vocab_list = read_json_data("./data/new_token_list.json")
        vocab_dict = {}
        for idx, elem in enumerate(self.vocab_list):
            vocab_dict[elem] = idx
        self.vocab_dict = vocab_dict
        self.vocab_len = len(self.vocab_list)

        self.decode_classes_list = ['PAD_token', 'SOS_token', 'END_token'] + [u'^', u'1', u'PI', u'temp_m', u'temp_l', u'temp_o',
                                                                 u'temp_n', u'temp_i', u'temp_h', u'temp_k', u'temp_j',
                                                                 u'temp_e', u'temp_d', u'temp_g', u'temp_f', u'temp_a',
                                                                 u'temp_c', u'temp_b', u'/', u'-', u'+', u'*', u'^']
        # 解码器输出的词表
        self.decode_classes_dict = {}
        for idx, elem in enumerate(self.decode_classes_list):
            self.decode_classes_dict[elem]=idx
        self.classes_len = len(self.decode_classes_list)

        self.train_data, self.valid_data, self.test_data = split_data(self.data_23k.data_dict)
        self.templates = read_json_data("./data/norm_templates.json")

    def data_batch_process(self, data):
        batch_encode_idx = [] #
        batch_decode_idx = []
        batch_encode_len = []
        batch_decode_len = []

        batch_idxs = [] # 第几条数据，id
        batch_text = [] # 文本内容
        batch_num_list = [] # 数字列表
        batch_ans = [] # 答案
        batch_template = []

        # 获取每个batch的原数据
        for elem in data:
            idx = elem[0]
            text = elem[1]['text']
            batch_text.append(text)
            text_idx = string_2_idx_sen(text.split(' '), self.vocab_dict)
            text_idx = [self.decode_classes_dict['SOS_token']] + text_idx + [self.decode_classes_dict['END_token']]
            batch_encode_idx.append(text_idx)
            batch_encode_len.append(len(text_idx))

            target = ['SOS_token']
            # target = elem[1]['target_template']
            target += self.templates[idx]
            # target.append('END_token')
            batch_template.append(target)
            target_idx = string_2_idx_sen(target, self.decode_classes_dict)
            batch_decode_idx.append(target_idx)
            batch_decode_len.append(len(target_idx))

            batch_idxs.append(idx)
            batch_num_list.append(elem[1]['num_list'])
            batch_ans.append(elem[1]['ans'])

        # 求这个batch里面最长的，后面做paddinh对齐
        max_encoder_len = max(batch_encode_len)
        max_decoder_len = max(batch_decode_len)
        batch_encode_pad_idx, batch_decode_pad_idx = [], []

        for i in range(len(data)):
            encode_sen_idx = batch_encode_idx[i]
            encode_sen_pad_idx = pad_sen(encode_sen_idx, max_encoder_len, self.vocab_dict['PAD_token'])
            batch_encode_pad_idx.append(encode_sen_pad_idx)

            decode_sen_idx = batch_decode_idx[i]
            decode_sen_pad_idx = pad_sen(decode_sen_idx, max_decoder_len, self.decode_classes_dict['PAD_token'])
            batch_decode_pad_idx.append(decode_sen_pad_idx)

        batch_data_dict = {}
        batch_data_dict['batch_encode_pad_idx'] = batch_encode_pad_idx
        # batch_data_dict['batch_text'] = batch_text
        batch_data_dict['batch_encode_idx'] = batch_encode_idx
        batch_data_dict['batch_encode_len'] = batch_encode_len

        batch_data_dict['batch_index'] = batch_idxs
        batch_data_dict['batch_decode_idx'] = batch_decode_idx
        batch_data_dict['batch_decode_len'] = batch_decode_len
        batch_data_dict['batch_decode_pad_idx'] = batch_decode_pad_idx
        batch_data_dict['batch_template'] = batch_template

        batch_data_dict['batch_num_list'] = batch_num_list
        batch_data_dict['batch_ans'] = batch_ans

        # if len(data) != 1:
        #     batch_data_dict = self.sorted_data(batch_data_dict)

        return batch_data_dict

    def yield_batch(self, data, batch_size):
        """
        生成一个batch的数据
        """
        step = math.ceil(len(data)/batch_size)
        for i in range(step):
            batch_start = i*batch_size
            batch_end = min(i*batch_size+batch_size, len(data))
            batch_data_dict = self.data_batch_process(data[batch_start:batch_end])
            yield batch_data_dict

    def sorted_data(self, data):
        new_data = {}
        batch_len = np.array(data['batch_encode_len'])
        sort_idx = np.argsort(-batch_len)
        for key, value in data.items():
            new_data[key] = np.array(value)[sort_idx]
        return new_data

