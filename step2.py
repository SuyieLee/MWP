import os
from Trainer import Trainer
import argparse
from step1 import getArgs
import torch
from model import EncoderRNN, DecoderRNN, Seq2Seq
from Trainer import Trainer
from dataloader import DataLoader
# from argparse import ArgumentParser
import torch.nn as nn

args = getArgs()
data_loader = DataLoader(args)
embed_model = nn.Embedding(data_loader.vocab_len, 128)
# embed_model.weight初始化是正态分布N(0,1)
# embed_model.weight.data.copy_(torch.from_numpy(data_loader.word2vec.embedding_vec))
encode_model = EncoderRNN(vocab_size=data_loader.vocab_len,
                            embed_model=embed_model,
                            embed_size=128,
                            hidden_size=args.encoder_hidden_size,
                            input_dropout=args.input_dropout,
                            dropout=args.dropout,
                            layers=int(args.layers),
                            bidirectional=args.bidirectional)

decoder_model = DecoderRNN(vocab_size=data_loader.vocab_len,
                            embed_model=embed_model,
                            hidden_size=args.decoder_hidden_size,
                            embed_size=128,
                            classes_size=data_loader.classes_len,
                            input_dropout=args.input_dropout,
                            dropout=args.dropout,
                            layers=int(args.layers),
                            sos_id=data_loader.vocab_dict['END_token'],
                            eos_id=data_loader.vocab_dict['END_token'],
                            bidirectional=args.bidirectional)
model = Seq2Seq(encode_model, decoder_model)

if args.cuda_use:
    model = model.cuda()

weight = torch.ones(data_loader.classes_len)
if args.cuda_use:
    weight = weight.cuda()

trainer = Trainer(model,
                      # loss=loss,
                      weight=weight,
                      vocab_dict=data_loader.vocab_dict,
                      vocab_list=data_loader.vocab_list,
                      data_loader=data_loader,
                      batch_size=args.batch_size,
                      decode_classes_dict=data_loader.decode_classes_dict,
                      decode_classes_list=data_loader.decode_classes_list,
                      cuda_use=args.cuda_use,
                      TRG_PAD_IDX=data_loader.decode_classes_dict['PAD_token'],
                      print_every=args.print_every,
                      checkpoint_dir_name=args.checkpoint_dir_name
                      )

lists = os.listdir('./model/')
lists.sort(key=lambda x: os.path.getmtime(('./model/' + x)))  # 获取最新产生的模型
file_last = os.path.join('./model/', lists[-1])
model.load_state_dict(torch.load(file_last))
print("------------开始测试-------------")
# model.load_state_dict(torch.load(path))s
test_ans_acc = trainer.evaluate(model, data_loader.test_data)
print("Test Acc: %.2f  Acc: %d / %d" % (100*test_ans_acc/len(data_loader.test_data), test_ans_acc, len(data_loader.test_data)))
