import os
from Trainer import Trainer
import argparse
from step1 import getArgs
import torch
from treemodel import EncoderRNN, Prediction, GenerateNode, Merge
from Seq2Tree import Seq2Tree
from Trainer import Trainer
from dataloader import DataLoader
# from argparse import ArgumentParser
import torch.nn as nn

args = getArgs()
data_loader = DataLoader(args)
embed_model = nn.Embedding(data_loader.vocab_len, args.embedding_size)
# embed_model.weight初始化是正态分布N(0,1)
# embed_model.weight.data.copy_(torch.from_numpy(data_loader.word2vec.embedding_vec))
encoder = EncoderRNN(vocab_size=data_loader.vocab_len,
                            embed_model=embed_model,
                            embed_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            input_dropout=args.input_dropout,
                            dropout=args.dropout,
                            layers=int(args.layers),
                            bidirectional=args.bidirectional)

predict = Prediction(args.hidden_size, op_nums=5, input_size=len(data_loader.generate_op))
generate = GenerateNode(args.hidden_size, op_nums=5, embedding_size=args.embedding_size)
merge = Merge(args.hidden_size, args.embedding_size)

if args.cuda_use:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

model = Seq2Tree(data_loader, encoder, predict, generate, merge, args.learning_rate, args.weight_decay, args.cuda_use)

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