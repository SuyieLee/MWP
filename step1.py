import argparse
import os

import torch
from treemodel import EncoderRNN, Prediction, GenerateNode, Merge
from Seq2Tree import Seq2Tree
from Trainer import Trainer
from dataloader import DataLoader
# from argparse import ArgumentParser
import torch.nn as nn
import random
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='恢复')
    parser.add_argument('--teacher_forcing_ratio', type=float, dest='teacher_forcing_ratio', default=0.83)
    parser.add_argument('--teacher_forcing', type=bool, dest='teacher_forcing', default=True)
    parser.add_argument('--data_name', type=str, dest='data_name', default='train_23k')
    parser.add_argument('--hidden_size', type=int, dest='hidden_size', default=512)
    # parser.add_argument('--decoder_hidden_size', type=int, dest='decoder_hidden_size', default=1024)
    parser.add_argument('--input_dropout', type=float, dest='input_dropout', default=0.5)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)
    parser.add_argument('--layers', type=int, dest='layers', default=2)
    parser.add_argument('--cuda-id', type=str, dest='cuda_id', default='1')
    parser.add_argument('--cuda_use', type=bool, dest='cuda_use', default=False)
    parser.add_argument('--checkpoint_dir_name', type=str, dest='checkpoint_dir_name', default="0000-0000", help='模型存储名字')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64)
    parser.add_argument('--epoch_num', type=int, dest='epoch_num', default=3)
    parser.add_argument('--bidirectional', type=bool, dest='bidirectional', default=True)
    parser.add_argument('--print_every', type=int, dest='print_every', default=10)
    parser.add_argument('--valid_every', type=int, dest='valid_every', default=2)
    parser.add_argument('--train_word2vec', type=bool, dest='train_word2vec', default=False)
    parser.add_argument('--all_vec', type=bool, dest='all_vec', default=False)
    parser.add_argument('--start_epoch', type=int, dest='start_epoch', default=0)
    parser.add_argument('--embedding_size', type=int, dest='embedding_size', default=128)
    parser.add_argument('--weight_decay', type=float, dest='weight_decay', default=1e-5)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-3)
    return parser.parse_args()


def step_one_train():
    if args.cuda_use:
        print("----------Using cuda----------")
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
    # pad = data_loader.decode_classes_dict['PAD_token']
    # loss = NLLLoss(weight, pad)
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
    print("------------开始训练-------------")
    start_epoch = args.start_epoch
    if start_epoch > 0:
        lists = os.listdir('./model/')
        lists.sort(key=lambda x: os.path.getmtime(('./model/' + x)))  # 获取最新产生的模型
        file_last = os.path.join('./model/', lists[-1])
        model.load_state_dict(torch.load(file_last))
    path = trainer.train(model, epoch_num=args.epoch_num, start_epoch=start_epoch, valid_every=args.valid_every)
    # print("------------开始测试-------------")
    # model.load_state_dict(torch.load(path))
    # test_ans_acc = trainer.evaluate(model, data_loader.test_data)
    # print("Test Acc: %.2f  Acc: %d / %d" % (100*test_ans_acc/len(data_loader.test_data), test_ans_acc, len(data_loader.test_data)))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(3407)
    args = getArgs()
    print(args)
    step_one_train()
