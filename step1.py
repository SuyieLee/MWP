import argparse
import os

import torch
from model import EncoderRNN, DecoderRNN, Seq2Seq
from Trainer import Trainer
from dataloader import DataLoader
from loss import NLLLoss
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
    parser.add_argument('--encoder_hidden_size', type=int, dest='encoder_hidden_size', default=512)
    parser.add_argument('--decoder_hidden_size', type=int, dest='decoder_hidden_size', default=1024)
    parser.add_argument('--input_dropout', type=float, dest='input_dropout', default=0.4)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)
    parser.add_argument('--layers', type=int, dest='layers', default=2)
    parser.add_argument('--cuda-id', type=str, dest='cuda_id', default='1')
    parser.add_argument('--cuda_use', type=bool, dest='cuda_use', default=False)
    parser.add_argument('--checkpoint_dir_name', type=str, dest='checkpoint_dir_name', default="0000-0000", help='模型存储名字')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64)
    parser.add_argument('--epoch_num', type=int, dest='epoch_num', default=1)
    parser.add_argument('--bidirectional', type=bool, dest='bidirectional', default=True)
    parser.add_argument('--print_every', type=int, dest='print_every', default=10)
    parser.add_argument('--valid_every', type=int, dest='valid_every', default=2)
    return parser.parse_args()


def step_one_train():
    if args.cuda_use:
        print("----------Using cuda----------")
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
                      TRG_PAD_IDX=data_loader.decode_classes_dict['PAD_token'],
                      print_every=args.print_every,
                      checkpoint_dir_name=args.checkpoint_dir_name
                      )
    print("------------开始训练-------------")
    trainer.train(model, epoch_num=args.epoch_num, resume=args.resume, valid_every=args.valid_every)
    print("------------开始测试-------------")
    test_ans_acc = trainer.evaluate(model, data_loader.test_data)
    print("Test Acc: %.2f  Acc: %d / %d" % (100*test_ans_acc/len(data_loader.test_data), test_ans_acc, len(data_loader.test_data)))


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
    step_one_train()
