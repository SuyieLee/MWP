# MWP问题
这个项目的想要实现seq2seq的MWP问题求解。参照Math_EN和RTNN复现。
##参考论文
Translating a Math Word Problem to an Expression Tree. EMNLP 2018

Template-Based Math Word Problem Solvers with Recursive Neural Networks. AAAI 2019

##参数列表
``` 
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='恢复')
    parser.add_argument('--teacher_forcing_ratio', type=float, dest='teacher_forcing_ratio', default=0.83)
    parser.add_argument('--teacher_forcing', type=bool, dest='teacher_forcing', default=True)
    parser.add_argument('--data_name', type=str, dest='data_name', default='train_23k')
    parser.add_argument('--encoder_hidden_size', type=int, dest='encoder_hidden_size', default=512)
    parser.add_argument('--decoder_hidden_size', type=int, dest='decoder_hidden_size', default=1024)
    parser.add_argument('--input_dropout', type=float, dest='input_dropout', default=0.5)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)
    parser.add_argument('--layers', type=int, dest='layers', default=2)
    parser.add_argument('--cuda-id', type=str, dest='cuda_id', default='1')
    parser.add_argument('--cuda_use', type=bool, dest='cuda_use', default=False)
    parser.add_argument('--checkpoint_dir_name', type=str, dest='checkpoint_dir_name', default="0000-0000", help='模型存储名字')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64)
    parser.add_argument('--epoch_num', type=int, dest='epoch_num', default=10)
    parser.add_argument('--bidirectional', type=bool, dest='bidirectional', default=True)
    parser.add_argument('--print_every', type=int, dest='print_every', default=10)
    parser.add_argument('--valid_every', type=int, dest='valid_every', default=2)
    parser.add_argument('--train_word2vec', type=bool, dest='train_word2vec', default=False)
    parser.add_argument('--all_vec', type=bool, dest='all_vec', default=False)
    parser.add_argument('--start_epoch', type=int, dest='start_epoch', default=0)```
```
##分支说明
main是使用已处理好的数据进行训练的，效果最好。

mydata是自己写的数据预处理过程，没有使用SNI，存在一些情况没有很好的处理，效果稍逊。

##运行
运行前建议先检查一下参数。
```
    bash run.sh
```