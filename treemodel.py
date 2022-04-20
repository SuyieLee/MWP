import copy
import torch.nn as nn
import torch
import random
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_model=None, embed_size=100, hidden_size=128, input_dropout=0, dropout=0,
                 layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.dropout = dropout
        self.layers = layers
        self.bidirectional = bidirectional
        if embed_model is None:
            self.embed_model = nn.Embedding(self.vocab_size, self.embed_size)
        else:
            self.embed_model = embed_model
        # 使用batch_first=True，可以使lstm接受维度为(batchsize，序列长度，输入维数)的数据输入，同时，lstm的输出数据维度也会变为batchsize放在第一维
        # 如果使用单层的话，就不要用dropout=self.dropout
        if self.layers == 1:
            self.rnn = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.layers,
                              bidirectional=self.bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.layers,
                              bidirectional=self.bidirectional, dropout=self.dropout)

    def forward(self, input, input_len, hidden=None):
        # embed = [input len, batch size]
        # output = [input len, batch size, hidden dim * directions)
        # hidden = [layers * directions, batch size, hidden dim]
        # cell = [layers * directions, batch size, hidden dim]
        # input_len是每一个向量没有padding前的原始长度，需要做packed用的
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs只包含最后一层网络的所有hidden
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        embed = self.embed_model(input)
        embed = self.input_dropout(embed)
        # packed = nn.utils.rnn.pad_packed_sequence(embed, input_len)
        pade_output, pade_hidden = self.rnn(embed, hidden)
        # pade_output, _ = nn.utils.rnn.pad_packed_sequence(pade_output)
        encoder_output = pade_output[:, :, :self.hidden_size] + pade_output[:, :, self.hidden_size:]  # 前向+后向
        problem_output = pade_output[0, :, self.hidden_size:] + pade_output[-1, :, :self.hidden_size]  # 后向0+前向n
        return encoder_output, problem_output


class Prediction(nn.Module):
    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.op_nums = op_nums
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.embedding_w = nn.Parameter(torch.randn(1, input_size, hidden_size))  # 需要对解码器的词汇做embedding

        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, self.op_nums)
        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        """
        current_node就是目标向量
        """
        current_embedding = []
        for st in node_stacks:
            if len(st) == 0:  # 如果树为空，以padding_hidden作为根结点的embedding
                current_embedding.append(padding_hidden)
            else:  # 如果树不空，取最后一个结点的embedding作为当前需要处理的embedding
                current_node = st[-1]
                current_embedding.append(current_node.embedding)

        # 生成左右子树的目标向量，l是自底向上编码的tl吗？
        current_node_temp = []
        for l, c in zip(left_childs, current_embedding):
            if l is None:  # 如果这个左子树是空的话，计算左子树
                c = self.dropout(c)  # Cl
                t = torch.tanh(self.concat_l(c))  # 对应论文中的公式 Qle
                g = torch.sigmoid(self.concat_lg(c))  # Gl
                current_node_temp.append(g * t)  # Gl*Qle
            else:  # 左子树存在，需要计算右子树
                ld = self.dropout(l)
                c = self.dropout(c)
                t = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                g = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append((g * t))

        current_node = torch.stack(current_node_temp)  # torch.stack拼接多个张量，直接放到一起，后面一步操作处理多个，最后再分开
        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)  # a_s
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # c

        # embedding_weight = [1, input size, hidden size]
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_w.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_w.repeat(*repeat_dims)  # embedding_weight = [batch size, intput size=2, hidden size]
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)

        embedding_weights = self.dropout(embedding_weight)
        s_input = torch.cat((current_node, current_context), 2)
        s_input = s_input.squeeze(1)
        s_input = self.dropout(s_input)

        final_score = self.score(s_input.unsqueeze(1), embedding_weights, mask_nums)

        ops = self.ops(s_input)

        return final_score, ops, current_node, current_context, embedding_weight


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, seq_mask):
        # hidden = [1, Batch size, hidden size]
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # hidden =[seq len, batch size, hidden size]
        # encoder_outputs =[seq len, batch size, hidden size]

        batch_size = encoder_outputs.size(1)
        energy = torch.cat((hidden, encoder_outputs), 2)  # energy = [seq len, batch size, hidden size=512 + hidden size]
        energy = energy.view(-1, self.input_size + self.hidden_size)  # energy = [seq len * batch size, hidden size + input size]

        score_feature = torch.tanh(self.attn(energy))  # score_feature = = [seq len * batch size, hidden size]
        score = self.score(score_feature)  # score = [seq len * batch size, 1]
        score = score.squeeze(1)  # score = [seq len, batch size]
        score = score.view(max_len, batch_size).transpose(0, 1)  # score = [batch size, seq len]
        if seq_mask is not None:
            score = score.masked_fill_(seq_mask.bool(), -1e12)
        score = F.softmax(score, dim=1)
        return score.unsqueeze(1)


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_s = nn.Linear(input_size + hidden_size, hidden_size)
        self.weight_n = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, embedding, nums_mask=None):
        # embedding = [batch size, seq len, hidden size]
        # hidden = [batch size, 1, hidden size + input size]
        max_len = embedding.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # hidden = [batch size, seq len, hidden size + input size]

        batch_size = embedding.size(0)
        energy = torch.cat((hidden, embedding), 2)
        energy = energy.view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.weight_s(energy))
        score = self.weight_n(score_feature)
        score = score.squeeze(1)
        score = score.view(batch_size, -1)
        if nums_mask is not None:
            score = score.masked_fill_(nums_mask.bool(), -1e12)
        return score


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()
        self.hidden_size = hidden_size
        self.op_nums = op_nums
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.dropout(node_embedding)
        current_context = self.dropout(current_context)

        left_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        left_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        right_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        right_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        left_child = left_child * left_child_g
        right_child = right_child * right_child_g

        return left_child, right_child, node_label


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        # 两层神经网络
        sub_tree_1 = self.dropout(sub_tree_1)
        sub_tree_2 = self.dropout(sub_tree_2)
        node_embedding = self.dropout(node_embedding)
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r