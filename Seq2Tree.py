import torch
import torch.nn as nn
from treemodel import *


class Seq2Tree(nn.Module):
    def __init__(self, data_loader, encoder, prediction, generation, merge, learning_rate, weight_decay, cuda_use):
        super().__init__()
        self.data_loader = data_loader
        self.encoder = encoder
        self.prediction = prediction
        self.generation = generation
        self.merge = merge
        self.encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.prediction_optimizer = torch.optim.Adam(prediction.parameters(), lr=learning_rate,
                                                     weight_decay=weight_decay)
        self.generation_optimizer = torch.optim.Adam(generation.parameters(), lr=learning_rate,
                                                     weight_decay=weight_decay)
        self.merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=20, gamma=0.5)
        self.prediction_scheduler = torch.optim.lr_scheduler.StepLR(self.prediction_optimizer, step_size=20, gamma=0.5)
        self.generation_scheduler = torch.optim.lr_scheduler.StepLR(self.generation_optimizer, step_size=20, gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(self.merge_optimizer, step_size=20, gamma=0.5)
        self.cuda_use = cuda_use

    def forward(self, input, input_len, target, target_len, num_size_batch, generate_nums, num_pos, nums_stack_batch):
        seq_mask = []
        max_len = max(input_len)
        for i in input_len:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_nums)
        for i in num_size_batch:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.prediction.hidden_size)]).unsqueeze(0)
        batch_size = len(input_len)

        if self.cuda_use:
            input = input.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        encoder_outputs, problem_output = self.encoder(input, input_len)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # 生成根节点
        max_target_length = max(target_len)  # 表达式的最大长度
        all_node_outputs = []

        copy_num_len = [len(_) for _ in num_pos]  # 数字列表的长度
        num_size = max(copy_num_len)  # 数字列表最大长度
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                       self.encoder.hidden_size)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.prediction(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)
            unk = self.data_loader.decode_classes_dict['UNK_token']
            num_start = 5  # 5是指符号的个数，除符号外数字开始的下标
            target_t, generate_input = self.generate_tree_input(
                target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if self.cuda_use:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.generation(
                current_embeddings, generate_input, current_context)  # current_embeddings = node embedding, generate_input = node label
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(
                            op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()
        if self.cuda_use:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()

        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss = self.masked_cross_entropy(all_node_outputs, target, target_len)
        # loss = loss_0 + loss_1
        loss.backward()
        # clip the grad
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.prediction.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.generation.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.merge.parameters(), 1)

        # Update parameters with optimizers
        return loss.item()

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        """
        将在encoder output中抽取出数字对应的编码，数字列表下标也做padding，并且记录哪些是padding的数字，然后做mask
        长度是batch中，数字的个数的最大值。
        """
        # num_pos 数字下标列表
        # num_size 数字列表最大长度
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.ByteTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.cuda_use:
            indices = indices.cuda()
            masked_index = masked_index.cuda()
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        # S x B x H -> (B x S) x H
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index.bool(), 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        # when the decoder input is copied num but the num has two pos, chose the max
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def masked_cross_entropy(self, logits, target, length):
        if torch.cuda.is_available():
            length = torch.LongTensor(length).cuda()
        else:
            length = torch.LongTensor(length)
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = nn.functional.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=0, index=target_flat)

        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = self.sequence_mask(sequence_length=length, max_len=target.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        # if loss.item() > 10:
        #     print(losses, target)
        return loss

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand