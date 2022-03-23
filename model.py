import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Dropout
from torch.autograd import Variable
from data_tools import post_solver, inverse_temp_to_num


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
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.layers, bidirectional=self.bidirectional, dropout=self.dropout)

    def forward(self, input):
        # embed = [input len, batch size]
        # output = [input len, batch size, hidden dim * directions)
        # hidden = [layers * directions, batch size, hidden dim]
        # cell = [layers * directions, batch size, hidden dim]
        embed = self.embed_model(input)
        embed = self.input_dropout(embed)
        output, hidden = self.rnn(embed)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_model=None, embed_size=100, hidden_size=1024, input_dropout=0, dropout=0,
                 layers=1, sos_id=None, eos_id=None, classes_size=None, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.dropout = dropout
        self.layers = layers
        self.classes_size = classes_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.bidirectional = bidirectional
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_model:
            self.embedding = embed_model
        else:
            self.embed_model = nn.Embedding(self.vocab_size, self.embed_size)
        if layers == 2:
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, layers, dropout=dropout)
        else:
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, layers)

        self.out = nn.Linear(self.hidden_size, self.classes_size)
        self.function = F.log_softmax

    def forward(self, input, hidden, max_length=128):
        # input = [batch size]
        # hidden = [layers * directions, batch size, hidden dim]
        # cell = [layers * directions, batch size, hidden dim]
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)
        # input = [1, batch size]
        # embedded = [1, batch size, emb size]
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.out(output.squeeze(0))
        prediction_softmax = self.function(prediction, dim=1)  # # 0是对列做归一化，1是对行做归一化
        # output = [seq len, batch size, hid dim * directions]
        # hidden = [layers * directions, batch size, hid dim]
        # cell = [layers * directions, batch size, hid dim]
        # prediction = [batch size, output dim]

        return prediction_softmax, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, input_len, target_len, teacher_forcing_ratio = 0.5):
        batch_size = len(input_len)
        target_len = max(target_len)
        target_vocab_size = self.decoder.classes_size
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden = self.encoder(input)
        # 处理hidden和cell，将双向的合并成一个纬度
        if self.encoder.bidirectional:
            hidden = tuple([self._cat_directions(h) for h in hidden])
            # hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)

        decoder_input = target[0, :]  # decoder 的第一个输入是<sos>
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)  # hidden = (hidden, cell)
            outputs[t] = output
            teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            if teacher_forcing:
                decoder_input = target[t]
            else:
                decoder_input = top1
        return outputs

    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class Trainer(object):
    def __init__(self, model, loss=None, weight=None, vocab_dict=None, vocab_list=None, data_loader=None, batch_size=32, decode_classes_dict=None, decode_classes_list=None,
                 cuda_use=True, TRG_PAD_IDX=None, print_every=10, checkpoint_dir_name=None):
        self.model = model
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.data_loader = data_loader
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.cuda_use = cuda_use
        self.print_every = print_every
        self.optimizer = optim.Adam(model.parameters())
        self.batch_size = batch_size
        self.TRG_PAD_IDX =TRG_PAD_IDX
        if loss is None:
            self.criterion = nn.NLLLoss(weight=weight, reduction='mean')
        else:
            self.criterion = loss

    def train(self, model, epoch_num=100, resume=False, valid_every=10):
        train_list = self.data_loader.train_data
        # test_list = self.data_loader.test_data
        valid_list = self.data_loader.valid_data
        # start_epoch = 1

        final_loss = 0
        for epoch in range(epoch_num):
            start_step = 0
            total_num = 0
            total_loss = 0
            match = 0
            total = 0
            model.train()

            print("Epoch " + str(epoch+1) + " start training!")
            for batch in self.data_loader.yield_batch(train_list, self.batch_size):
                input = batch['batch_encode_pad_idx']
                input_len = batch['batch_encode_len']
                target = batch['batch_decode_pad_idx']
                target_len = batch['batch_decode_len']
                function_ans = batch['batch_ans']
                num_list = batch['batch_num_list']
                batch_size = len(input)
                total_num += batch_size

                input = Variable(torch.LongTensor(input))
                target = Variable(torch.LongTensor(target))

                input = input.transpose(0, 1)
                target = target.transpose(0, 1)

                if self.cuda_use:
                    input = input.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()

                output = model(input, target, input_len, target_len)

                # target = [trg len * batch size]
                # output = [trg len, batch size, classes len]
                classes_len = output.shape[-1]
                output = output[1:].view(-1, classes_len)
                target = target[1:].contiguous().view(-1)

                if self.cuda_use:
                    output = output.cuda()
                    target = target.cuda()

                # batch_acc_num = self.get_ans_acc(output, function_ans, batch_size, num_list)
                loss = self.criterion(output, target)
                total_loss += loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # CLIP=1
                self.optimizer.step()

                # non_padding = target.ne(self.TRG_PAD_IDX)
                # correct = output.view(-1).eq(target).masked_select(non_padding).sum().item()  # data[0]
                # match += correct
                # total += non_padding.sum().item()

                start_step += 1
                if start_step % self.print_every == 0:
                    print("Step " + str(start_step) + " Batch Loss: " + str(total_loss/total_num))

            if (epoch+1) % valid_every == 0 and epoch > 0:
                valid_loss = self.evaluate(model, valid_list)
                print("Epoch " + str(epoch + 1) + " Valid Batch Loss: " + str(valid_loss/len(valid_list)))
            final_loss = total_loss
            print("Epoch " + str(epoch+1) + " Batch Loss: " + str(final_loss/len(train_list)))
        return final_loss/len(train_list)

    def evaluate(self, model, data):
        model.eval()
        epoch_loss = 0
        for batch in self.data_loader.yield_batch(data, self.batch_size):
            input = batch['batch_encode_pad_idx']
            input_len = batch['batch_encode_len']
            target = batch['batch_decode_pad_idx']
            target_len = batch['batch_decode_len']

            input = Variable(torch.LongTensor(input))
            target = Variable(torch.LongTensor(target))

            input = input.transpose(0, 1)
            target = target.transpose(0, 1)

            if self.cuda_use:
                input = input.cuda()
                target = target.cuda()

            output = model(input, target, input_len, target_len)

            classes_len = output.shape[-1]
            output = output[1:].view(-1, classes_len)
            target = target[1:].contiguous().view(-1)
            if self.cuda_use:
                output = output.cuda()
                target = target.cuda()
            loss = self.criterion(output, target)
            epoch_loss += loss
        return epoch_loss/len(data)

    def get_ans_acc(self, output, function_ans, batch_size, num_list):
        acc = 0
        output = output.view(-1, batch_size, len(self.decode_classes_list))
        output = output.transpose(0, 1)
        for i in range(len(output)):
            templates = self.get_template(output[i])
            # print(templates)
            try:
                equ = inverse_temp_to_num(templates, num_list[i])
                # print(equ)
                predict_ans = post_solver(equ)
                if predict_ans == function_ans:
                    acc += 1
            except:
                acc += 0
        return acc

    def get_template(self, pred):
        templates = []
        for vec in pred:
            idx = vec.argmax(0).item()
            if idx == self.decode_classes_dict['PAD_token'] or idx == self.decode_classes_dict['END_token']:
                break
            templates.append(self.decode_classes_list[idx])
        return templates







