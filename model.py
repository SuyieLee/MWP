import random
import torch
import torch.nn as nn
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


