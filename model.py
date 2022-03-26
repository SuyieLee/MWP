import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_model=None, embed_size=100, encode_hidden_size=128, decode_hidden_size=128, input_dropout=0, dropout=0,
                 layers=1, bidirectional=False, mode='lstm'):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.dropout = dropout
        self.layers = layers
        self.bidirectional = bidirectional
        self.mode = mode
        if embed_model is None:
            self.embed_model = nn.Embedding(self.vocab_size, self.embed_size)
        else:
            self.embed_model = embed_model
        # 使用batch_first=True，可以使lstm接受维度为(batchsize，序列长度，输入维数)的数据输入，同时，lstm的输出数据维度也会变为batchsize放在第一维
        # 如果使用单层的话，就不要用dropout=self.dropout
        if self.layers == 1:
            self.rnn = nn.GRU(self.embed_size, self.encode_hidden_size, num_layers=self.layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, self.encode_hidden_size, num_layers=self.layers, bidirectional=self.bidirectional, dropout=self.dropout)

        if self.mode == 'gru':
            self.fc = nn.Linear(self.encode_hidden_size * 2, self.decode_hidden_size)

    def forward(self, input):
        # embed = [input len, batch size]
        # output = [input len, batch size, hidden dim * directions)
        # hidden = [layers * directions, batch size, hidden dim]
        # cell = [layers * directions, batch size, hidden dim]
        embed = self.embed_model(input)
        embed = self.input_dropout(embed)
        output, hidden = self.rnn(embed)
        if self.mode == 'gru':
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden = [batch size, decode hidden size]
        return hidden, output


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_model=None, embed_size=100, encode_hidden_size=None, decode_hidden_size=1024, input_dropout=0, dropout=0,
                 layers=1, classes_size=None, bidirectional=False, attn=None):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.classes_size = classes_size
        self.bidirectional = bidirectional
        self.attn = attn
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_model:
            self.embedding = embed_model
        else:
            self.embed_model = nn.Embedding(self.vocab_size, self.embed_size)
        if layers == 2:
            self.rnn = nn.GRU(self.encode_hidden_size*2+self.embed_size, self.decode_hidden_size, layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(self.encode_hidden_size*2+self.embed_size, self.decode_hidden_size, layers)

        self.out = nn.Linear(self.encode_hidden_size*2+self.embed_size+self.decode_hidden_size, self.classes_size)
        self.function = F.log_softmax

    def forward(self, input, hidden, encoder_output):
        # input = [batch size]
        # hidden = [batch size, decode hidden dim]
        # cell = [layers * directions, batch size, hidden dim]
        input = input.unsqueeze(0)  # input = [1, batch size]
        embedded = self.embedding(input)  # embedded = [1, batch size, emb size]
        embedded = self.input_dropout(embedded)

        a = self.attn(hidden, encoder_output).unsqueeze(1)
        encoder_output = encoder_output.permute(1,0,2)
        w = torch.bmm(a, encoder_output)
        w = w.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, w), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # 两层，Expected hidden size (2, 64, 1024), got [1, 64, 1024]

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        w = w.squeeze(0)

        prediction = self.out(torch.cat((output, w, embedded), dim=1))
        prediction_softmax = self.function(prediction, dim=1)  # # 0是对列做归一化，1是对行做归一化
        # output = [seq len, batch size, hid dim * directions]
        # hidden = [layers * directions, batch size, hid dim]
        # cell = [layers * directions, batch size, hid dim]
        # prediction = [batch size, output dim]

        return prediction_softmax, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(self, encode_hidden_size, decode_hidden_size):
        super().__init__()
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size
        self.attn = nn.Linear(self.encode_hidden_size*2 + self.decode_hidden_size, self.decode_hidden_size)
        self.v = nn.Linear(self.decode_hidden_size, 1, bias=False)

    def forward(self, hidden, output):
        input_max_len = output.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, input_max_len, 1)
        output = output.permute(1,0,2)
        energy = torch.tanh(self.attn(torch.cat((hidden, output), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, mode):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mode = mode

    def forward(self, input, target, input_len, target_len, teacher_forcing_ratio = 0.5):
        batch_size = len(input_len)
        target_len = max(target_len)
        target_vocab_size = self.decoder.classes_size
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, encoder_output = self.encoder(input)
        # 处理hidden和cell，将双向的合并成一个纬度
        if self.encoder.bidirectional and self.mode == 'lstm':
            hidden = tuple([self._cat_directions(h) for h in hidden])
            # hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)

        decoder_input = target[0, :]  # decoder 的第一个输入是<sos>
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_output)  # hidden = (hidden, cell)
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

