# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep


class LSTM_NER(nn.Module):
    def __init__(self, data):
        super(LSTM_NER, self).__init__()
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.bilstm_flag = data.HP_bilstm
        self.hidden_dim = data.HP_hidden_dim
        # word embedding
        self.wordrep = WordRep(data)
        self.l_hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        self.architectures1_fc_dropout = data.HP_architectures1_dropout

        self.input_size = self.wordrep.total_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.lstms = nn.ModuleList([nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag)])
        for _ in range(data.HP_architectures1_layer-1):
            self.lstms.append(nn.LSTM(data.HP_hidden_dim, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag))
        if self.gpu:
            self.lstms = self.lstms.cuda()
            self.l_hidden2tag = self.l_hidden2tag.cuda()

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        word_represent = self.forward_word(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                           char_seq_recover)
        return self.forward_rest(word_represent, word_seq_lengths.cpu())

    def forward_word(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover):
        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        return word_represent

    def forward_rest(self, word_represent, word_seq_lengths):
        if not self.training:
            ordered_lens, index = word_seq_lengths.sort(descending=True)
            ordered_x = word_represent[index]
        else:
            ordered_x, ordered_lens = word_represent, word_seq_lengths

        for i,lstm in enumerate(self.lstms):
            pack_input = pack_padded_sequence(ordered_x, ordered_lens.cpu(), batch_first=True)
            pack_output, _ = lstm(pack_input)
            ordered_x, _ = pad_packed_sequence(pack_output, batch_first=True)

        if not self.training:
            recover_index = index.argsort()
            lstm_out = ordered_x[recover_index]
        else:
            lstm_out = ordered_x

        h2t_in = add_dropout(lstm_out, self.architectures1_fc_dropout)

        logits = self.l_hidden2tag(h2t_in)
        return lstm_out, logits


def add_dropout(x, dropout):
    ''' x: batch * seq_len * hidden '''
    return F.dropout2d(x.transpose(1,2)[...,None], p=dropout, training=True).squeeze(-1).transpose(1,2)