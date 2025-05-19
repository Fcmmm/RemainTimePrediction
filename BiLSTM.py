import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                           num_layers=self.n_layer, bidirectional=True)
        self.out = nn.Linear(hidden_dim , out_size)

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)

        input = self.embedding(X).permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]

        # 初始化隐藏状态和单元状态，确保它们与 LSTM 层的期望维度一致
        num_directions = 2  # 因为是双向 LSTM
        hidden_state = torch.zeros(self.n_layer * num_directions, self.batch_size, self.hidden_dim).to(device)
        cell_state = torch.zeros(self.n_layer * num_directions, self.batch_size, self.hidden_dim).to(device)

        output, (final_hidden_state, final_cell_state) = self.rnn(input, (hidden_state, cell_state))

        # 选择最终的隐藏状态和单元状态
        # 对于双向 LSTM，final_hidden_state 和 final_cell_state 都是一个元组
        # (h_forward, h_backward)，您可以选择一个或将它们结合起来
        hn = final_hidden_state[-1]  # 选择最后一层的前向隐藏状态

        # 如果需要，将前向和后向的隐藏状态结合起来
        # hn = torch.cat((final_hidden_state[0][-1], final_hidden_state[1][-1]), dim=1)

        # 通过输出层进行分类
        output = self.out(hn)

        return output
