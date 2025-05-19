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
class LSTM(nn.Module):
    # vocab_size 输入特征数量 embedding_dim 隐藏层数量 hidden_dim 隐藏层神经元数量 out_size batch_size n_layer dropout 抗过拟合机制 embedding
    def __init__(self,vocab_size,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        print(vocab_size)
        print(embedding_dim)
        print(hidden_dim)
        print(out_size)
        print(embedding)
        print(batch_size)
        print(n_layer)
        print(dropout)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                           num_layers=self.n_layer, bidirectional=False)
        self.out = nn.Linear(hidden_dim, out_size)
    def forward(self, X):
        # 确保所有张量都在同一设备上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)

        input = self.embedding(X).permute(1, 0, 2)

        hidden_state = torch.randn(self.n_layer, self.batch_size, self.hidden_dim, device=device)
        cell_state = torch.randn(self.n_layer, self.batch_size, self.hidden_dim, device=device)

        hidden_state = Variable(hidden_state)
        cell_state = Variable(cell_state)

        output, (final_hidden_state, final_cell_state) = self.rnn(input, (hidden_state, cell_state))
        hn = output[-1]
        output = self.out(hn).to(device)
        return output
