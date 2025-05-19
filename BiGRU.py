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
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, CUDA_type=True):
        super(BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.CUDA_type = CUDA_type
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.biType = 2  # 注意这里不需要再次赋值
        print('Initialization BiGRU Model')

        # 根据是否使用GPU来决定模型层的初始化
        if self.CUDA_type:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                          num_layers=self.n_layer, bidirectional=True).to(self.device)
        self.out = nn.Linear(hidden_dim * self.biType, out_size).to(self.device)
        self.embedding = self.embedding.to(self.device)

    def forward(self, X):
        input = self.embedding(X).to(self.device)  # 确保输入和embedding在同一个设备上
        input = input.permute(1, 0, 2)  # 调整维度以匹配GRU的输入需求
        hidden_state = torch.zeros(self.n_layer * self.biType, self.batch_size, self.hidden_dim).to(self.device)
        output, final_hidden_state = self.rnn(input, hidden_state)
        hn = output[-1]
        output = self.out(hn)
        return output
