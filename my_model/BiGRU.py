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
                 embedding=None, CUDA_type=True, numeric_input_dim=7, numeric_output_dim=4):
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
        self.biType = 2
        self.device = torch.device('cuda') if self.CUDA_type else torch.device('cpu')

        # 新增：处理数值特征的线性层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim).to(self.device)

        # BiGRU层输入维度需加上数值特征维度
        self.rnn = nn.GRU(
            input_size=embedding_dim + numeric_output_dim,
            hidden_size=hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_layer,
            bidirectional=True
        ).to(self.device)
        self.out = nn.Linear(hidden_dim * self.biType, out_size).to(self.device)
        self.embedding = self.embedding.to(self.device)

    def forward(self, X):
        """
        X: [batch_size, seq_length, feature_dim]
        第一个特征是事件类型，其余是数值特征
        """
        batch_size, seq_length, feature_dim = X.size()

        event_types = X[:, :, 0].long()
        numeric_attrs = X[:, :, 1:].float()

        # 事件类型做embedding
        embedded_event = self.embedding(event_types)  # [batch, seq, emb_dim]
        # 数值特征线性层
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch, seq, numeric_output_dim]

        # 拼接
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)  # [batch, seq, emb+num]
        combined_input = combined_input.permute(1, 0, 2)  # [seq, batch, input_dim]

        # 初始化隐藏状态
        hidden_state = torch.zeros(self.n_layer * self.biType, batch_size, self.hidden_dim).to(self.device)

        output, final_hidden_state = self.rnn(combined_input, hidden_state)
        hn = output[-1]  # [batch, hidden_dim*2]
        output = self.out(hn)
        return output

