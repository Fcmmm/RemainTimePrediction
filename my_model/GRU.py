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


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(GRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embedding_dim)
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 添加数值属性处理层（与LSTMAtt保持一致）
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim).cuda()

        print('Initialization GRU Model')

        # GRU层，输入维度修改为embedding_dim + numeric_output_dim
        self.rnn = nn.GRU(input_size=embedding_dim + numeric_output_dim, hidden_size=hidden_dim,
                          dropout=self.dropout, num_layers=self.n_layer, bidirectional=False).cuda()

        # 输出层
        self.out = nn.Linear(hidden_dim, out_size).cuda()

    def forward(self, X):
        """
        前向传播，处理与LSTMAtt相同格式的输入数据
        X: [batch_size, seq_length, feature_dim]
        其中第一个特征是事件类型，其余是数值属性
        """
        batch_size, seq_length, feature_dim = X.size()

        # 分离事件类型和数值属性（与LSTMAtt相同的处理方式）
        event_types = X[:, :, 0].long()  # 取第一个特征作为 event_type
        numeric_attrs = X[:, :, 1:].float()  # 其余特征作为数值属性

        # 添加维度检查（与LSTMAtt相同）
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            print(f"Warning: Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        # 事件类型嵌入
        embedded_event = self.embedding(event_types)  # [batch_size, seq_length, embedding_dim]

        # 对数值属性进行线性变换（与LSTMAtt相同）
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch_size, seq_length, numeric_output_dim]

        # 拼接嵌入和数值属性（与LSTMAtt相同）
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)
        # [batch_size, seq_length, embedding_dim + numeric_output_dim]

        # 调整输入顺序以适配 GRU (seq_length, batch_size, input_dim)
        combined_input = combined_input.permute(1, 0, 2)

        # 初始化隐藏状态
        hidden_state = torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda()

        # GRU 处理
        output, final_hidden_state = self.rnn(combined_input, hidden_state)

        # 使用最后一个时间步的输出
        hn = output[-1]  # [batch_size, hidden_dim]

        # 最终输出
        output = self.out(hn)

        return output
