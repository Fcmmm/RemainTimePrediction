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
class LSTMAtt(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None,numeric_input_dim = 7,numeric_output_dim = 4):
        super(LSTMAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim).cuda()
        self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim, hidden_dim).cuda()).cuda()
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer).cuda())
        print('Initialization LSTMAtt Model')
        self.rnn = nn.LSTM(input_size=embedding_dim+numeric_output_dim, hidden_size=hidden_dim, dropout=self.dropout,
                           num_layers=self.n_layer, bidirectional=False).cuda()
        self.out = nn.Linear(hidden_dim, out_size).cuda()
    # def attention_net(self, rnn_output):
    #     attn_weights = torch.matmul(rnn_output,self.weight_Mu).cuda()
    #     soft_attn_weights = F.softmax(attn_weights, 1)
    #     context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
    #     return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, hidden_dim * num_directions(=2)]

    def attention_net(self, rnn_output):
        # rnn_output: [batch_size, seq_length, hidden_dim]
        # 修改注意力机制的实现
        batch_size, seq_length, hidden_dim = rnn_output.size()

        attn_weights = torch.matmul(rnn_output, self.weight_Mu)  # [batch_size, seq_length, n_layer]
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # 在序列维度上做softmax

        # 如果n_layer=1，需要处理维度
        if soft_attn_weights.size(-1) == 1:
            soft_attn_weights = soft_attn_weights.squeeze(-1)  # [batch_size, seq_length]
            # 加权求和得到上下文向量
            context = torch.bmm(soft_attn_weights.unsqueeze(1), rnn_output).squeeze(1)  # [batch_size, hidden_dim]
        else:
            # 多层的情况，取平均或使用其他策略
            soft_attn_weights = soft_attn_weights.mean(dim=-1)  # [batch_size, seq_length]
            context = torch.bmm(soft_attn_weights.unsqueeze(1), rnn_output).squeeze(1)  # [batch_size, hidden_dim]

        return context, soft_attn_weights.data.cpu().numpy()
    def forward(self, X):
        batch_size, seq_length, feature_dim = X.size()
        event_types = X[:, :, 0].long()  # 取第一个特征作为 event_type
        numeric_attrs = X[:, :, 1:].float()  # 其余特征作为数值属性

        # 添加维度检查
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            print(f"Warning: Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")
            # 可以动态调整或抛出错误

        embedded_event = self.embedding(event_types)
        # 对数值属性进行线性变换
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch_size, seq_length, numeric_output_dim]
        # 拼接嵌入和数值属性
        combined_input = torch.cat((embedded_event, transformed_numeric),dim=-1)  # [batch_size, seq_length, embedding_dim + numeric_output_dim]

        # 调整输入顺序以适配 LSTM (seq_length, batch_size, input_dim)
        combined_input = combined_input.permute(1, 0, 2)
        # 初始化隐藏状态和细胞状态
        hidden_state = Variable(torch.randn(self.n_layer, batch_size, self.hidden_dim).cuda())
        cell_state = Variable(torch.randn(self.n_layer, batch_size, self.hidden_dim).cuda())

        # LSTM 处理
        output, (final_hidden_state, final_cell_state) = self.rnn(combined_input, (hidden_state, cell_state))

        # 调整输出顺序回 [batch_size, seq_length, hidden_dim]
        output = output.permute(1, 0, 2)

        # 应用注意力机制
        output, attention = self.attention_net(output)

        # 最终输出
        output = self.out(output)

        return output
