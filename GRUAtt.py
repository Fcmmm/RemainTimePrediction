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

class GRUAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(GRUAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer))
        print('Initialization GRU Model')
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                          num_layers=self.n_layer, bidirectional=False)
        self.out = nn.Linear(hidden_dim, out_size)

    def attention_net(self, rnn_output):
        attn_weights = torch.matmul(rnn_output, self.weight_Mu)
        soft_attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = self.embedding.to(device)
        X = X.to(device)
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        hidden_state = torch.zeros(self.n_layer, self.batch_size, self.hidden_dim).to(device)
        output, _ = self.rnn(input, hidden_state)
        output = output.permute(1, 0, 2)  # output : [batch_size, seq_len, hidden_dim]
        output = self.attention_net(output)
        output = self.out(output)
        return output  # model : [batch_size, num_classes]