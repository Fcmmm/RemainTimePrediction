import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable  # 注意 Variable 已被弃用，但暂时保留用于修改
import torch.nn.functional as F
from math import sqrt

class BiGRUAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(BiGRUAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, n_layer))
        print('Initialization BiGRU Model')
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                          num_layers=self.n_layer, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, out_size)

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
        hidden_state = torch.zeros(self.n_layer * 2, self.batch_size, self.hidden_dim).to(device)
        output, final_hidden_state = self.rnn(input, hidden_state)
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, hidden_dim * 2]
        output = self.attention_net(output)
        output = self.out(output)
        return output  # model : [batch_size, num_classes]