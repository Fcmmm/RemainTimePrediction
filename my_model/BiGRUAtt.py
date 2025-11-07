import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable  # 注意 Variable 已被弃用，但暂时保留用于修改
import torch.nn.functional as F
from math import sqrt

class BiGRUAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(BiGRUAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 数值特征线性层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)

        # BiGRU输入维度要加上numeric_output_dim
        self.rnn = nn.GRU(
            input_size=embedding_dim + numeric_output_dim,
            hidden_size=hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_layer,
            bidirectional=True
        )

        self.out = nn.Linear(hidden_dim * 2, out_size)

        # 注意力权重参数，hidden_dim*2 × 1
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        nn.init.xavier_uniform_(self.weight_Mu)

    def attention_net(self, rnn_output):
        # rnn_output: [batch_size, seq_length, hidden_dim*2]
        attn_scores = torch.matmul(rnn_output, self.weight_Mu)  # [batch_size, seq_length, 1]
        attn_weights = F.softmax(attn_scores, dim=1)            # [batch_size, seq_length, 1]
        # 加权求和得到context
        context = torch.sum(rnn_output * attn_weights, dim=1)   # [batch_size, hidden_dim*2]
        return context, attn_weights.squeeze(-1).detach().cpu().numpy()

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        self.numeric_layer = self.numeric_layer.to(device)
        self.rnn = self.rnn.to(device)
        self.out = self.out.to(device)
        self.weight_Mu = self.weight_Mu.to(device)
        self.embedding = self.embedding.to(device)

        batch_size, seq_length, feature_dim = X.size()
        event_types = X[:, :, 0].long()           # 取第一个特征作为 event_type
        numeric_attrs = X[:, :, 1:].float()       # 其余特征作为数值属性

        # 维度检查
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            raise ValueError(f"Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        # 事件类型嵌入
        embedded_event = self.embedding(event_types)  # [batch_size, seq_length, embedding_dim]

        # 数值特征线性变换
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch_size, seq_length, numeric_output_dim]

        # 拼接
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)  # [batch, seq, emb+num]
        combined_input = combined_input.permute(1, 0, 2)  # [seq_length, batch_size, input_dim]

        # 初始化hidden
        num_directions = 2
        hidden_state = torch.zeros(self.n_layer * num_directions, batch_size, self.hidden_dim).to(device)

        # BiGRU
        output, _ = self.rnn(combined_input, hidden_state)  # [seq_length, batch_size, hidden_dim*2]

        # 回到 [batch_size, seq_length, hidden_dim*2]
        output = output.permute(1, 0, 2)

        # 注意力机制
        context, attention = self.attention_net(output)  # [batch_size, hidden_dim*2]

        # 输出
        output = self.out(context)  # [batch_size, out_size]
        return output