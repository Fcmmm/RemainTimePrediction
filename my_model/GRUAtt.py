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
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(GRUAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        attn_dim = hidden_dim  # 或其他
        self.query_proj = nn.Linear(hidden_dim, attn_dim)
        self.key_proj = nn.Linear(hidden_dim, attn_dim)
        self.value_proj = nn.Linear(hidden_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)

        # 数值特征线性层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)

        # GRU输入维度要加上numeric_output_dim
        self.rnn = nn.GRU(
            input_size=embedding_dim + numeric_output_dim,
            hidden_size=hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_layer,
            bidirectional=False
        )

        self.out = nn.Linear(hidden_dim, out_size)

        # 注意力权重参数，hidden_dim × 1
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.weight_Mu)

    # def attention_net(self, rnn_output):
    #     # rnn_output: [batch_size, seq_length, hidden_dim]
    #     attn_scores = torch.matmul(rnn_output, self.weight_Mu)  # [batch_size, seq_length, 1]
    #     attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_length, 1]
    #     # 加权和得到上下文
    #     context = torch.sum(rnn_output * attn_weights, dim=1)  # [batch_size, hidden_dim]
    #     return context, attn_weights.squeeze(-1).detach().cpu().numpy()

    def attention_net(self, rnn_output, mask=None):
        """
        rnn_output: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] or None
        返回: context [batch, hidden_dim], attn_weights [batch, seq_len]
        """
        # 1. Linear projections for query/key/value
        query = self.query_proj(rnn_output)  # [batch, seq_len, attn_dim]
        key = self.key_proj(rnn_output)  # [batch, seq_len, attn_dim]
        value = self.value_proj(rnn_output)  # [batch, seq_len, attn_dim]
        # 2. Attention score (scaled dot-product)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)  # [batch, seq_len, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]
        attn_weights_avg = attn_weights.mean(dim=1)  # [batch, seq_len]，可视化时用
        # 3. Attention聚合
        context_seq = torch.matmul(attn_weights, value)  # [batch, seq_len, attn_dim]
        context_seq = self.out_proj(context_seq)  # [batch, seq_len, hidden_dim]
        # 4. 残差连接与归一化
        out = context_seq + rnn_output
        if hasattr(self, 'layernorm'):
            out = self.layernorm(out)
        # 5. 全局池化 (默认mean，可自定义)
        context = out.mean(dim=1)  # [batch, hidden_dim]
        return context, attn_weights_avg.detach().cpu().numpy()

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        self.numeric_layer = self.numeric_layer.to(device)
        self.rnn = self.rnn.to(device)
        self.out = self.out.to(device)
        self.weight_Mu = self.weight_Mu.to(device)
        self.embedding = self.embedding.to(device)

        batch_size, seq_length, feature_dim = X.size()
        event_types = X[:, :, 0].long()
        numeric_attrs = X[:, :, 1:].float()

        # 维度检查
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            raise ValueError(f"Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        # 嵌入处理
        embedded_event = self.embedding(event_types)  # [batch_size, seq_length, embedding_dim]

        # 数值特征线性映射
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch_size, seq_length, numeric_output_dim]

        # 拼接
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)  # [batch_size, seq_length, total_dim]
        combined_input = combined_input.permute(1, 0, 2)  # [seq_length, batch_size, input_dim]

        # 初始化隐藏状态
        hidden_state = torch.zeros(self.n_layer, batch_size, self.hidden_dim).to(device)

        # GRU
        output, _ = self.rnn(combined_input, hidden_state)  # [seq_length, batch_size, hidden_dim]

        # 变换回 [batch_size, seq_length, hidden_dim]
        output = output.permute(1, 0, 2)

        # 注意力机制
        context, attention = self.attention_net(output)  # [batch_size, hidden_dim]

        # 输出
        output = self.out(context)  # [batch_size, out_size]
        return output