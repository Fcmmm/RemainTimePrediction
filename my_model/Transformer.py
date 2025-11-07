import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from math import sqrt

# MultiHeadAttention：实现了多头注意力机制
# TransformerBlock：包含了注意力层和前馈网络
# TransformerModel：主模型，整合了所有组件
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.4):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2)
        out = out.reshape(batch_size, seq_len, embedding_dim)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.4):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=64, n_layers=1,
                 dropout=0.4, embedding=None, num_heads=3,
                 numeric_input_dim=7, numeric_output_dim=3):
        super(Transformer, self).__init__()
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads}). "
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.batch_size = batch_size
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embedding_dim)
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)
        self.input_dim = embedding_dim + numeric_output_dim

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=self.input_dim,  # 这里输入维度要变！
                num_heads=num_heads,
                ff_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(self.input_dim, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        self.embedding = self.embedding.to(device)
        self.numeric_layer = self.numeric_layer.to(device)
        self.fc = self.fc.to(device)
        for block in self.transformer_blocks:
            block.to(device)
        # X shape: [batch, seq, feature_dim]
        batch_size, seq_length, feature_dim = X.size()
        event_types = X[:, :, 0].long()     # 类别特征
        numeric_attrs = X[:, :, 1:].float() # 数值特征

        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            raise ValueError(f"Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        embedded_event = self.embedding(event_types)                   # [batch, seq, embedding_dim]
        transformed_numeric = self.numeric_layer(numeric_attrs)        # [batch, seq, numeric_output_dim]
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1) # [batch, seq, input_dim]

        x = combined_input
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x