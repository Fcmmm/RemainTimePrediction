import torch
import torch.nn as nn
from torch.autograd import Variable


class BiQRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(BiQRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 前向和后向的卷积层
        self.conv_forward = nn.Conv1d(embedding_dim, hidden_dim * 3, kernel_size=1)
        self.conv_backward = nn.Conv1d(embedding_dim, hidden_dim * 3, kernel_size=1)

        self.dropout_layer = nn.Dropout(dropout)
        # 输出层接收双向的hidden states，所以维度翻倍
        self.out = nn.Linear(hidden_dim * 2, out_size)

    def _process_direction(self, conv_out, device, forward=True):
        # 分割到Z, F, O门
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        # 初始化隐状态
        h = torch.zeros(self.batch_size, self.hidden_dim, device=device)

        # 序列长度
        seq_len = Z.size(0)

        # 根据方向决定处理顺序
        if forward:
            sequence_range = range(seq_len)
        else:
            sequence_range = range(seq_len - 1, -1, -1)

        for t in sequence_range:
            h = F[t] * h + (1 - F[t]) * Z[t]

        return h

    def forward(self, X):
        # 确保所有张量都在同一设备上
        device = X.device

        # 嵌入层
        embedded = self.embedding(X).permute(0, 2, 1)  # (batch_size, embedding_dim, sequence_length)

        # 前向和后向的卷积操作
        conv_out_forward = self.conv_forward(embedded)
        conv_out_backward = self.conv_backward(embedded)

        # 转换维度为 (sequence_length, batch_size, hidden_dim * 3)
        conv_out_forward = conv_out_forward.permute(2, 0, 1)
        conv_out_backward = conv_out_backward.permute(2, 0, 1)

        # 分别处理前向和后向
        h_forward = self._process_direction(conv_out_forward, device, forward=True)
        h_backward = self._process_direction(conv_out_backward, device, forward=False)

        # 合并前向和后向的隐状态
        h_combined = torch.cat((h_forward, h_backward), dim=1)

        # Dropout和输出层
        h_combined = self.dropout_layer(h_combined)
        output = self.out(h_combined)

        return output