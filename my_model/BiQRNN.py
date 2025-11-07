import torch
import torch.nn as nn
from torch.autograd import Variable

class BiQRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(BiQRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 数值特征处理层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)

        # 卷积输入维度更新
        conv_input_dim = embedding_dim + numeric_output_dim

        self.conv_forward = nn.Conv1d(conv_input_dim, hidden_dim * 3, kernel_size=1)
        self.conv_backward = nn.Conv1d(conv_input_dim, hidden_dim * 3, kernel_size=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, out_size)

    def _process_direction(self, conv_out, device, forward=True):
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        seq_len, batch_size, _ = Z.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hs = []

        # 前向或反向处理
        indices = range(seq_len) if forward else reversed(range(seq_len))
        for t in indices:
            h = F[t] * h + (1 - F[t]) * Z[t]
            hs.append(h.unsqueeze(0))

        if not forward:
            hs.reverse()

        # 合并为 (seq_len, batch_size, hidden_dim)
        return torch.cat(hs, dim=0)

    def forward(self, X):
        device = X.device
        batch_size, seq_length, feature_dim = X.size()

        # 拆分嵌入 + 数值特征
        event_types = X[:, :, 0].long()
        numeric_attrs = X[:, :, 1:].float()

        # 嵌入与数值特征拼接
        embedded_event = self.embedding(event_types)
        transformed_numeric = self.numeric_layer(numeric_attrs)
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)  # (B, L, D)

        # 转为卷积格式 (B, D, L)
        combined_input = combined_input.permute(0, 2, 1)

        # 卷积 (B, D, L) → (B, 3H, L) → (L, B, 3H)
        conv_out_forward = self.conv_forward(combined_input).permute(2, 0, 1)
        conv_out_backward = self.conv_backward(combined_input).permute(2, 0, 1)

        # 处理前向与后向
        h_forward = self._process_direction(conv_out_forward, device, forward=True)
        h_backward = self._process_direction(conv_out_backward, device, forward=False)

        # 合并 (L, B, 2H)
        h_combined = torch.cat((h_forward, h_backward), dim=2)

        # 提取最后一个时刻的隐藏状态（也可做 mean pooling）
        final_rep = h_combined[-1]  # shape: (batch_size, 2H)

        # Dropout + 输出
        final_rep = self.dropout_layer(final_rep)
        output = self.out(final_rep)

        return output
