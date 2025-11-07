import torch
import torch.nn as nn
from torch.autograd import Variable

class QRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(QRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 数值特征线性变换
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)

        # 卷积输入维度 = 嵌入 + 数值
        conv_input_dim = embedding_dim + numeric_output_dim
        self.conv = nn.Conv1d(conv_input_dim, hidden_dim * 3, kernel_size=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_size)

    def _process_direction(self, conv_out, device):
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        seq_len, batch_size, _ = Z.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hs = []

        for t in range(seq_len):
            h = F[t] * h + (1 - F[t]) * Z[t]
            o = O[t] * h  # 用O门门控输出
            hs.append(o.unsqueeze(0))

        return torch.cat(hs, dim=0)  # shape: (seq_len, batch_size, hidden_dim)

    def forward(self, X):
        device = X.device
        batch_size, seq_length, feature_dim = X.size()

        # 分离嵌入 + 数值
        event_types = X[:, :, 0].long()
        numeric_attrs = X[:, :, 1:].float()

        embedded_event = self.embedding(event_types)
        transformed_numeric = self.numeric_layer(numeric_attrs)

        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)  # (B, L, D)
        combined_input = combined_input.permute(0, 2, 1)  # (B, D, L)

        # 卷积 + 转置 (B, 3H, L) → (L, B, 3H)
        conv_out = self.conv(combined_input).permute(2, 0, 1)

        # 前向 QRNN
        h_forward = self._process_direction(conv_out, device)

        # 取最后时刻的隐藏状态
        final_rep = h_forward[-1]  # (B, H)

        final_rep = self.dropout_layer(final_rep)
        output = self.out(final_rep)
        return output
