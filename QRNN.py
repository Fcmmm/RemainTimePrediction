import torch
import torch.nn as nn
from torch.autograd import Variable

class QRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0, embedding=None):
        super(QRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        print(vocab_size)
        print(embedding_dim)
        print(hidden_dim)
        print(out_size)
        print(embedding)
        print(batch_size)
        print(n_layer)
        print(dropout)

        # 使用卷积层代替LSTM中的输入处理
        self.conv = nn.Conv1d(embedding_dim, hidden_dim * 3, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_size)

    def forward(self, X):
        # 确保所有张量都在同一设备上
        device = X.device

        # 嵌入层
        embedded = self.embedding(X).permute(0, 2, 1)  # 转换为 (batch_size, embedding_dim, sequence_length)

        # 卷积操作
        conv_out = self.conv(embedded)
        conv_out = conv_out.permute(2, 0, 1)  # 转换为 (sequence_length, batch_size, hidden_dim * 3)

        # 分割到Z, F, O门
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        # 初始化隐状态
        h = torch.zeros(self.batch_size, self.hidden_dim, device=device)
        for t in range(Z.size(0)):
            h = F[t] * h + (1 - F[t]) * Z[t]

        # 输出层
        h = self.dropout_layer(h)
        output = self.out(h)
        return output