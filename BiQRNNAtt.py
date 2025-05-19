import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiQRNNAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(BiQRNNAtt, self).__init__()
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

        # 注意力机制的参数
        self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim * 2, hidden_dim * 2))
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, n_layer))

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, out_size)

        # 初始化注意力权重
        nn.init.xavier_uniform_(self.weight_W)
        nn.init.xavier_uniform_(self.weight_Mu)

    def attention_net(self, qrnn_output):
        """
        计算注意力权重
        qrnn_output: [batch_size, seq_len, hidden_dim * 2]
        """
        attn_weights = torch.matmul(qrnn_output, self.weight_Mu)  # [batch_size, seq_len, n_layer]
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len, n_layer]

        # 计算上下文向量
        context = torch.bmm(qrnn_output.transpose(1, 2), soft_attn_weights).squeeze(2)

        return context, soft_attn_weights.detach().cpu().numpy()

    def _process_direction(self, conv_out, device, forward=True):
        # 分割到Z, F, O门
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        # 初始化隐状态
        hidden_states = []
        h = torch.zeros(self.batch_size, self.hidden_dim, device=device)

        # 序列长度
        seq_len = Z.size(0)

        # 根据方向决定处理顺序
        if forward:
            sequence_range = range(seq_len)
        else:
            sequence_range = range(seq_len - 1, -1, -1)

        # 收集所有时间步的隐状态
        for t in sequence_range:
            h = F[t] * h + (1 - F[t]) * Z[t]
            hidden_states.append(h)

        # 将隐状态堆叠成序列
        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states

    def forward(self, X):
        device = X.device
        X = X.to(device)
        self.weight_W = self.weight_W.to(device)
        self.weight_Mu = self.weight_Mu.to(device)

        # 嵌入层
        embedded = self.embedding(X).permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

        # 前向和后向的卷积操作
        conv_out_forward = self.conv_forward(embedded)
        conv_out_backward = self.conv_backward(embedded)

        # 转换维度为 [seq_len, batch_size, hidden_dim * 3]
        conv_out_forward = conv_out_forward.permute(2, 0, 1)
        conv_out_backward = conv_out_backward.permute(2, 0, 1)

        # 分别处理前向和后向，获取所有时间步的隐状态
        h_forward = self._process_direction(conv_out_forward, device, forward=True)
        h_backward = self._process_direction(conv_out_backward, device, forward=False)

        # 合并前向和后向的隐状态 [batch_size, seq_len, hidden_dim * 2]
        h_combined = torch.cat((h_forward, h_backward), dim=2)

        # 应用注意力机制
        context, attention = self.attention_net(h_combined)

        # Dropout和输出层
        context = self.dropout_layer(context)
        output = self.out(context)

        return output  # 返回输出和注意力权重