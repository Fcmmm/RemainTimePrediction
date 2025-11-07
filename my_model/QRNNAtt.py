import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class QRNNAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None):
        super(QRNNAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 单向卷积层
        self.conv = nn.Conv1d(embedding_dim, hidden_dim * 3, kernel_size=1)

        # 注意力机制的参数
        self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim, hidden_dim))
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer))

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_size)

        # 初始化注意力权重
        nn.init.xavier_uniform_(self.weight_W)
        nn.init.xavier_uniform_(self.weight_Mu)

    def attention_net(self, qrnn_output):
        """
        计算注意力权重
        qrnn_output: [batch_size, seq_len, hidden_dim]
        """
        attn_weights = torch.matmul(qrnn_output, self.weight_Mu)  # [batch_size, seq_len, n_layer]
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len, n_layer]

        # 计算上下文向量
        context = torch.bmm(qrnn_output.transpose(1, 2), soft_attn_weights).squeeze(2)

        return context, soft_attn_weights.detach().cpu().numpy()

    def _process_sequence(self, conv_out, device):
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

        # 收集所有时间步的隐状态
        for t in range(seq_len):
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

        # 卷积操作
        conv_out = self.conv(embedded)

        # 转换维度为 [seq_len, batch_size, hidden_dim * 3]
        conv_out = conv_out.permute(2, 0, 1)

        # 处理序列，获取所有时间步的隐状态
        hidden_states = self._process_sequence(conv_out, device)

        # 应用注意力机制
        context, attention = self.attention_net(hidden_states)

        # Dropout和输出层
        context = self.dropout_layer(context)
        output = self.out(context)

        return output