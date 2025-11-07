import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiQRNNAtt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer=1, dropout=0,
                 embedding=None, numeric_input_dim=7, numeric_output_dim=4):
        super(BiQRNNAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, embedding_dim)
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 数值特征线性层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)
        # QRNN输入维度
        self.input_dim = embedding_dim + numeric_output_dim

        # QRNN前向和后向卷积层
        self.conv_forward = nn.Conv1d(self.input_dim, hidden_dim * 3, kernel_size=1)
        self.conv_backward = nn.Conv1d(self.input_dim, hidden_dim * 3, kernel_size=1)

        # 注意力机制参数
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        nn.init.xavier_uniform_(self.weight_Mu)

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, out_size)

    def attention_net(self, qrnn_output):
        # qrnn_output: [batch_size, seq_len, hidden_dim*2]
        attn_scores = torch.matmul(qrnn_output, self.weight_Mu)   # [batch, seq, 1]
        attn_weights = F.softmax(attn_scores, dim=1)              # [batch, seq, 1]
        context = torch.sum(qrnn_output * attn_weights, dim=1)    # [batch, hidden_dim*2]
        return context, attn_weights.squeeze(-1).detach().cpu().numpy()

    def _process_direction(self, conv_out, device, batch_size, forward=True):
        # conv_out: [seq_len, batch_size, hidden_dim * 3]
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)  # [seq_len, batch, hidden_dim]
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)
        seq_len = Z.size(0)
        # 顺序
        sequence_range = range(seq_len) if forward else range(seq_len - 1, -1, -1)
        # 收集为 [seq_len, batch, hidden_dim]
        hidden_states = []
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        for t in sequence_range:
            h = F[t] * h + (1 - F[t]) * Z[t]
            hidden_states.append(h)
        # stack顺序决定shape！你需要 [seq_len, batch, hidden_dim]
        hidden_states = torch.stack(hidden_states, dim=0)
        # 但你后面想要 [batch, seq_len, hidden_dim]，所以要 permute
        hidden_states = hidden_states.permute(1, 0, 2)  # [batch, seq_len, hidden_dim]
        return hidden_states

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        self.embedding = self.embedding.to(device)
        self.numeric_layer = self.numeric_layer.to(device)
        self.conv_forward = self.conv_forward.to(device)
        self.conv_backward = self.conv_backward.to(device)
        self.weight_Mu = self.weight_Mu.to(device)
        self.out = self.out.to(device)
        self.dropout_layer = self.dropout_layer.to(device)

        batch_size, seq_length, feature_dim = X.size()
        event_types = X[:, :, 0].long()         # [batch, seq]
        numeric_attrs = X[:, :, 1:].float()     # [batch, seq, num_feat]
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            raise ValueError(f"Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        # 事件类型embedding
        embedded_event = self.embedding(event_types)                   # [batch, seq, embedding_dim]
        # 数值特征linear
        transformed_numeric = self.numeric_layer(numeric_attrs)        # [batch, seq, numeric_output_dim]
        # 拼接
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)   # [batch, seq, input_dim]
        # 转为[batch, input_dim, seq]供Conv1d用
        combined_input = combined_input.permute(0, 2, 1)               # [batch, input_dim, seq]

        # QRNN双向卷积
        conv_out_forward = self.conv_forward(combined_input)           # [batch, hidden_dim*3, seq]
        conv_out_backward = self.conv_backward(combined_input)         # [batch, hidden_dim*3, seq]
        # 转换为 [seq, batch, hidden_dim*3] 以便后续处理
        conv_out_forward = conv_out_forward.permute(2, 0, 1)
        conv_out_backward = conv_out_backward.permute(2, 0, 1)

        # QRNN双向状态
        h_forward = self._process_direction(conv_out_forward, device, batch_size,
                                            forward=True)  # [batch, seq, hidden_dim]
        h_backward = self._process_direction(conv_out_backward, device, batch_size,
                                             forward=False)  # [batch, seq, hidden_dim]
        h_combined = torch.cat((h_forward, h_backward), dim=2)  # [batch, seq, hidden_dim*2]

        # 注意力机制
        context, attention = self.attention_net(h_combined)  # [batch, hidden_dim*2]
        output = self.out(context)  # [batch, out_size]

        return output