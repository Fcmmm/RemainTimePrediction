import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# 定义TCN中的一个卷积块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TCN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size,
                 batch_size=1, dropout=0, embedding=None,
                 kernel_size=2, numeric_input_dim=7, numeric_output_dim=4):
        super(TCN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim  # 在TCN中用作通道数的参考
        self.out_size = out_size
        self.batch_size = batch_size
        self.embedding = embedding
        self.dropout = dropout
        self.numeric_input_dim = numeric_input_dim
        self.numeric_output_dim = numeric_output_dim

        print('Initialization TCN Model')

        # 数值属性线性变换层
        self.numeric_layer = nn.Linear(numeric_input_dim, numeric_output_dim)

        # TCN层配置，使用hidden_dim作为通道数的基础
        # 设置TCN的通道数，通常使用递增的方式
        tcn_num_channels = [hidden_dim, hidden_dim, hidden_dim]

        # TCN层，输入维度为embedding_dim + numeric_output_dim
        self.tcn = TemporalConvNet(
            num_inputs=embedding_dim + numeric_output_dim,
            num_channels=tcn_num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 输出层
        self.out = nn.Linear(tcn_num_channels[-1], out_size)

    def forward(self, X):
        """

        X: [batch_size, seq_length, feature_dim]
        其中第一个特征是事件类型，其余是数值属性
        """
        batch_size, seq_length, feature_dim = X.size()

        # 分离事件类型和数值属性
        event_types = X[:, :, 0].long()  # 取第一个特征作为 event_type
        numeric_attrs = X[:, :, 1:].float()  # 其余特征作为数值属性

        # 添加维度检查
        actual_numeric_dim = numeric_attrs.size(-1)
        if actual_numeric_dim != self.numeric_layer.in_features:
            print(f"Warning: Expected {self.numeric_layer.in_features} numeric features, got {actual_numeric_dim}")

        # 事件类型嵌入
        embedded_event = self.embedding(event_types)  # [batch_size, seq_length, embedding_dim]

        # 对数值属性进行线性变换
        transformed_numeric = self.numeric_layer(numeric_attrs)  # [batch_size, seq_length, numeric_output_dim]

        # 拼接嵌入和数值属性
        combined_input = torch.cat((embedded_event, transformed_numeric), dim=-1)
        # [batch_size, seq_length, embedding_dim + numeric_output_dim]

        # 调整维度以适配TCN输入：从[batch_size, seq_length, features] 到 [batch_size, features, seq_length]
        combined_input = combined_input.permute(0, 2, 1)

        # TCN处理
        tcn_output = self.tcn(combined_input)  # [batch_size, tcn_num_channels[-1], seq_length]

        # 调整回[batch_size, seq_length, tcn_num_channels[-1]]以便进行最终输出
        tcn_output = tcn_output.permute(0, 2, 1)

        # 可以选择使用最后一个时间步的输出或者全局平均池化
        # 这里使用最后一个时间步的输出
        final_output = tcn_output[:, -1, :]  # [batch_size, tcn_num_channels[-1]]

        # 最终输出层
        output = self.out(final_output)  # [batch_size, out_size]

        return output
