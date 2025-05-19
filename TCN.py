import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt

# 扩张残差块
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=(dilation*(kernel_size-1)//2), dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=(dilation*(kernel_size-1)//2), dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)

        if residual.shape[2] != output.shape[2]:
            residual = F.pad(residual, (0, output.shape[2] - residual.shape[2]))

        if residual.shape[1] != output.shape[1]:
            residual = F.pad(residual, (0, 0, 0, output.shape[1] - residual.shape[1]))

        output = self.relu(output + residual)
        return output

class TCN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=64, n_layers=1, kernel_size=1,
                 dropout=0, embedding=None, dilations=[1, 2, 4, 8]):
        super(TCN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = dilations

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(DilatedResidualBlock(embedding_dim, embedding_dim, kernel_size, dilation))

        self.fc = nn.Linear(embedding_dim, out_size)  # 将 hidden_dim 改为 embedding_dim

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(0, 2, 1)

        output = input
        for conv_block in self.conv_blocks:
            output = conv_block(output)

        output = output.permute(0, 2, 1)
        output = torch.mean(output, dim=1)
        output = self.fc(output)

        return output
