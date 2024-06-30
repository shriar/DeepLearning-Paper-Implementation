# - masking is not done.

import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
import torch.nn.functional as F

import math


class CNN(nn.Module):
    def __init__(self, cnn_unit1, cnn_unit2, cnn_unit3, kernel_size, dropout, pool_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(16, cnn_unit1, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(cnn_unit1, cnn_unit2, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool2 = nn.MaxPool1d(pool_size)
        
        self.conv3 = nn.Conv1d(cnn_unit2, cnn_unit3, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool3 = nn.MaxPool1d(pool_size)
        
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(96)

    def forward(self, x):        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc(x)
        
        return x
    
class CNNGateGenerator(nn.Module):
    def __init__(self, input_shape, cnn_unit1, cnn_unit2, cnn_unit3, cnn_unit4, kernel_size, dropout, pool_size, masked_value, gate_min, gate_max):
        super(CNNGateGenerator, self).__init__()
        
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=cnn_unit1, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=cnn_unit1, out_channels=cnn_unit2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels=cnn_unit2, out_channels=cnn_unit3, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv1d(in_channels=cnn_unit3, out_channels=cnn_unit4, kernel_size=kernel_size, padding=kernel_size//2)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        
        self.gate_min = gate_min
        self.gate_max = gate_max
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (16, 96)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv4(x))
        
        x = torch.clamp(x, min=self.gate_min, max=self.gate_max)
        x = x.permute(0, 2, 1) # 
        return x