import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, dilation=dilation
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # residual connections
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # residual
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class TCN(nn.Module):
    def __init__(self, input_dim=1280, num_classes=502, num_channels=[256, 256, 256, 256], 
                 kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.tcn = nn.Sequential(*layers)
        
        # classifier
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, 512),  # *2 for max+avg pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        

    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # -> (batch, features, time)
        x = self.tcn(x)
        
        # temporal pooling
        max_pool = torch.max(x, dim=2)[0]
        avg_pool = torch.mean(x, dim=2)
        x = torch.cat([max_pool, avg_pool], dim=1)
        
        logits = self.fc(x) 
        
        return logits