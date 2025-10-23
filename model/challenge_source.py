"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class ChallengeSource(nn.Module):
    def __init__(self, dropout_conv=0.4, dropout_fc=0.4):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(9,9), stride=(2,2), padding=4)  # ??? padding = 2  stride = (2,2)??
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization after the first conv layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(9,9), stride=(2,2), padding=4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(9,9), stride=(2,2), padding=4)
        self.bn3 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 2 * 2, 20)
        self.fc2 = nn.Linear(20, 8)

        # Define dropout layers
        self.dropout_conv = nn.Dropout(dropout_conv)
        self.dropout_fc = nn.Dropout(dropout_fc)

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            ## TODO: initialize the parameters for the convolutional layers
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1.0 / (9 * 9 * conv.in_channels)))
            nn.init.constant_(conv.bias, 0.0)
        
        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc1.weight, mean=0.0, std=sqrt(1.0 / (self.fc1.in_features)))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=sqrt(1.0 / (self.fc2.in_features)))
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass
        # print("Before the convolutional:", x.shape)
        # Hint: printing out x.shape after each layer could be helpful!
        ## TODO: forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        # print("After conv1 and pool:", x.shape)  # Debugging output
        
        # Convolutional Layer 2 + ReLU + Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        # print("After conv2 and pool:", x.shape)  # Debugging output
        
        # Convolutional Layer 3 + ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        # print("After conv3 and pool:", x.shape)  # Debugging output
        
        # Flatten the output for the fully connected layer
        x = x.flatten(start_dim=1)
        # print("After flattening:", x.shape)  # Debugging output
        x = self.fc1(x)
        x = self.dropout_fc(x)

        x = self.fc2(x)
        # print("after the FC:", x.shape)
        return x


