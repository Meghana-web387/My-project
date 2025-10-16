# model.py

import torch
import torch.nn as nn
from torchvision import models

class ResNetGray(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetGray, self).__init__()
        # Use weights=None as we'll load our own trained weights, not ImageNet's
        self.model = models.resnet18(weights=None)
        # Modify the first convolutional layer for 1 input channel (grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer for your 8 output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define label mapping and inverse mapping (optional, but good for clarity)
blood_groups = {'A+': 0, 'A-': 1, 'B+': 2, 'B-': 3, 'AB+': 4, 'AB-': 5, 'O+': 6, 'O-': 7}
inv_blood_groups = {v: k for k, v in blood_groups.items()}