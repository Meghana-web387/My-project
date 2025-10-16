# model.py

import torch
import torch.nn as nn
import torchvision.models as models

# Blood group mappings
blood_groups = {
    0: 'A+',
    1: 'A-', 
    2: 'B+',
    3: 'B-',
    4: 'AB+',
    5: 'AB-',
    6: 'O+',
    7: 'O-'
}

# Inverse mapping for predictions
inv_blood_groups = {v: k for k, v in blood_groups.items()}

class ResNetGray(nn.Module):
    """
    Modified ResNet-18 for grayscale fingerprint image classification
    """
    def __init__(self, num_classes=8):
        super(ResNetGray, self).__init__()
        
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify first layer to accept grayscale (1 channel) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)