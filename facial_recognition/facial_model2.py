import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class Facial_model2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv2d = nn.Conv2d(1,3,3,stride = 1)
        
        self.pretrained = EfficientNet.from_pretrained('efficientnet-b4')
        self.FC = nn.Linear(1000,2)

    def forward(self,x):
        # x = F.relu(self.conv2d(x))
        x = F.relu(self.pretrained(x))
        x = self.FC(x)
        # x = torch.sigmoid(x)
        return x