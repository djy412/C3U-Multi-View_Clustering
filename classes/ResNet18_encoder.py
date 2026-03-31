# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import torch.nn as nn
import torch
import torchvision.models as models

class ResNet18EncoderFC(nn.Module):
    def __init__(self, latent_dim=512):
        super(ResNet18EncoderFC, self).__init__()
        resnet18 = models.resnet18()
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):      
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        z = self.fc(x)

        return z
#*****************************************************************************




