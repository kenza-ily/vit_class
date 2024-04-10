import torch
import torch.nn as nn
from torchvision.models import vit_b_32

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = vit_b_32(pretrained=True)
        
        # Freeze all layers in the pretrained model
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Replace the head with a new linear layer
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x
