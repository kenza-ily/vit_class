import torch.nn as nn
from torchvision.models import vit_b_32

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = vit_b_32(pretrained=True)
        self.fc = nn.Linear(self.vit.heads.head.out_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x