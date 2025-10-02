import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
