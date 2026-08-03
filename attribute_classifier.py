"""Frozen ANN attribute classifier used only as training guidance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                      nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class AttributeClassifier(nn.Module):
    def __init__(self, num_attributes=40):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
                                    ResidualBlock(64, 128, 2), ResidualBlock(128, 256, 2))
        self.head = nn.Linear(256, num_attributes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load_frozen_classifier(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = AttributeClassifier(len(checkpoint['attr_names'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint
