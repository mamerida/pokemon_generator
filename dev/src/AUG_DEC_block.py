import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloques auxiliares
class AUG_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1):
        super(AUG_block, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.block(X)

class DEC_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, alpha=0.2):
        super(DEC_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True)
        )

    def forward(self, X):
        return self.block(X)