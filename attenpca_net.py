"""AttenPCA-Net: 2D CNN over a topographic map of temporal principal components.

Input shape is ``(batch, n_components, height, width)``. The component axis
is the convolution channel axis. Two 1 x 1 expansion blocks map any number of
components to 32 channels. Four 3 x 3 blocks then read the 10-20 grid.
Squeeze-and-excitation recalibrates channels inside every block, and a
spatial-attention map is applied before the classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainer import TorchClassifier


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention over the electrode grid."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return x * attention


class SqueezeExcitation(nn.Module):
    """Channel attention. ``reduction`` follows the SENet bottleneck."""

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.shape
        pooled = x.mean(dim=[2, 3])
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * weights.view(batch, channels, 1, 1)


class MultiHeadSelfAttention(nn.Module):
    """Spatial self-attention. Constructed with the network, not used in forward."""

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.view(batch, channels, -1).permute(0, 2, 1)
        tokens, _ = self.mhsa(tokens, tokens, tokens)
        return tokens.permute(0, 2, 1).view(batch, channels, height, width)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            SqueezeExcitation(out_channels),
            nn.Dropout(0.3),
        )
        if in_channels != out_channels:
            self.residual_transform = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.residual_transform = nn.Identity()

    def forward(self, x):
        return self.conv_block(x)


class AttenPCANet(nn.Module):
    """Score one topographic sample as non-P300 versus P300.

    ``frames`` is the number of retained principal components. ``rows`` and
    ``cols`` are the 10-20 grid size, 11 x 11 for the full montage.
    """

    def __init__(self, frames, rows=11, cols=11):
        super(AttenPCANet, self).__init__()

        self.initial_layers = nn.Sequential(
            nn.Conv2d(frames, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            SqueezeExcitation(16),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            SqueezeExcitation(32),
        )

        self.conv_layers = nn.ModuleList()
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        for _ in range(4):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                    nn.BatchNorm2d(out_channels),
                    SqueezeExcitation(out_channels),
                    nn.Dropout(0.3),
                )
            )
            rows -= kernel_size - 1
            cols -= kernel_size - 1
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)

        # Present so the registered modules match the original network.
        # Neither is applied in ``forward``.
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_channels * rows * cols, 64)
        self.fc2 = nn.Linear(64, 2)
        self.mhsa = MultiHeadSelfAttention(embed_dim=in_channels, num_heads=8)
        self.spa = SpatialAttention()

    def forward(self, x):
        x = self.initial_layers(x)
        for layer in self.conv_layers:
            x = F.elu(layer(x))
        x = self.spa(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x


class AttenPCANetTrainer(TorchClassifier):
    def get_model(self, **kwargs) -> nn.Module:
        return AttenPCANet(**kwargs)

    def get_loss(self):
        weights = [1.0, 1.0]
        return nn.CrossEntropyLoss(torch.tensor(weights)).to(self.device)

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def set_patience(self):
        return 30

    def set_batch_size(self):
        return 128


if __name__ == "__main__":
    segment_size = 25
    row = 11
    col = 11
    sample = torch.randn(1, segment_size, row, col)
    network = AttenPCANet(frames=segment_size, rows=row, cols=col)
    print(tuple(network(sample).shape))
