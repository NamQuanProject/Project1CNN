"""CNN model builders (PyTorch) for multi-label digit classification.

All models return raw logits (no sigmoid) -- pair with
nn.BCEWithLogitsLoss for numerically stable training, and apply
torch.sigmoid() to the output at inference time.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Baseline CNN, matching the architecture in cnn.ipynb (TF/Keras version):
    3x (conv + relu + maxpool), then dense(256) + dropout(0.3) + dense(10).
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced = input_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    """Improved CNN: BatchNorm, an extra conv block, global average pooling,
    and heavier dropout compared to the baseline. Starting point for the
    "propose improvements" part of the lab exercise -- tune further as needed.
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=64):
        super().__init__()
        filters = (32, 64, 128, 128)
        blocks = []
        c = in_channels
        for f in filters:
            blocks += [
                nn.Conv2d(c, f, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c = f
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Basic residual block: (Conv -> BN -> ReLU) x2 with a skip connection.

    Downsamples by stride 2 in the first conv when `downsample=True`; the
    shortcut path uses a matching 1x1 stride-2 conv + BN so it can be added
    to the main path.
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetCNN(nn.Module):
    """Improvement 2: replaces the plain conv stack with residual blocks.

    - A 3x3 stem conv (no downsampling), then 3 residual stages with
      filters 32 -> 64 -> 128, each downsampling by 2 (64x64 -> 8x8, same
      total reduction as the baseline's 3 maxpools).
    - Head: SpatialDropout (nn.Dropout2d) -> GlobalAveragePooling ->
      Dropout(0.35) -> Linear(10). No Flatten/Dense(256): the baseline's
      ~2.1M dense-layer parameters were the main source of overfitting, so
      this head is deliberately much smaller.
    - Pair with AdamW(lr=3e-4, weight_decay=1e-4) and augmentation that
      avoids rotation/flip (translate + zoom + contrast only), since
      rotating/flipping digits can turn a 6 into a 9 or vice versa.
    """

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32, 32, downsample=True)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)

        self.spatial_dropout = nn.Dropout2d(0.15)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spatial_dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


MODEL_BUILDERS = {
    "baseline": BaselineCNN,
    "improved": ImprovedCNN,
    "resnet": ResNetCNN,
}

# Recommended hyperparameters per model, applied in train.py when the
# corresponding CLI flag is left at its default (None -> "use the model's
# recommendation"). Explicit --lr / --optimizer / --weight-decay always win.
MODEL_HPARAM_DEFAULTS = {
    "resnet": {"lr": 3e-4, "optimizer": "adamw", "weight_decay": 1e-4},
}
