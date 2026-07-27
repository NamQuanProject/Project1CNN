"""CNN model builders (PyTorch) for multi-label digit classification.

Both models return raw logits (no sigmoid) -- pair with
nn.BCEWithLogitsLoss for numerically stable training, and apply
torch.sigmoid() to the output at inference time.
"""

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


MODEL_BUILDERS = {
    "baseline": BaselineCNN,
    "improved": ImprovedCNN,
}
