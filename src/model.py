"""CNN model builders (PyTorch) for multi-label digit classification.

All models return raw logits (no sigmoid) -- pair with
nn.BCEWithLogitsLoss for numerically stable training, and apply
torch.sigmoid() to the output at inference time.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


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


def _load_resnet50_backbone(pretrained):
    if not pretrained:
        return tv_models.resnet50(weights=None)
    try:
        return tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception as exc:  # pragma: no cover - depends on network access
        print(
            f"WARNING: could not download ImageNet-pretrained ResNet-50 weights ({exc}). "
            "Falling back to random initialization."
        )
        return tv_models.resnet50(weights=None)


class ResNet50CNN(nn.Module):
    """Improvement 3: a real torchvision ResNet-50 backbone, adapted for
    small (64x64, 1-channel) multi-label digit images.

    Adaptations vs. the stock ImageNet ResNet-50:
    - conv1: 3x3 stride-1 (instead of 7x7 stride-2), and no initial maxpool.
      ImageNet's aggressive 4x early downsampling is tuned for 224x224 input;
      on our 64x64 canvas it would collapse to a 2x2 feature map before the
      residual stages even start, destroying the fine detail needed to tell
      overlapping digits apart. Removing it keeps an 8x8x2048 feature map
      entering the final stage instead (a standard adaptation used for
      CIFAR-/small-image ResNets).
    - fc: replaced with Linear(2048, num_classes); the model returns raw
      logits (no sigmoid), matching the other models here.
    - Optional ImageNet-pretrained weights (`pretrained=True`, default) for
      layer1-4. The stem and fc are always freshly initialized (their shapes
      changed), but the pretrained mid/high-level conv filters in the
      residual stages are still a much better starting point than random
      init, even for grayscale synthetic digits -- this is why train.py
      trains the backbone at a lower learning rate than the new stem/fc
      (see `head_and_backbone_named_parameters` and
      `MODEL_HPARAM_DEFAULTS["resnet50"]["backbone_lr"]`).
    """

    def __init__(self, in_channels=1, num_classes=10, pretrained=True):
        super().__init__()
        backbone = _load_resnet50_backbone(pretrained)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def head_and_backbone_named_parameters(self):
        """Split named parameters into the freshly-initialized "head"
        (stem conv1 + final fc, both shape-changed from ImageNet) and the
        pretrained "backbone" (layer1-4), so train.py can give them
        different learning rates.
        """
        head_prefixes = ("backbone.conv1.", "backbone.fc.")
        head, backbone = [], []
        for name, param in self.named_parameters():
            (head if name.startswith(head_prefixes) else backbone).append((name, param))
        return head, backbone


MODEL_BUILDERS = {
    "baseline": BaselineCNN,
    "improved": ImprovedCNN,
    "resnet": ResNetCNN,
    "resnet50": ResNet50CNN,
}


def build_model(model_name, pretrained=True):
    """Construct a model by name. `pretrained` only affects resnet50 (whether
    to load ImageNet weights for the backbone); it's ignored for other models.
    """
    if model_name == "resnet50":
        return ResNet50CNN(pretrained=pretrained)
    return MODEL_BUILDERS[model_name]()


def split_decay_params(named_params):
    """Split named parameters into (decay, no_decay).

    Standard best practice: don't apply weight decay to 1D parameters
    (BatchNorm weight/bias, and Linear/Conv biases) -- decaying those hurts
    normalization behavior and gives no regularization benefit.
    """
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_param_groups(model, lr, weight_decay, backbone_lr=None):
    """Build optimizer param groups with decay/no-decay split, and (for
    models that define `head_and_backbone_named_parameters`, e.g. ResNet50CNN)
    a separate, lower learning rate for the pretrained backbone.
    """
    if backbone_lr is not None and hasattr(model, "head_and_backbone_named_parameters"):
        head_named, backbone_named = model.head_and_backbone_named_parameters()
        param_sets = ((head_named, lr), (backbone_named, backbone_lr))
    else:
        param_sets = ((list(model.named_parameters()), lr),)

    groups = []
    for named_params, group_lr in param_sets:
        decay, no_decay = split_decay_params(named_params)
        if decay:
            groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
    return groups


# Recommended hyperparameters per model, applied in train.py when the
# corresponding CLI flag is left at its default (None -> "use the model's
# recommendation"). Explicit CLI flags always win.
MODEL_HPARAM_DEFAULTS = {
    "resnet": {
        "lr": 3e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
    },
    "resnet50": {
        "lr": 1e-3,  # head (new conv1 + fc) learning rate
        "backbone_lr": 1e-4,  # pretrained layer1-4, 10x lower so early updates don't wreck them
        "optimizer": "adamw",
        "weight_decay": 0.05,
        "batch_size": 256,
        "scheduler": "cosine_warmup",
        "warmup_epochs": 5,
        "ema_decay": 0.9998,
        "grad_clip_norm": 1.0,
        "patience": 10_000,  # effectively disabled: let the cosine schedule run to completion
        "ema": True,
        "pretrained": True,
    },
}
