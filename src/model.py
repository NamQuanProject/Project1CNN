"""ResNet-style CNN for multi-label digit classification.

The model returns raw logits (no sigmoid) -- pair with nn.BCEWithLogitsLoss
(or the other losses in train.py) for numerically stable training, and apply
torch.sigmoid() to the output at inference time.
"""

import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020, "ECA-Net:
    Efficient Channel Attention for Deep Convolutional Neural Networks",
    arxiv.org/abs/1910.03151). Replaces both the earlier SE and CBAM
    channel gates here: SE's channels -> hidden -> channels bottleneck
    (dimensionality reduction) is, per the ECA paper's analysis, actually
    harmful to channel-attention quality, not just a compute-saving
    compromise. ECA instead runs a single lightweight 1D conv directly over
    the pooled per-channel descriptor -- each channel's gate depends on its
    k nearest channel neighbors (local cross-channel interaction) -- with
    far fewer parameters than SE's two dense layers and no bottleneck.

    Kernel size k is adaptive to channel count (odd, via the paper's
    formula k = |log2(C)/gamma + b/gamma|, gamma=2, b=1): e.g. k=3 for
    64 channels, k=5 for 128/256.

    (CBAM -- channel + spatial attention -- was also tried in this model
    and empirically underperformed on this dataset/training budget; ECA is
    simpler and cheaper than either SE or CBAM, and directly addresses the
    SE bottleneck the CBAM swap was originally trying to improve on.)
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1  # round to nearest odd kernel size
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * torch.sigmoid(y)


class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016): randomly drops the entire
    residual branch for some training samples, so the network can't rely on
    any single block always being present -- an ensembling-style
    regularizer. A no-op at eval time or when drop_prob == 0.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep_prob * mask


class ResidualBlock(nn.Module):
    """Residual block: (Conv -> BN -> SiLU) x2, optional ECA channel
    attention on the residual branch, optional stochastic depth (DropPath),
    then a skip connection.

    Downsamples by stride 2 in the first conv when `downsample=True`; the
    shortcut path uses a matching 1x1 stride-2 conv + BN so it can be added
    to the main path. Uses SiLU (x * sigmoid(x)) instead of ReLU: smooth and
    non-zero everywhere (no "dead unit" gradient collapse the way ReLU can
    have), which several modern CNN designs (EfficientNet and others) found
    to outperform ReLU slightly at negligible extra cost.
    """

    def __init__(self, in_channels, out_channels, downsample=False, use_eca=True, drop_path=0.0):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.eca = ECA(out_channels) if use_eca else nn.Identity()
        self.drop_path = DropPath(drop_path)

        self.shortcut = None
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        out = self.drop_path(out)
        return self.act(out + identity)


class ResNetCNN(nn.Module):
    """Residual CNN for multi-label digit classification.

    Upgraded recipe (widened/deepened vs. the earlier 6-block/32-64-128
    version, plus a channel-attention swap and a smoother activation):

    - Stem: a single 3x3 conv+BN+SiLU, no downsampling (stays at 64x64),
      now outputting 64 channels (was 32).
    - 12 residual blocks in 3 stages of 4, channels 64 -> 128 -> 256, only
      the first block of stage 2 and stage 3 downsamples (64x64 -> 32x32 ->
      16x16) -- same "don't over-downsample" philosophy as before (a
      "downsample every block" design would end at a much smaller map,
      hurting separation of several small, overlapping digits), just with
      more blocks per stage and more channels per stage. The 3
      same-resolution blocks per stage add nonlinear depth before handing
      off to the next (downsampling + channel-widening) stage.
    - Each residual block includes an ECA channel-attention gate (see
      `ECA`) and stochastic depth (see `DropPath`), with drop probability
      increasing with block depth (0 -> `max_drop_path` across the 12
      blocks) -- regularization to offset the substantially larger capacity
      of this configuration vs. the earlier one.
    - Head: SpatialDropout (nn.Dropout2d, 0.15) -> GlobalAveragePooling ->
      Dropout(0.35) -> Linear(256, 10). No Flatten/Dense(256-unit-MLP):
      avoids a large dense classifier, historically the main source of
      overfitting for this dataset size (~50K training images).
    - Conv weights use explicit Kaiming-normal ("he_normal") init
      (`nonlinearity="relu"` is used for the gain calculation since PyTorch
      has no dedicated SiLU entry; relu's gain is the standard stand-in for
      SiLU/Swish in practice, since both behave near-linearly for positive
      inputs).
    - Pair with AdamW, a cosine-annealed LR schedule (`--scheduler cosine`),
      MixUp (`--mixup-alpha`), and augmentation that avoids rotation/flip
      (translate + zoom + contrast only), since rotating/flipping digits
      can turn a 6 into a 9 or vice versa.
    """

    def __init__(self, in_channels=1, num_classes=10, use_eca=True, max_drop_path=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # (in_channels, out_channels, downsample) -- 3 stages of 4 blocks,
        # channels 64 -> 128 -> 256, downsampling only at each stage's
        # first block.
        block_specs = [
            (64, 64, False),
            (64, 64, False),
            (64, 64, False),
            (64, 64, False),
            (64, 128, True),
            (128, 128, False),
            (128, 128, False),
            (128, 128, False),
            (128, 256, True),
            (256, 256, False),
            (256, 256, False),
            (256, 256, False),
        ]
        n = len(block_specs)
        drop_probs = [max_drop_path * i / (n - 1) for i in range(n)]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(cin, cout, downsample=down, use_eca=use_eca, drop_path=dp)
                for (cin, cout, down), dp in zip(block_specs, drop_probs)
            ]
        )

        self.spatial_dropout = nn.Dropout2d(0.15)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(256, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.spatial_dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


MODEL_BUILDERS = {
    "resnet": ResNetCNN,
}


def build_model(model_name="resnet"):
    """Construct a model by name."""
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


def build_param_groups(model, lr, weight_decay):
    """Build optimizer param groups with a decay/no-decay split."""
    decay, no_decay = split_decay_params(model.named_parameters())
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


# Recommended hyperparameters, applied in train.py when the corresponding CLI
# flag is left at its default (None -> "use the recommended default").
# Explicit CLI flags always win.
MODEL_HPARAM_DEFAULTS = {
    "resnet": {
        "lr": 3e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        # Plain cosine annealing (torch.optim.lr_scheduler.CosineAnnealingLR,
        # T_max=epochs, eta_min=min_lr) instead of ReduceLROnPlateau: a
        # fixed, well-behaved decay for a training run of a known length.
        "scheduler": "cosine",
        "min_lr": 1e-6,
        # Because a fixed cosine schedule assumes running to completion,
        # early-stopping patience is generous (vs. the smaller model's 10)
        # so it doesn't cut the anneal short; this is also now a bigger
        # model (12 blocks, 64->128->256ch) that may need more epochs.
        "patience": 30,
        # Early-stopping / checkpoint-selection watches exact-match accuracy
        # directly (mode "max") instead of val_loss -- that's the metric that
        # actually matters for this task, and can improve even while val_loss
        # is flat or slightly rising.
        "monitor_metric": "exact_match_accuracy",
        # Extra strengthening:
        "ema": True,
        "ema_decay": 0.9995,
        "label_smoothing": 0.05,
        # MixUp (Zhang et al., 2018): blends pairs of training images and
        # linearly interpolates their multi-hot labels by the same factor.
        # 0.2 is the standard value from the paper; regularizes the larger
        # capacity of this configuration.
        "mixup_alpha": 0.2,
        # Asymmetric Loss (Ben-Baruch/Ridnik et al., ICCV 2021) instead of
        # plain BCE: handles the positive/negative imbalance in this task
        # (~6-8 of 10 digit slots present per image, so negatives outnumber
        # positives) via asymmetric focusing + probability-shifting for easy
        # negatives. See train.py:asymmetric_loss_with_logits. Paper-recommended
        # defaults (asl_gamma_neg=4, asl_gamma_pos=1, asl_clip=0.05) apply via
        # train.py's fallbacks; asl_weight=1.0 here means pure ASL, not
        # blended with BCE -- lower it (e.g. --asl-weight 0.5) to experiment
        # with an explicit ASL+BCE combination.
        "loss_type": "asl",
    },
}
