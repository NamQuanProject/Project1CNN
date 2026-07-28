"""ResNet-style CNN for multi-label digit classification.

The model returns raw logits (no sigmoid) -- pair with nn.BCEWithLogitsLoss
(or the other losses in train.py) for numerically stable training, and apply
torch.sigmoid() to the output at inference time.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM channel attention (Woo et al., ECCV 2018, "CBAM: Convolutional
    Block Attention Module", arxiv.org/abs/1807.06521): like
    Squeeze-Excitation, a global-context gate that reweights each channel by
    how useful it is for the current input -- but pools with BOTH average
    AND max (not just average like SE), feeding both descriptors through the
    same shared MLP before summing. The paper's ablations found this
    "finer" than SE's average-only channel descriptor: max-pool captures
    which channels have a strong, spatially-localized activation somewhere
    in the image (e.g. a channel that fires strongly on one specific digit's
    stroke), which plain average-pooling dilutes away.

        Mc(F) = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """CBAM spatial attention (Woo et al., ECCV 2018): pools *across the
    channel axis* (not spatially) with both average and max, producing two
    single-channel H x W maps that summarize "how active is this pixel
    location, on average / at its most active channel". These are
    concatenated and passed through a 7x7 conv + sigmoid to produce a
    per-pixel gate. Channel attention answers "which feature channels
    matter"; this answers "which pixel locations matter" -- complementary,
    and specifically useful here since the digits that matter occupy
    varying, specific regions of the 64x64 canvas among overlapping
    distractors, information channel-only attention (SE, or CBAM's own
    channel module) cannot represent.

        Ms(F) = sigmoid(conv7x7([AvgPool_channel(F); MaxPool_channel(F)]))

    Kernel size 7x7 is the paper's recommended default (their ablation found
    it outperforms 3x3 -- a larger receptive field helps localize which
    region the current feature map should attend to).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018,
    arxiv.org/abs/1807.06521): sequentially applies channel attention (see
    `ChannelAttention`) then spatial attention (see `SpatialAttention`) --
    the paper found sequential channel-then-spatial to outperform parallel
    or spatial-then-channel arrangements.

        F'  = Mc(F) (x) F
        F'' = Ms(F') (x) F'
    """

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


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
    """Residual block: (Conv -> BN -> ReLU) x2, optional CBAM (channel +
    spatial attention, see `CBAM`) on the residual branch, optional
    stochastic depth (DropPath), then a skip connection.

    Downsamples by stride 2 in the first conv when `downsample=True`; the
    shortcut path uses a matching 1x1 stride-2 conv + BN so it can be added
    to the main path.
    """

    def __init__(
        self, in_channels, out_channels, downsample=False, use_cbam=True, cbam_reduction=16, drop_path=0.0
    ):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels, reduction=cbam_reduction) if use_cbam else nn.Identity()
        self.drop_path = DropPath(drop_path)

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
        out = self.cbam(out)
        out = self.drop_path(out)
        return self.relu(out + identity)


class ResNetCNN(nn.Module):
    """Residual CNN for multi-label digit classification.

    Block topology matches a reference implementation (a classmate's
    `assignment1_cnn.py`) that reached ~80% exact-match accuracy on this
    dataset, plus further strengthening techniques layered on top (CBAM
    channel+spatial attention, stochastic depth):

    - Stem: a single 3x3 conv+BN+ReLU, no downsampling (stays at 64x64).
    - 6 residual blocks: res1 (32ch, stride 1) -> res2 (64ch, stride 2) ->
      res3 (64ch, stride 1) -> res4 (128ch, stride 2) -> res5/res6 (128ch,
      stride 1). Only 2 of the 6 blocks downsample (64x64 -> 32x32 ->
      16x16) -- keeping more spatial detail into the final feature map than
      a "downsample every block" design (which would end at 8x8). That
      matters for separating several small, overlapping digits. The
      "same-resolution" blocks (res1/res3/res5/res6) add extra nonlinear
      depth at each scale instead of immediately discarding resolution.
    - Each residual block includes a CBAM gate (see `CBAM`: channel
      attention -- "which feature channels matter" -- followed by spatial
      attention -- "which pixel locations matter", the latter something a
      channel-only gate like plain Squeeze-Excitation cannot represent, and
      directly relevant here since the digits that matter occupy varying,
      specific regions among overlapping distractors) and stochastic depth
      (see `DropPath`), with drop probability increasing with depth (0 ->
      `max_drop_path` across the blocks, deeper blocks being more
      overfitting-prone) -- regularization to offset the extra capacity
      from the added blocks/CBAM gates.
    - Head: SpatialDropout (nn.Dropout2d, 0.15) -> GlobalAveragePooling ->
      Dropout(0.35) -> Linear(10). No Flatten/Dense(256): avoids a large
      dense classifier, historically the main source of overfitting for
      this dataset size (~50K training images).
    - Conv weights use explicit Kaiming-normal ("he_normal") init, matching
      the reference implementation and standard practice for ReLU networks
      (PyTorch's default conv init is a Kaiming *uniform* variant that isn't
      as well suited to deep ReLU stacks).
    - Pair with AdamW(lr=3e-4, weight_decay=1e-4) and augmentation that
      avoids rotation/flip (translate + zoom + contrast only), since
      rotating/flipping digits can turn a 6 into a 9 or vice versa.
    """

    def __init__(self, in_channels=1, num_classes=10, use_cbam=True, max_drop_path=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # (in_channels, out_channels, downsample)
        block_specs = [
            (32, 32, False),
            (32, 64, True),
            (64, 64, False),
            (64, 128, True),
            (128, 128, False),
            (128, 128, False),
        ]
        n = len(block_specs)
        drop_probs = [max_drop_path * i / (n - 1) for i in range(n)]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(cin, cout, downsample=down, use_cbam=use_cbam, drop_path=dp)
                for (cin, cout, down), dp in zip(block_specs, drop_probs)
            ]
        )

        self.spatial_dropout = nn.Dropout2d(0.15)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(128, num_classes)

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
        # Matches the reference recipe (assignment1_cnn.py, ~80% exact-match):
        # LR plateau-reduction is patient (3 epochs) and reaches a lower floor
        # than early-stopping patience (10 epochs) -- the two are deliberately
        # decoupled, see --lr-patience / --min-lr / --patience in train.py.
        "lr_patience": 3,
        "min_lr": 1e-6,
        "patience": 10,
        # Early-stopping / checkpoint-selection watches exact-match accuracy
        # directly (mode "max") instead of val_loss -- that's the metric that
        # actually matters for this task, and can improve even while val_loss
        # is flat or slightly rising.
        "monitor_metric": "exact_match_accuracy",
        # Extra strengthening beyond the reference recipe:
        "ema": True,
        "ema_decay": 0.9995,
        "label_smoothing": 0.05,
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
