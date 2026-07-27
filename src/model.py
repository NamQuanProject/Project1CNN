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


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018): a cheap
    global-context gate that reweights each channel by how useful it is for
    the current input, instead of treating every channel equally. Lets the
    network suppress, e.g., channels that mostly respond to a distractor
    digit rather than the ones that matter for this particular image.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


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
    """Residual block: (Conv -> BN -> ReLU) x2, optional Squeeze-Excitation
    on the residual branch, optional stochastic depth (DropPath), then a
    skip connection.

    Downsamples by stride 2 in the first conv when `downsample=True`; the
    shortcut path uses a matching 1x1 stride-2 conv + BN so it can be added
    to the main path.
    """

    def __init__(self, in_channels, out_channels, downsample=False, use_se=True, se_reduction=8, drop_path=0.0):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_channels, reduction=se_reduction) if use_se else nn.Identity()
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
        out = self.se(out)
        out = self.drop_path(out)
        return self.relu(out + identity)


class ResNetCNN(nn.Module):
    """Improvement 2: replaces the plain conv stack with residual blocks,
    matching the 5-block topology + He-normal init from a reference
    implementation that reached ~80% exact-match accuracy on this dataset
    (a classmate's `assignment1_cnn.py`), plus two further strengthening
    techniques layered on top (Squeeze-Excitation attention, stochastic
    depth):

    - Stem: a single 3x3 conv+BN+ReLU, no downsampling (stays at 64x64).
    - 5 residual blocks: res1 (32ch, stride 1) -> res2 (64ch, stride 2) ->
      res3 (64ch, stride 1) -> res4 (128ch, stride 2) -> res5 (128ch,
      stride 1). Only 2 of the 5 blocks downsample (64x64 -> 32x32 ->
      16x16) -- keeping more spatial detail into the final feature map than
      a "downsample every block" design (the earlier 3-block version here,
      which ended at 8x8). That matters for separating several small,
      overlapping digits. The "same-resolution" blocks (res1/res3/res5) add
      extra nonlinear depth at each scale instead of immediately discarding
      resolution.
    - Each residual block includes a Squeeze-and-Excitation gate (see
      `SqueezeExcite`) and stochastic depth (see `DropPath`), with drop
      probability increasing with depth (0 -> `max_drop_path` across the 5
      blocks, deeper blocks being more overfitting-prone) -- regularization
      to offset the extra capacity from the added blocks/SE gates.
    - Head: SpatialDropout (nn.Dropout2d, 0.15) -> GlobalAveragePooling ->
      Dropout(0.35) -> Linear(10). No Flatten/Dense(256): avoids the
      baseline's ~2.1M-parameter dense classifier, its main overfitting
      driver.
    - Conv weights use explicit Kaiming-normal ("he_normal") init, matching
      the reference implementation and standard practice for ReLU networks
      (PyTorch's default conv init is a Kaiming *uniform* variant that isn't
      as well suited to deep ReLU stacks).
    - Pair with AdamW(lr=3e-4, weight_decay=1e-4) and augmentation that
      avoids rotation/flip (translate + zoom + contrast only), since
      rotating/flipping digits can turn a 6 into a 9 or vice versa.
    """

    def __init__(self, in_channels=1, num_classes=10, use_se=True, max_drop_path=0.1):
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
        ]
        n = len(block_specs)
        drop_probs = [max_drop_path * i / (n - 1) for i in range(n)]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(cin, cout, downsample=down, use_se=use_se, drop_path=dp)
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


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2, the standard U-Net conv block. Same-padding
    (padding=1) throughout, so spatial size only changes via pool/upsample --
    no cropping needed for skip connections, unlike the original U-Net paper.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetCNN(nn.Module):
    """Improvement 4: a small U-Net, repurposed for multi-label
    classification instead of its usual per-pixel segmentation use.

    - Encoder: 3x (DoubleConv + maxpool), channels base -> base*2 -> base*4,
      then a DoubleConv bottleneck at base*8 (64x64 -> 8x8, same total
      reduction as the other models here).
    - Decoder: 3x (ConvTranspose2d upsample -> concat matching encoder skip
      -> DoubleConv), mirroring the encoder back up to full 64x64 resolution
      at `base` channels.
    - Classification head: GlobalAveragePooling -> Dropout -> Linear(10),
      instead of U-Net's usual 1x1-conv-per-pixel segmentation head. The
      motivation for trying U-Net here is that its skip connections keep
      full-resolution detail available late in the network, which can help
      separate small overlapping digits that a plain encoder would blur
      away by the time it reaches the bottleneck.
    - Kept deliberately small (base=16, ~480K total params): ResNet-50's
      23.5M params badly overfit this 50k-image dataset, so this stays much
      closer to the custom `resnet` model's parameter budget (~688K) rather
      than repeating that mistake.
    """

    def __init__(self, in_channels=1, num_classes=10, base=16, dropout=0.3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.pool_out = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base, num_classes)

    def forward(self, x):
        e1 = self.enc1(x)  # base,   64x64
        e2 = self.enc2(self.pool(e1))  # base*2, 32x32
        e3 = self.enc3(self.pool(e2))  # base*4, 16x16
        b = self.bottleneck(self.pool(e3))  # base*8, 8x8

        d3 = self.up3(b)  # base*4, 16x16
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)  # base*2, 32x32
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)  # base,   64x64
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.pool_out(d1)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)


MODEL_BUILDERS = {
    "baseline": BaselineCNN,
    "improved": ImprovedCNN,
    "resnet": ResNetCNN,
    "resnet50": ResNet50CNN,
    "unet": UNetCNN,
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
    },
    "unet": {
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
