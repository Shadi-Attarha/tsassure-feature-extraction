"""
Augmentation Ablation Study — Widar3.0 BVP Gesture Recognition
===============================================================

Tests whether augmentations improve accuracy when training on a
distilled subset of Widar BVP data.

All augmentations are user-provided (from your codebase).
No third-party or invented augmentations are added.

BVP tensor shape throughout: (B, T, H, W) = (B, 22, 20, 20)
  T = 22 time frames
  H = 20 x-velocity bins
  W = 20 y-velocity bins

Usage:
    # smoke test (verify script runs end-to-end, ~2 min)
    python test_augmentation_widar.py \
        --indices distilled_kmeans_ipc150.pt --smoke_test

    # full ablation
    python test_augmentation_widar.py \
        --indices distilled_kmeans_ipc150.pt --n_runs 5

    # specific configs only
    python test_augmentation_widar.py \
        --indices distilled_kmeans_ipc150.pt \
        --configs baseline flip_h flip_h+flip_w velocity_group

Output:
    aug_ablation_<tag>.csv          — raw per-run accuracies
    aug_ablation_<tag>_summary.txt  — ranked summary with DeltaBaseline
"""

import os
import csv
import argparse
import dataclasses
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_ROOT   = '/data/sattarha/Widardata2/'
NUM_CLASSES = 6
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Push&Pull', 'Sweep', 'Clap', 'Slide',
               'Draw-O(H)', 'Draw-Zigzag(H)']


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class Widar_Dataset(Dataset):
    def __init__(self, root_dir):
        self.data_list = glob.glob(root_dir + '/*/*.csv')
        self.folder    = sorted(glob.glob(root_dir + '/*/'))
        self.category  = {self.folder[i].split('/')[-2]: i
                          for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        y    = self.category[path.split('/')[-2]]
        x    = np.genfromtxt(path, delimiter=',')
        x    = (x - 0.0025) / 0.0119
        x    = x.reshape(22, 20, 20)
        x    = np.clip(x, -3, 3) / 3.0
        return torch.FloatTensor(x), y


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class Widar_CNN3D_Eval(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(1,   64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64,  128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(256 * 2 * 2 * 2, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                         # (B, 1, T, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def make_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# AUGMENTATION PARAMETERS
#
# All augmentations operate on (B, T, H, W) = (B, 22, 20, 20)
#   T = 22  time frames
#   H = 20  x-velocity bins
#   W = 20  y-velocity bins
# ─────────────────────────────────────────────────────────────
@dataclasses.dataclass
class AugParam:
    # velocity-space flips
    prob_flip_h:            float = 0.5
    prob_flip_w:            float = 0.5

    # velocity-space spatial transforms
    ratio_scale:            float = 1.2
    ratio_crop_pad:         float = 0.125
    ratio_cutout:           float = 0.2

    # temporal augmentations
    max_temporal_shift:     float = 3.0
    prob_temporal_flip:     float = 0.5
    ratio_temporal_cutout:  float = 0.2

    # amplitude augmentations
    amplitude:              float = 0.4
    noise:                  float = 0.05

    # DiffAug batch mode
    batchmode:              bool  = False


# ─────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS  (your code, verbatim)
# All expect and return shape (B, T, H, W)
# ─────────────────────────────────────────────────────────────
def set_seed_DiffAug(param):
    pass


def rand_flip_h(x, param):
    """
    Flip x-velocity axis (H, dim=2).
    Physical meaning: mirrors the gesture left <-> right.
    WARNING: disable if slide_left / slide_right are distinct classes.
    """
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode:
        randf[:] = randf[0]
    return torch.where(randf < param.prob_flip_h, x.flip(2), x)


def rand_flip_w(x, param):
    """
    Flip y-velocity axis (W, dim=3).
    Physical meaning: mirrors the gesture forward <-> backward.
    WARNING: check class symmetry before enabling.
    """
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode:
        randf[:] = randf[0]
    return torch.where(randf < param.prob_flip_w, x.flip(3), x)


def rand_scale_velocity(x, param):
    """
    Scale velocity bins via affine grid (H, W).
    Physical meaning: different body-to-antenna distances.
    T is treated as the channel dimension; H and W are scaled identically.
    """
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0,    0],
              [0,    sy[i], 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    return F.grid_sample(x, grid, align_corners=True)


def rand_translate_velocity(x, param):
    """
    Translate in velocity space (H, W) with clamp padding.
    Physical meaning: slightly different body orientation to antennas.
    """
    ratio   = param.ratio_crop_pad
    shift_h = int(x.size(2) * ratio + 0.5)
    shift_w = int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    trans_h = torch.randint(-shift_h, shift_h + 1,
                             size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    trans_w = torch.randint(-shift_w, shift_w + 1,
                             size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:
        trans_h[:] = trans_h[0]
        trans_w[:] = trans_w[0]

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_h = torch.clamp(grid_h + trans_h + 1, 0, x.size(2) + 1)
    grid_w = torch.clamp(grid_w + trans_w + 1, 0, x.size(3) + 1)
    x_pad  = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (x_pad.permute(0, 2, 3, 1).contiguous()
         [grid_b, grid_h, grid_w].permute(0, 3, 1, 2))
    return x


def rand_cutout_velocity(x, param):
    """
    Zero out a rectangular patch in velocity space (H, W) for all T frames.
    Physical meaning: partial antenna occlusion or dead velocity bins.
    """
    cutout_h = int(x.size(2) * param.ratio_cutout + 0.5)
    cutout_w = int(x.size(3) * param.ratio_cutout + 0.5)
    set_seed_DiffAug(param)
    off_h = torch.randint(0, x.size(2) + (1 - cutout_h % 2),
                          size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    off_w = torch.randint(0, x.size(3) + (1 - cutout_w % 2),
                          size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:
        off_h[:] = off_h[0]
        off_w[:] = off_w[0]

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(x.size(0),  dtype=torch.long, device=x.device),
        torch.arange(cutout_h,   dtype=torch.long, device=x.device),
        torch.arange(cutout_w,   dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_h = torch.clamp(grid_h + off_h - cutout_h // 2, 0, x.size(2) - 1)
    grid_w = torch.clamp(grid_w + off_w - cutout_w // 2, 0, x.size(3) - 1)

    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_b, grid_h, grid_w] = 0
    return x * mask.unsqueeze(1)


def rand_temporal_shift(x, param):
    """
    Continuous bilinear shift along T via affine_grid.
    Gradient flows smoothly back through interpolated frames.
    """
    B, T, H, W = x.shape
    set_seed_DiffAug(param)
    max_norm = param.max_temporal_shift / (T / 2.0)
    shifts   = (torch.rand(B, device=x.device) * 2 - 1) * max_norm
    if param.batchmode:
        shifts[:] = shifts[0]

    x_r   = x.permute(0, 2, 3, 1).reshape(B, H * W, 1, T)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = shifts

    grid      = F.affine_grid(theta, x_r.shape, align_corners=True)
    x_shifted = F.grid_sample(x_r, grid, mode='bilinear',
                               align_corners=True, padding_mode='border')
    return x_shifted.reshape(B, H, W, T).permute(0, 3, 1, 2)


def rand_temporal_flip(x, param):
    """
    Reverse the T axis (time-reverse the gesture).
    WARNING: only valid if class label is symmetric under time-reversal.
    """
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), device=x.device)
    if param.batchmode:
        randf[:] = randf[0]
    flipped = x.flip(1)
    mask    = (randf < param.prob_temporal_flip)[:, None, None, None]
    return torch.where(mask, flipped, x)


def rand_temporal_cutout(x, param):
    """
    Zero out a contiguous block of T frames.
    Physical meaning: missing CSI packets / person momentarily still.
    """
    T          = x.size(1)
    cutout_len = max(1, int(T * param.ratio_temporal_cutout))
    set_seed_DiffAug(param)
    offset = torch.randint(0, T - cutout_len + 1,
                           size=[x.size(0)], device=x.device)
    if param.batchmode:
        offset[:] = offset[0]

    mask = torch.ones(x.size(0), T, 1, 1, dtype=x.dtype, device=x.device)
    for i in range(x.size(0)):
        mask[i, offset[i]: offset[i] + cutout_len] = 0
    return x * mask


def rand_amplitude(x, param):
    """
    Multiply BVP map by a per-sample random scalar.
    Physical meaning: signal strength variation due to distance/environment.
    """
    set_seed_DiffAug(param)
    scale = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
             * param.amplitude + (1.0 - param.amplitude / 2))
    if param.batchmode:
        scale[:] = scale[0]
    return x * scale


def rand_noise(x, param):
    """
    Add per-sample Gaussian noise.
    Physical meaning: multipath interference and thermal noise in CSI.
    """
    set_seed_DiffAug(param)
    return x + param.noise * torch.randn_like(x)


# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────
AUGMENTATION_REGISTRY: Dict[str, tuple] = {
    #  name                    function                   warning (or None)
    'flip_h':            (rand_flip_h,            'check directional class symmetry (H axis)'),
    'flip_w':            (rand_flip_w,            'check directional class symmetry (W axis)'),
    'scale_velocity':    (rand_scale_velocity,    None),
    'translate_velocity':(rand_translate_velocity,None),
    'cutout_velocity':   (rand_cutout_velocity,   None),
    'temporal_shift':    (rand_temporal_shift,    None),
    'temporal_flip':     (rand_temporal_flip,     'check class symmetry under time-reversal'),
    'temporal_cutout':   (rand_temporal_cutout,   None),
    'amplitude':         (rand_amplitude,         None),
    'noise':             (rand_noise,             None),
}

# ─────────────────────────────────────────────────────────────
# CONFIGS — named augmentation combinations to ablate
# Add your own here, or pass via --configs
# ─────────────────────────────────────────────────────────────
CONFIGS: Dict[str, List[str]] = {
    # baseline
    'baseline':            [],

    # every augmentation individually
    'flip_h':              ['flip_h'],
    'flip_w':              ['flip_w'],
    'scale_velocity':      ['scale_velocity'],
    'translate_velocity':  ['translate_velocity'],
    'cutout_velocity':     ['cutout_velocity'],
    'temporal_shift':      ['temporal_shift'],
    'temporal_flip':       ['temporal_flip'],
    'temporal_cutout':     ['temporal_cutout'],
    'amplitude':           ['amplitude'],
    'noise':               ['noise'],

    # physically motivated groups
    'velocity_group':      ['flip_h', 'flip_w',
                            'scale_velocity', 'translate_velocity',
                            'cutout_velocity'],
    'temporal_group':      ['temporal_shift', 'temporal_cutout'],
    'signal_group':        ['amplitude', 'noise'],

    # conservative: lower-risk augmentations only
    'conservative':        ['flip_h', 'scale_velocity',
                            'temporal_shift', 'amplitude', 'noise'],

    # all augmentations
    'all':                 list(AUGMENTATION_REGISTRY.keys()),
}


def apply_augmentations(x: torch.Tensor,
                        aug_names: List[str],
                        param: AugParam) -> torch.Tensor:
    """Apply augmentations in listed order. Input/output: (B, T, H, W)."""
    for name in aug_names:
        fn, _ = AUGMENTATION_REGISTRY[name]
        x = fn(x, param)
    return x


# ─────────────────────────────────────────────────────────────
# DATASET WRAPPER
# Augmentation is applied on-the-fly in __getitem__ so each
# epoch sees independently sampled transforms.
# ─────────────────────────────────────────────────────────────
class AugmentedSubset(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor,
                 aug_names: List[str],
                 param: Optional[AugParam]):
        self.xs        = xs     # (N, T, H, W) = (N, 22, 20, 20)
        self.ys        = ys     # (N,)
        self.aug_names = aug_names
        self.param     = param

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]       # (T, H, W)
        y = self.ys[idx]
        if self.param is not None and self.aug_names:
            # unsqueeze batch dim for aug functions, squeeze back after
            x = apply_augmentations(
                x.unsqueeze(0), self.aug_names, self.param
            ).squeeze(0)
        return x, y


# ─────────────────────────────────────────────────────────────
# SHAPE PRINTING
# Called at every config so you can confirm dims never change.
# ─────────────────────────────────────────────────────────────
def print_tensor_shape(xs: torch.Tensor, ys: torch.Tensor,
                       label: str):
    per_class = {int(c): int((ys == c).sum())
                 for c in range(NUM_CLASSES)}
    print(f"  [{label}]")
    print(f"    shape      : {tuple(xs.shape)}  "
          f"(N x T x H x W = N x 22 x 20 x 20)")
    print(f"    dtype      : {xs.dtype}")
    print(f"    value range: [{xs.min():.3f}, {xs.max():.3f}]")
    print(f"    per class  : {per_class}")


def print_augmented_batch_shape(dist_x: torch.Tensor,
                                aug_names: List[str],
                                param: AugParam,
                                config_name: str,
                                device: str):
    """
    Run one batch of 8 samples through the augmentation pipeline
    and print input vs output shape + value range.
    Confirms aug does not accidentally change tensor dimensions.
    """
    if not aug_names:
        return
    sample = dist_x[:8].to(device)
    with torch.no_grad():
        augmented = apply_augmentations(sample, aug_names, param)
    print(f"  [Augmented sample batch — config: {config_name}]")
    print(f"    input  shape     : {tuple(sample.shape)}  (B x T x H x W)")
    print(f"    output shape     : {tuple(augmented.shape)}")
    print(f"    input  val range : [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"    output val range : [{augmented.min():.3f}, {augmented.max():.3f}]")
    if tuple(sample.shape) != tuple(augmented.shape):
        print(f"    WARNING: shape changed after augmentation!")
    else:
        print(f"    shape preserved: OK")


# ─────────────────────────────────────────────────────────────
# TRAINING + EVALUATION FOR ONE CONFIG
# ─────────────────────────────────────────────────────────────
def run_config(config_name: str,
               aug_names: List[str],
               aug_param: AugParam,
               dist_x: torch.Tensor,
               dist_y: torch.Tensor,
               test_x: torch.Tensor,
               test_y: torch.Tensor,
               n_runs: int,
               epochs: int,
               warmup_epochs: int,
               label_smoothing: float,
               dropout: float,
               device: str) -> Dict:

    print(f"\n{'='*65}")
    print(f"CONFIG : [{config_name}]")
    print(f"AUGS   : {aug_names if aug_names else ['none (baseline)']}")
    print()

    # Print any class-symmetry warnings for this config
    for name in aug_names:
        _, warning = AUGMENTATION_REGISTRY[name]
        if warning:
            print(f"  WARNING [{name}]: {warning}")

    # ── Train set shape (before aug — aug happens per-sample in __getitem__)
    print()
    print_tensor_shape(dist_x, dist_y,
                       f"TRAIN set — {config_name} (pre-aug, {len(dist_x)} samples)")

    # ── Show one augmented batch so you can verify dims + value range
    print_augmented_batch_shape(dist_x, aug_names, aug_param,
                                config_name, device)

    # ── Test set shape (never augmented)
    print()
    print_tensor_shape(test_x, test_y,
                       f"TEST set — {config_name} (no aug, always clean)")
    print()

    # Build test loader once — no augmentation
    test_loader = DataLoader(
        AugmentedSubset(test_x, test_y, [], None),
        batch_size=256, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run_accs  = []

    print(f"  {'Run':<5}  {'BestAcc':>8}")
    print(f"  {'---':<5}  {'-------':>8}")

    for run in range(n_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)

        train_set = AugmentedSubset(dist_x, dist_y,
                                    aug_names, aug_param)
        loader = DataLoader(train_set,
                            batch_size=min(64, len(dist_x)),
                            shuffle=True, num_workers=0)

        # On the first run print actual training batch shape from DataLoader
        if run == 0:
            xb_sample, yb_sample = next(iter(loader))
            print(f"\n  [DataLoader training batch — config: {config_name}]")
            print(f"    xb shape  : {tuple(xb_sample.shape)}  "
                  f"(B x T x H x W)")
            print(f"    yb shape  : {tuple(yb_sample.shape)}")
            print(f"    dtype     : {xb_sample.dtype}")
            print(f"    val range : [{xb_sample.min():.3f}, "
                  f"{xb_sample.max():.3f}]")
            print()

        model = Widar_CNN3D_Eval(NUM_CLASSES, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        best = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss   = criterion(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total   += len(yb)
                acc  = 100.0 * correct / total
                best = max(best, acc)

        run_accs.append(best)
        print(f"  {run+1:<5}  {best:>7.2f}%")

    mean_acc = float(np.mean(run_accs))
    std_acc  = float(np.std(run_accs))
    print(f"  {'MEAN':>5}  {mean_acc:>7.2f}% +/- {std_acc:.2f}%")

    return {
        'config':    config_name,
        'aug_names': ','.join(aug_names) if aug_names else 'none',
        'mean_acc':  mean_acc,
        'std_acc':   std_acc,
        'run_accs':  run_accs,
        'n_runs':    n_runs,
        'epochs':    epochs,
    }


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
def print_summary(results: List[Dict], baseline_acc: float,
                  n_samples: int, save_path: str):
    sorted_results = sorted(results, key=lambda r: -r['mean_acc'])
    lines = []

    lines.append(f"\n{'='*70}")
    lines.append(f"AUGMENTATION ABLATION SUMMARY")
    lines.append(f"Distilled samples : {n_samples}")
    lines.append(f"Runs per config   : {results[0]['n_runs']}")
    lines.append(f"Epochs per run    : {results[0]['epochs']}")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'Rank':<5}  {'Config':<25}  {'Acc':>8}  {'Std':>6}  "
        f"{'DeltaBase':>10}"
    )
    lines.append('-' * 60)

    for rank, r in enumerate(sorted_results, 1):
        delta     = r['mean_acc'] - baseline_acc
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        lines.append(
            f"{rank:<5}  {r['config']:<25}  "
            f"{r['mean_acc']:>7.2f}%  {r['std_acc']:>5.2f}%  "
            f"{delta_str:>10}"
        )

    lines.append('=' * 70)
    lines.append("DeltaBase = accuracy gain over no-augmentation baseline")
    lines.append("All augmentations sourced from your codebase.")
    lines.append('=' * 70)

    text = '\n'.join(lines)
    print(text)
    with open(save_path, 'w') as f:
        f.write(text)
    print(f"\nSummary saved -> {save_path}")


def save_csv(results: List[Dict], save_path: str):
    rows = []
    for r in results:
        for i, acc in enumerate(r['run_accs']):
            rows.append({
                'config':    r['config'],
                'aug_names': r['aug_names'],
                'run':       i + 1,
                'acc':       acc,
                'mean_acc':  r['mean_acc'],
                'std_acc':   r['std_acc'],
            })
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw results saved -> {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f"Device : {DEVICE}")
    print(f"Loading distilled indices from: {args.indices}")

    ckpt         = torch.load(args.indices, map_location='cpu',
                              weights_only=False)
    selected_idx = ckpt['indices']
    method       = ckpt.get('method', 'unknown')
    ipc          = ckpt.get('ipc', '?')
    print(f"  Method: {method}  IPC: {ipc}  Samples: {len(selected_idx)}")

    # ── Load datasets ────────────────────────────────────────
    print("\nLoading Widar datasets...")
    dst_train = Widar_Dataset(DATA_ROOT + 'train/')
    dst_test  = Widar_Dataset(DATA_ROOT + 'test/')
    print(f"  Train total: {len(dst_train)}  Test total: {len(dst_test)}")

    print(f"\nBuilding distilled tensors ({len(selected_idx)} samples)...")
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor([dst_train[i][1] for i in selected_idx],
                           dtype=torch.long)

    print("Building test tensors...")
    test_x = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
    test_y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                           dtype=torch.long)

    # Overall shape overview printed once before any configs run
    print(f"\n{'='*65}")
    print("DATASET SHAPE OVERVIEW (printed once at startup)")
    print(f"{'='*65}")
    print_tensor_shape(dist_x, dist_y,
                       f"Distilled train — {len(selected_idx)} samples")
    print()
    print_tensor_shape(test_x, test_y,
                       f"Full test set — {len(test_x)} samples")
    print(f"{'='*65}")

    # ── Select configs ───────────────────────────────────────
    if args.configs:
        run_names = args.configs
        for name in run_names:
            if name not in CONFIGS:
                raise ValueError(
                    f"Unknown config '{name}'. "
                    f"Available: {sorted(CONFIGS.keys())}"
                )
    else:
        run_names = list(CONFIGS.keys())

    if 'baseline' not in run_names:
        run_names = ['baseline'] + run_names

    print(f"\nConfigs to run ({len(run_names)}): {run_names}")

    # ── Augmentation params ──────────────────────────────────
    aug_param = AugParam(
        prob_flip_h           = args.prob_flip_h,
        prob_flip_w           = args.prob_flip_w,
        ratio_scale           = args.ratio_scale,
        ratio_crop_pad        = args.ratio_crop_pad,
        ratio_cutout          = args.ratio_cutout,
        max_temporal_shift    = args.max_temporal_shift,
        prob_temporal_flip    = args.prob_temporal_flip,
        ratio_temporal_cutout = args.ratio_temporal_cutout,
        amplitude             = args.amplitude,
        noise                 = args.noise,
    )

    print("\nAugmentation hyperparameters:")
    for field in dataclasses.fields(aug_param):
        print(f"  {field.name:<25} {getattr(aug_param, field.name)}")

    print(f"\nTraining settings:")
    print(f"  epochs          : {args.epochs}")
    print(f"  warmup_epochs   : {args.warmup_epochs}")
    print(f"  n_runs          : {args.n_runs}")
    print(f"  label_smoothing : {args.label_smoothing}")
    print(f"  dropout         : {args.dropout}")

    # ── Run ablation ─────────────────────────────────────────
    results      = []
    baseline_acc = 0.0

    for config_name in run_names:
        aug_names = CONFIGS[config_name]
        r = run_config(
            config_name     = config_name,
            aug_names       = aug_names,
            aug_param       = aug_param,
            dist_x          = dist_x,
            dist_y          = dist_y,
            test_x          = test_x,
            test_y          = test_y,
            n_runs          = args.n_runs,
            epochs          = args.epochs,
            warmup_epochs   = args.warmup_epochs,
            label_smoothing = args.label_smoothing,
            dropout         = args.dropout,
            device          = DEVICE,
        )
        results.append(r)
        if config_name == 'baseline':
            baseline_acc = r['mean_acc']

    # ── Save outputs ─────────────────────────────────────────
    tag          = os.path.splitext(os.path.basename(args.indices))[0]
    csv_path     = f"aug_ablation_{tag}.csv"
    summary_path = f"aug_ablation_{tag}_summary.txt"

    save_csv(results, csv_path)
    print_summary(results, baseline_acc,
                  n_samples=len(selected_idx),
                  save_path=summary_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augmentation ablation for Widar distilled subsets'
    )

    parser.add_argument('--indices', type=str, required=True,
                        help='Path to distilled_<method>_ipc<N>.pt')
    parser.add_argument('--configs', nargs='+', default=None,
                        choices=list(CONFIGS.keys()),
                        help='Configs to test. Default: all. '
                             'baseline always included.')

    # Training
    parser.add_argument('--epochs',          type=int,   default=500)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--n_runs',          type=int,   default=5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout',         type=float, default=0.3)

    # Augmentation hyperparams
    parser.add_argument('--prob_flip_h',           type=float, default=0.5)
    parser.add_argument('--prob_flip_w',           type=float, default=0.5)
    parser.add_argument('--ratio_scale',           type=float, default=1.2)
    parser.add_argument('--ratio_crop_pad',        type=float, default=0.125)
    parser.add_argument('--ratio_cutout',          type=float, default=0.2)
    parser.add_argument('--max_temporal_shift',    type=float, default=3.0)
    parser.add_argument('--prob_temporal_flip',    type=float, default=0.5)
    parser.add_argument('--ratio_temporal_cutout', type=float, default=0.2)
    parser.add_argument('--amplitude',             type=float, default=0.4)
    parser.add_argument('--noise',                 type=float, default=0.05)

    parser.add_argument('--smoke_test', action='store_true',
                        help='50 epochs, 2 runs to verify script before full run')

    args = parser.parse_args()

    if args.smoke_test:
        print("WARNING: SMOKE TEST MODE — epochs=50, n_runs=2")
        args.epochs = 50
        args.n_runs = 2

    main(args)
