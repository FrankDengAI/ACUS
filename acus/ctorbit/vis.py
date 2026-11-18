"""Save input / held-out query / world-volume slice grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _to_np(img: torch.Tensor) -> np.ndarray:
    x = img.detach().float().cpu().clamp(0, 1).numpy()
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    return x


def _hstack(imgs: list[np.ndarray], pad: int = 2) -> np.ndarray:
    h = max(a.shape[0] for a in imgs)
    rows = []
    for a in imgs:
        if a.shape[0] != h:
            t = torch.from_numpy(a)[None, None]
            t = F.interpolate(t, size=(h, int(a.shape[1] * h / a.shape[0])), mode="bilinear", align_corners=False)
            a = t[0, 0].numpy()
        rows.append(a)
        rows.append(np.ones((h, pad), np.float32))
    return np.concatenate(rows[:-1], axis=1)


def save_case_panel(
    out_path: Path,
    obs: torch.Tensor,
    obs_mask: torch.Tensor,
    q_gt: torch.Tensor,
    q_pred: torch.Tensor,
    vol: torch.Tensor,
    gt_vol: torch.Tensor,
    title: str = "",
    q_pred_vol: torch.Tensor | None = None,
    fdk_obs: torch.Tensor | None = None,
    fdk_fill: torch.Tensor | None = None,
) -> None:
    """obs [K,1,H,W]; queries [Q,1,H,W]; vol [1,D,H,W]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(obs_mask.sum().item())
    obs_list = [_to_np(obs[i]) for i in range(max(k, 1))]
    qg = [_to_np(q_gt[i]) for i in range(q_gt.shape[0])]
    qp = [_to_np(q_pred[i]) for i in range(q_pred.shape[0])]
    v = vol.detach().float().cpu().clamp(0, 1)[0]
    g = gt_vol.detach().float().cpu().clamp(0, 1)
    if g.ndim == 4:
        g = g[0]
    mid = v.shape[0] // 2

    def _xyz(t: torch.Tensor) -> list[np.ndarray]:
        x = t.detach().float().cpu().clamp(0, 1)
        if x.ndim == 4:
            x = x[0]
        m = x.shape[0] // 2
        return [x[m].numpy(), x[:, m].numpy(), x[:, :, m].numpy()]

    slices_v = _xyz(v)
    slices_g = _xyz(g)
    vol_chunks = slices_g
    vol_lab = "GT"
    if fdk_obs is not None:
        vol_chunks = vol_chunks + _xyz(fdk_obs)
        vol_lab += " | FDK_obs"
    if fdk_fill is not None:
        vol_chunks = vol_chunks + _xyz(fdk_fill)
        vol_lab += " | FDK_fill"
    vol_chunks = vol_chunks + slices_v
    vol_lab += " | Ours"

    rows = 5 if q_pred_vol is not None else 4
    fig, axes = plt.subplots(rows, 1, figsize=(min(2.0 * max(k, len(qg), len(vol_chunks)), 22), 2.4 * rows + 1))
    fig.suptitle(title, fontsize=10)

    def show(ax, row, lab):
        ax.imshow(row, cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    show(axes[0], _hstack(obs_list), "obs")
    show(axes[1], _hstack(qg), "query GT")
    show(axes[2], _hstack(qp), "query pred (2D/3D blend)")
    ri = 3
    if q_pred_vol is not None:
        qpv = [_to_np(q_pred_vol[i]) for i in range(q_pred_vol.shape[0])]
        show(axes[3], _hstack(qpv), "P(V) cone-beam")
        ri = 4
    show(axes[ri], _hstack(vol_chunks), vol_lab + " (ax/cor/sag)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_curves(jsonl_path: Path, out_path: Path) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = [], []
    if not jsonl_path.is_file():
        return
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "psnr_q" in rec:
            train.append(rec)
        if "val_psnr_query" in rec:
            val.append(rec)
    if not train:
        return
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    st = [r["step"] for r in train]
    ax[0, 0].plot(st, [r["loss"] for r in train], lw=1)
    ax[0, 0].set_title("train loss")
    ax[0, 1].plot(st, [r["psnr_q"] for r in train], lw=1, label="query")
    if train and "psnr_o" in train[-1]:
        ax[0, 1].plot(st, [r.get("psnr_o", float("nan")) for r in train], lw=1, label="obs reproj")
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("train PSNR")
    ax[1, 0].plot(st, [r["l_q"] for r in train], lw=1, label="Lq")
    ax[1, 0].plot(st, [r["l_o"] for r in train], lw=1, label="Lo")
    ax[1, 0].plot(st, [r["l_v"] for r in train], lw=1, label="Lv")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title("loss terms")
    if val:
        ax[1, 1].plot([r["step"] for r in val], [r["val_psnr_query"] for r in val], "o-", lw=1, label="model")
        if "val_psnr_lerp" in val[-1]:
            ax[1, 1].plot([r["step"] for r in val], [r["val_psnr_lerp"] for r in val], "--", lw=1, label="lerp")
        if "val_psnr_nn" in val[-1]:
            ax[1, 1].plot([r["step"] for r in val], [r["val_psnr_nn"] for r in val], ":", lw=1, label="NN")
        ax[1, 1].axhline(30.0, color="gray", ls=":", lw=0.8, label="30dB")
        if "val_psnr_vol" in val[-1]:
            ax[1, 1].plot([r["step"] for r in val], [r.get("val_psnr_vol", float("nan")) for r in val], "s-", lw=1, label="vol")
        if "val_psnr_fdk" in val[-1]:
            ax[1, 1].plot([r["step"] for r in val], [r.get("val_psnr_fdk", float("nan")) for r in val], "^--", lw=1, label="FDK_obs")
        ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("val PSNR vs baselines")
    for a in ax.ravel():
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def prune_old_vis(vis_dir: Path, keep_evals: int = 2) -> None:
    files = sorted(vis_dir.glob("step*.png"), key=lambda p: p.stat().st_mtime)
    steps = sorted({p.name.split("_")[0] for p in files})
    drop = set(steps[:-keep_evals])
    for p in files:
        if p.name.split("_")[0] in drop:
            p.unlink(missing_ok=True)
