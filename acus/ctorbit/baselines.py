from __future__ import annotations

import math

import torch
import torch.nn.functional as F

def _circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d.abs()

def psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, gt, reduction="none").flatten(2).mean(-1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))

def nearest_and_lerp(
    obs_views: torch.Tensor,
    obs_rel: torch.Tensor,
    query_rel: torch.Tensor,
    obs_mask: torch.Tensor | None,
    out_hw: int,
) -> dict[str, torch.Tensor]:

    b, k, c, h, w = obs_views.shape
    q = query_rel.shape[1]
    dist = _circ_dist(query_rel.unsqueeze(-1), obs_rel.unsqueeze(1))
    if obs_mask is not None:
        dist = dist.masked_fill(obs_mask.unsqueeze(1) < 0.5, 1e6)
    d_sort, idx_sort = dist.sort(dim=-1)
    nn_idx = idx_sort[..., 0]
    gather = nn_idx.view(b, q, 1, 1, 1).expand(b, q, c, h, w)
    nn_img = torch.gather(obs_views, 1, gather)

    if k == 1:
        lerp = nn_img
    else:
        idx2 = idx_sort[..., 1]
        g2 = idx2.view(b, q, 1, 1, 1).expand(b, q, c, h, w)
        img2 = torch.gather(obs_views, 1, g2)
        d1 = d_sort[..., 0].clamp_min(1e-6)
        d2 = d_sort[..., 1].clamp_min(1e-6)
        w1 = (d2 / (d1 + d2)).view(b, q, 1, 1, 1)
        lerp = w1 * nn_img + (1.0 - w1) * img2

    if h != out_hw or w != out_hw:
        nn_img = F.interpolate(nn_img.flatten(0, 1), size=out_hw, mode="bilinear", align_corners=False).view(
            b, q, c, out_hw, out_hw
        )
        lerp = F.interpolate(lerp.flatten(0, 1), size=out_hw, mode="bilinear", align_corners=False).view(
            b, q, c, out_hw, out_hw
        )
    deg = d_sort[..., 0] * (180.0 / math.pi)
    return {"nn": nn_img, "lerp": lerp, "gap_deg": deg}

def summarize_psnr(pred: torch.Tensor, gt: torch.Tensor, gap_deg: torch.Tensor) -> dict[str, float]:

    pv = psnr_torch(pred, gt)
    out = {
        "psnr": float(pv.mean()),
        "psnr_near": float("nan"),
        "psnr_far": float("nan"),
        "n_near": 0,
        "n_far": 0,
        "gap_mean": float(gap_deg.mean()),
    }
    near = gap_deg < 20.0
    far = gap_deg >= 40.0
    if near.any():
        out["psnr_near"] = float(pv[near].mean())
        out["n_near"] = int(near.sum())
    if far.any():
        out["psnr_far"] = float(pv[far].mean())
        out["n_far"] = int(far.sum())
    return out
