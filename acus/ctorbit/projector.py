"""Differentiable cone-beam projector (batched, keeps grad)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def geo_from_yaml(cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    p = cfg["projector"]
    nDet = p["nDetector"]
    dDet = p["dDetector"]
    nVoxel = p["nVoxel"]
    dVoxel = p["dVoxel"]
    return {
        "nv": int(nDet[0]),
        "nu": int(nDet[1]),
        "dDet_v": float(dDet[0]) / 1000.0,
        "dDet_u": float(dDet[1]) / 1000.0,
        "DSD": float(p["DSD"]) / 1000.0,
        "DSO": float(p["DSO"]) / 1000.0,
        "sVoxel": (
            float(nVoxel[0]) * float(dVoxel[0]) / 1000.0,
            float(nVoxel[1]) * float(dVoxel[1]) / 1000.0,
            float(nVoxel[2]) * float(dVoxel[2]) / 1000.0,
        ),
    }


def cone_project(
    vol: torch.Tensor,
    angles: torch.Tensor,
    geo: dict[str, Any],
    n_samples: int = 96,
    out_hw: int | None = None,
) -> torch.Tensor:
    """
    vol: [B, 1, D, H, W] occupancy in [-aligned] grid_sample space
    angles: [B, Q] radians
    returns: [B, Q, 1, nv, nu]  (v flipped to match preprocess)
    """
    b, q = angles.shape
    nv0, nu0 = int(geo["nv"]), int(geo["nu"])
    nv = out_hw or nv0
    nu = out_hw or nu0
    dDet_v = float(geo["dDet_v"]) * (nv0 / nv)
    dDet_u = float(geo["dDet_u"]) * (nu0 / nu)
    DSD = float(geo["DSD"])
    DSO = float(geo["DSO"])
    sx, sy, sz = geo["sVoxel"]
    device, dtype = vol.device, vol.dtype
    half = torch.tensor([sx, sy, sz], device=device, dtype=dtype) / 2.0

    u = (torch.arange(nu, device=device, dtype=dtype) - (nu - 1) / 2.0) * dDet_u
    v = (torch.arange(nv, device=device, dtype=dtype) - (nv - 1) / 2.0) * dDet_v
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # [nv, nu]
    ts = torch.linspace(0, 1, n_samples, device=device, dtype=dtype)

    outs = []
    for qi in range(q):
        ang = angles[:, qi]
        c, s = torch.cos(ang), torch.sin(ang)
        src = torch.stack([DSO * s, -DSO * c, torch.zeros_like(c)], dim=-1)  # [B,3]
        det_c = torch.stack([-(DSD - DSO) * s, (DSD - DSO) * c, torch.zeros_like(c)], dim=-1)
        axis_u = torch.stack([c, s, torch.zeros_like(c)], dim=-1)
        axis_v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(b, 3)
        det_pts = (
            det_c[:, None, None, :]
            + uu[None, ..., None] * axis_u[:, None, None, :]
            + vv[None, ..., None] * axis_v[:, None, None, :]
        )
        dirs = det_pts - src[:, None, None, :]
        dirs = torch.where(dirs.abs() < 1e-8, torch.full_like(dirs, 1e-8), dirs)
        t0 = (-half - src)[:, None, None, :] / dirs
        t1 = (half - src)[:, None, None, :] / dirs
        tmin = torch.minimum(t0, t1).amax(dim=-1).clamp(min=0)
        tmax = torch.maximum(t0, t1).amin(dim=-1)
        valid = tmax > tmin
        t = tmin[..., None] + (tmax - tmin).clamp(min=0)[..., None] * ts.view(1, 1, 1, -1)
        pts = src[:, None, None, None, :] + t[..., None] * dirs[..., None, :]
        grid = (pts / half).view(b, nv, nu * n_samples, 1, 3)
        samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        samp = samp.view(b, 1, nv, nu, n_samples)
        dt = ((tmax - tmin).clamp(min=0) / max(n_samples - 1, 1)).clamp(min=0)
        line = (samp * valid[:, None, ..., None].to(dtype) * dt[:, None, ..., None]).sum(dim=-1)
        outs.append(line)
    projs = torch.stack(outs, dim=1)  # [B,Q,1,nv,nu]
    return torch.flip(projs, dims=[-2])


def sample_cone_rays(
    angles: torch.Tensor,
    geo: dict[str, Any],
    out_hw: int,
    n_samples: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query cone-beam rays in the same frame as cone_project / preprocess.

    angles: [B, Q] radians
    returns:
      pts   [B, Q, nv, nu, M, 3] in [-1, 1] volume coords
      dt    [B, Q, nv, nu]       parametric step (same units as cone_project)
      valid [B, Q, nv, nu]
    """
    b, q = angles.shape
    nv0, nu0 = int(geo["nv"]), int(geo["nu"])
    nv = nu = int(out_hw)
    dDet_v = float(geo["dDet_v"]) * (nv0 / nv)
    dDet_u = float(geo["dDet_u"]) * (nu0 / nu)
    DSD = float(geo["DSD"])
    DSO = float(geo["DSO"])
    sx, sy, sz = geo["sVoxel"]
    half = torch.tensor([sx, sy, sz], device=device, dtype=dtype) / 2.0
    u = (torch.arange(nu, device=device, dtype=dtype) - (nu - 1) / 2.0) * dDet_u
    v = (torch.arange(nv, device=device, dtype=dtype) - (nv - 1) / 2.0) * dDet_v
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    ts = torch.linspace(0, 1, n_samples, device=device, dtype=dtype)
    pts_q, dt_q, valid_q = [], [], []
    for qi in range(q):
        ang = angles[:, qi]
        c, s = torch.cos(ang), torch.sin(ang)
        src = torch.stack([DSO * s, -DSO * c, torch.zeros_like(c)], dim=-1)
        det_c = torch.stack([-(DSD - DSO) * s, (DSD - DSO) * c, torch.zeros_like(c)], dim=-1)
        axis_u = torch.stack([c, s, torch.zeros_like(c)], dim=-1)
        axis_v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(b, 3)
        det_pts = (
            det_c[:, None, None, :]
            + uu[None, ..., None] * axis_u[:, None, None, :]
            + vv[None, ..., None] * axis_v[:, None, None, :]
        )
        dirs = det_pts - src[:, None, None, :]
        dirs = torch.where(dirs.abs() < 1e-8, torch.full_like(dirs, 1e-8), dirs)
        t0 = (-half - src)[:, None, None, :] / dirs
        t1 = (half - src)[:, None, None, :] / dirs
        tmin = torch.minimum(t0, t1).amax(dim=-1).clamp(min=0)
        tmax = torch.maximum(t0, t1).amin(dim=-1)
        valid = tmax > tmin
        t = tmin[..., None] + (tmax - tmin).clamp(min=0)[..., None] * ts.view(1, 1, 1, -1)
        pts = src[:, None, None, None, :] + t[..., None] * dirs[..., None, :]
        pts_q.append((pts / half).view(b, nv, nu, n_samples, 3))
        dt_q.append(((tmax - tmin).clamp(min=0) / max(n_samples - 1, 1)).clamp(min=0))
        valid_q.append(valid)
    return torch.stack(pts_q, 1), torch.stack(dt_q, 1), torch.stack(valid_q, 1)


def world_to_uv(
    pts_norm: torch.Tensor,
    ang: torch.Tensor,
    geo: dict[str, Any],
    out_hw: int,
    flip_v: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """pts_norm [B,N,3] in [-1,1] volume coords; ang [B] radians.
    returns uv [B,N,2] in [-1,1] for grid_sample, valid [B,N].
    flip_v matches cone_project's last-axis flip on stored projections.
    """
    b, n, _ = pts_norm.shape
    nv0, nu0 = int(geo["nv"]), int(geo["nu"])
    nv = nu = int(out_hw)
    dDet_v = float(geo["dDet_v"]) * (nv0 / nv)
    dDet_u = float(geo["dDet_u"]) * (nu0 / nu)
    DSD = float(geo["DSD"])
    DSO = float(geo["DSO"])
    sx, sy, sz = geo["sVoxel"]
    device, dtype = pts_norm.device, pts_norm.dtype
    half = torch.tensor([sx, sy, sz], device=device, dtype=dtype).view(1, 1, 3) / 2.0
    p = pts_norm * half
    c, s = torch.cos(ang), torch.sin(ang)
    src = torch.stack([DSO * s, -DSO * c, torch.zeros_like(c)], dim=-1)
    det_c = torch.stack([-(DSD - DSO) * s, (DSD - DSO) * c, torch.zeros_like(c)], dim=-1)
    axis_u = torch.stack([c, s, torch.zeros_like(c)], dim=-1)
    axis_v = torch.zeros(b, 3, device=device, dtype=dtype)
    axis_v[:, 2] = 1.0
    normal = torch.stack([s, -c, torch.zeros_like(c)], dim=-1)
    direction = p - src[:, None, :]
    denom = (direction * normal[:, None, :]).sum(-1)
    numer = ((det_c - src) * normal).sum(-1, keepdim=True)
    denom_u = denom.unsqueeze(-1)
    t = numer / torch.where(denom_u.abs() < 1e-6, torch.ones_like(denom_u), denom_u)
    hit = src[:, None, :] + t * direction
    rel = hit - det_c[:, None, :]
    u_mm = (rel * axis_u[:, None, :]).sum(-1)
    v_mm = (rel * axis_v[:, None, :]).sum(-1)
    u_n = 2.0 * (u_mm / dDet_u + (nu - 1) / 2.0) / max(nu - 1, 1) - 1.0
    v_n = 2.0 * (v_mm / dDet_v + (nv - 1) / 2.0) / max(nv - 1, 1) - 1.0
    if flip_v:
        v_n = -v_n
    valid = (u_n.abs() <= 1.05) & (v_n.abs() <= 1.05) & (denom.abs() > 1e-6)
    uv = torch.stack([u_n, v_n], dim=-1)
    return uv, valid


def voxel_centers(res: int, device, dtype) -> torch.Tensor:
    t = torch.linspace(-1, 1, res, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(t, t, t, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).view(1, -1, 3)


def backproject(
    views: torch.Tensor,
    angles: torch.Tensor,
    geo: dict[str, Any],
    res: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unfiltered cone-beam backprojection.
    views [B,K,C,H,W] (C=1 intensity or C=feat); angles [B,K]
    returns [B,C,R,R,R]
    """
    b, k, ch, h, w = views.shape
    pts = voxel_centers(res, views.device, views.dtype).expand(b, -1, -1)
    acc = views.new_zeros(b, ch, res * res * res)
    cnt = views.new_zeros(b, 1, res * res * res)
    DSO = float(geo["DSO"])
    sx, sy, sz = geo["sVoxel"]
    half = torch.tensor([sx, sy, sz], device=views.device, dtype=views.dtype).view(1, 1, 3) / 2.0
    p = pts * half
    for ki in range(k):
        if mask is not None and not bool((mask[:, ki] > 0.5).any()):
            continue
        uv, valid = world_to_uv(pts, angles[:, ki], geo, out_hw=h)
        if mask is not None:
            valid = valid & (mask[:, ki] > 0.5)[:, None]
        ang = angles[:, ki]
        c, s = torch.cos(ang), torch.sin(ang)
        src = torch.stack([DSO * s, -DSO * c, torch.zeros_like(c)], dim=-1)
        dist = (p - src[:, None, :]).norm(dim=-1).clamp_min(1e-3)
        geo_w = (DSO / dist).clamp(0.05, 8.0)
        grid = uv.view(b, -1, 1, 2)
        samp = F.grid_sample(
            views[:, ki], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(b, ch, -1)
        wgt = valid.to(dtype=views.dtype) * geo_w
        wgt = wgt[:, None, :]
        acc = acc + samp * wgt
        cnt = cnt + wgt
    vol = acc / cnt.clamp_min(1e-4)
    return vol.view(b, ch, res, res, res)


def ramp_filter_u(views: torch.Tensor) -> torch.Tensor:
    """Ram-Lak (+ Hamming) along detector u. views [B,K,1,H,W] in [0,1] (X-ray)."""
    b, k, c, h, w = views.shape
    att = (1.0 - views).clamp(0, 1).reshape(-1, w)
    freq = torch.fft.rfftfreq(w, device=views.device, dtype=torch.float32)
    kernel = freq.abs()
    scale = freq.max().clamp_min(1e-6)
    kernel = kernel * (0.54 + 0.46 * torch.cos(torch.pi * freq / scale))
    y = torch.fft.irfft(torch.fft.rfft(att.float(), dim=-1) * kernel, n=w, dim=-1)
    return y.to(dtype=views.dtype).view(b, k, c, h, w)


def fdk_backproject(
    views: torch.Tensor,
    angles: torch.Tensor,
    geo: dict[str, Any],
    res: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ramp-filtered cone-beam BP. Returns [B,1,R,R,R] (not intensity-normalized)."""
    hw = min(int(views.shape[-1]), max(int(res), 128))
    if views.shape[-1] != hw:
        v = F.interpolate(views.flatten(0, 1), size=hw, mode="bilinear", align_corners=False)
        views = v.view(views.size(0), views.size(1), 1, hw, hw)
    return backproject(ramp_filter_u(views), angles, geo, res, mask)


def unit_range(vol: torch.Tensor) -> torch.Tensor:
    """Per-sample 2–98 percentile to [0,1]. vol [B,1,D,H,W]."""
    b = vol.size(0)
    flat = vol.reshape(b, -1).float()
    lo = torch.quantile(flat, 0.02, dim=1).view(b, 1, 1, 1, 1)
    hi = torch.quantile(flat, 0.98, dim=1).view(b, 1, 1, 1, 1)
    return ((vol - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def affine_match(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Least-squares a*pred+b ≈ gt. cone_project line integrals vs stored views."""
    x = pred.reshape(-1).float()
    y = gt.reshape(-1).float()
    if mask is not None:
        m = mask
        while m.dim() < pred.dim():
            m = m.unsqueeze(-1)
        m = (m.expand_as(pred) > 0.5).reshape(-1)
        if bool(m.any()):
            x, y = x[m], y[m]
    mx, my = x.mean(), y.mean()
    var = ((x - mx) ** 2).mean().clamp_min(1e-8)
    a = ((x - mx) * (y - my)).mean() / var
    a = a.clamp(0.05, 40.0)
    b = my - a * mx
    return a * pred + b
