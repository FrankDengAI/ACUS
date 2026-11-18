"""Hybrid: X-WIN-style action predictor (all patch tokens → pixels) + triplane world P_a(V)."""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import nearest_and_lerp
from .encoder import build_encoder
from .mae_vit import Block
from .projector import backproject, cone_project, fdk_backproject, sample_cone_rays, unit_range, voxel_centers, world_to_uv


def fourier_angle(angles: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freq = torch.arange(half, device=angles.device, dtype=angles.dtype)
    freq = (1000 ** (freq / max(half - 1, 1))).view(*([1] * angles.ndim), -1)
    ang = angles.unsqueeze(-1)
    return torch.cat([torch.sin(ang * freq), torch.cos(ang * freq)], dim=-1)


def sincos_2d(n: int, dim: int, device, dtype) -> torch.Tensor:
    """X-WIN-style 2D sine-cosine positional embedding, [n*n, dim]."""
    g = int(n ** 0.5)
    y, x = torch.meshgrid(
        torch.arange(g, device=device, dtype=dtype),
        torch.arange(g, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    half = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)))
    ox = pos[:, 0:1] * omega
    oy = pos[:, 1:2] * omega
    emb = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class PixelHead(nn.Module):
    """14x14 tokens → HxW. Bilinear upsample + conv (no ConvTranspose checkerboard)."""

    def __init__(self, dim=768, out_size=224, use_aux=False):
        super().__init__()
        self.out_size = out_size
        self.use_aux = use_aux
        self.proj = nn.Conv2d(dim, 256, 1)
        chs = [256, 128, 64, 32]
        layers: list[nn.Module] = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(a, b, 3, padding=1),
                nn.GELU(),
            ]
        layers += [nn.Conv2d(chs[-1], 1, 3, padding=1)]
        self.up = nn.Sequential(*layers)
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)
        if use_aux:
            self.aux_fuse = nn.Conv2d(2, 1, 3, padding=1)
            nn.init.zeros_(self.aux_fuse.weight)
            nn.init.zeros_(self.aux_fuse.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        skip: torch.Tensor | None = None,
        gap_deg: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, q, n, c = tokens.shape
        g = int(n ** 0.5)
        x = tokens.reshape(b * q, g, g, c).permute(0, 3, 1, 2)
        x = self.up(self.proj(x))
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = x.view(b, q, 1, self.out_size, self.out_size)
        if skip is None:
            y = torch.sigmoid(x)
        else:
            if gap_deg is None:
                gain = skip.new_full((b, q, 1, 1, 1), 0.5)
            else:
                # Identity skip below ~8°. Residual only turns on as the angular hole grows.
                gate = ((gap_deg - 8.0) / 12.0).clamp(0, 1)
                gain = gate * (0.2 + 1.6 * (gap_deg / 70.0).clamp(0, 1))
                gain = gain.view(b, q, 1, 1, 1)
            y = (skip + gain * torch.tanh(x)).clamp(0, 1)
        if self.use_aux and aux is not None:
            yb = y.reshape(b * q, 1, self.out_size, self.out_size)
            ab = aux.reshape(b * q, 1, aux.shape[-2], aux.shape[-1])
            if ab.shape[-1] != self.out_size:
                ab = F.interpolate(ab, size=self.out_size, mode="bilinear", align_corners=False)
            y = (yb + self.aux_fuse(torch.cat([yb, ab], dim=1))).clamp(0, 1)
            y = y.view(b, q, 1, self.out_size, self.out_size)
        return y


class ScaleHead(nn.Module):
    """Per-pixel log-scale of the query-view distribution. Zero residual, bias log(0.08)."""

    def __init__(self, dim=768, out_size=224):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(dim, 128, 1)
        chs = [128, 64, 32]
        layers: list[nn.Module] = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(a, b, 3, padding=1),
                nn.GELU(),
            ]
        layers += [nn.Conv2d(chs[-1], 1, 3, padding=1)]
        self.up = nn.Sequential(*layers)
        nn.init.zeros_(self.up[-1].weight)
        nn.init.constant_(self.up[-1].bias, math.log(0.08))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, q, n, c = tokens.shape
        g = int(n ** 0.5)
        x = tokens.reshape(b * q, g, g, c).permute(0, 3, 1, 2)
        x = self.up(self.proj(x))
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        # Images live in [0, 1]. Cap the scale so a wide fiber stays wide
        # without a variance that covers every possible projection.
        return x.view(b, q, 1, self.out_size, self.out_size).clamp(-6.0, math.log(0.35))


class FlowHead(nn.Module):
    """Zero-init 2D warp from tokens. Identity at step 0, then deforms lerp skip."""

    def __init__(self, dim=768, out_size=224):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(dim, 64, 1)
        chs = [64, 32, 16, 8]
        layers: list[nn.Module] = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(a, b, 3, padding=1),
                nn.GELU(),
            ]
        layers += [nn.Conv2d(chs[-1], 2, 3, padding=1)]
        self.up = nn.Sequential(*layers)
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, tokens: torch.Tensor, gap_deg: torch.Tensor | None = None) -> torch.Tensor:
        b, q, n, c = tokens.shape
        g = int(n ** 0.5)
        x = tokens.reshape(b * q, g, g, c).permute(0, 3, 1, 2)
        x = self.up(self.proj(x))
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        flow = 0.2 * torch.tanh(x).view(b, q, 2, self.out_size, self.out_size)
        if gap_deg is not None:
            gate = ((gap_deg - 8.0) / 12.0).clamp(0, 1).view(b, q, 1, 1, 1)
            flow = flow * gate
        return flow


def warp_imgs(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """img [B,Q,1,H,W]; flow [B,Q,2,H,W] in normalized coords."""
    b, q, c, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, h, w, 2).expand(b, q, h, w, 2)
    grid = grid + flow.permute(0, 1, 3, 4, 2)
    out = F.grid_sample(
        img.reshape(b * q, c, h, w),
        grid.reshape(b * q, h, w, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out.view(b, q, c, h, w)


class GeoMixWide(nn.Module):
    """Frozen-xwin residual with more hint channels (P(V), lerp skip, |2d-3d|). Last layer zero-init."""

    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, img2d, img3d, skip, gap_deg):
        b, q, _, h, w = img2d.shape
        if img3d.shape[-1] != h:
            img3d = F.interpolate(img3d.flatten(0, 1), size=h, mode="bilinear", align_corners=False).view(b, q, 1, h, w)
        if skip.shape[-1] != h:
            skip = F.interpolate(skip.flatten(0, 1), size=h, mode="bilinear", align_corners=False).view(b, q, 1, h, w)
        x = torch.cat([img2d, img3d, skip, (img2d - img3d).abs()], dim=2).reshape(b * q, 4, h, w)
        r = self.net(x).view(b, q, 1, h, w)
        g = ((gap_deg - 12.0) / 28.0).clamp(0, 1).view(b, q, 1, 1, 1)
        return (img2d + g * torch.tanh(r)).clamp(0, 1)


class GeoMixHead(nn.Module):
    """Zero-init residual: start exactly at frozen xwin, use P(V) as a geometric hint.

    Raw P(V) from sparse UBP is ~15 dB on far queries (worse than xwin 18.5), so it must
    not replace the 2D prediction. A 2-channel conv can copy xwin and only add 3D where
    it reduces error.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, img2d: torch.Tensor, img3d: torch.Tensor, gap_deg: torch.Tensor) -> torch.Tensor:
        b, q, _, h, w = img2d.shape
        if img3d.shape[-1] != h or img3d.shape[-2] != w:
            img3d = F.interpolate(img3d.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False).view(
                b, q, 1, h, w
            )
        x = torch.cat([img2d, img3d], dim=2).reshape(b * q, 2, h, w)
        r = self.net(x).view(b, q, 1, h, w)
        g = ((gap_deg - 12.0) / 28.0).clamp(0, 1).view(b, q, 1, 1, 1)
        return (img2d + g * torch.tanh(r)).clamp(0, 1)


class WorldUnproject(nn.Module):
    """Lift K views into a 3D attenuation field.

    kind=bp: intensity backprojection only (v3).
    kind=unet: intensity + patch-feature BP, 3D UNet residual (zero-init → starts at bp).
    kind=triplane: axis-aligned planes from BP features, MLP residual (zero-init → bp).
    """

    def __init__(self, dim=768, feat_ch=8, res=64, kind="bp"):
        super().__init__()
        self.res = res
        self.kind = kind
        self.feat_ch = feat_ch
        self.i_gain = nn.Parameter(torch.tensor(1.0))
        self.i_bias = nn.Parameter(torch.tensor(0.0))
        self.intensity_hw = 64
        self.feat_hw = 32
        if kind in ("unet", "triplane"):
            self.feat_proj = nn.Sequential(nn.Linear(dim, feat_ch), nn.GELU())
        if kind == "unet":
            cin = 1 + feat_ch
            self.enc1 = nn.Sequential(nn.Conv3d(cin, 16, 3, padding=1), nn.GELU())
            self.enc2 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GELU())
            self.dec2 = nn.Sequential(nn.ConvTranspose3d(32, 16, 2, stride=2), nn.GELU())
            self.dec1 = nn.Conv3d(32, 1, 3, padding=1)
            nn.init.zeros_(self.dec1.weight)
            nn.init.zeros_(self.dec1.bias)
        if kind == "triplane":
            cin = 1 + feat_ch
            self.plane_xy = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1))
            self.plane_xz = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1))
            self.plane_yz = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1))
            self.mlp = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _feat_vol(self, patches, abs_obs, obs_mask, geo):
        b, k, n, c = patches.shape
        g = int(n ** 0.5)
        fmap = self.feat_proj(patches).view(b, k, g, g, self.feat_ch).permute(0, 1, 4, 2, 3)
        if fmap.shape[-1] != self.feat_hw:
            fmap = F.interpolate(fmap.flatten(0, 1), size=self.feat_hw, mode="bilinear", align_corners=False)
            fmap = fmap.view(b, k, self.feat_ch, self.feat_hw, self.feat_hw)
        return backproject(fmap, abs_obs, geo, self.res, obs_mask)

    def forward(self, patches, obs_views, abs_obs, obs_mask, geo) -> torch.Tensor:
        b, k = obs_views.shape[:2]
        ih = self.intensity_hw
        obs_s = obs_views
        if obs_s.shape[-1] != ih:
            obs_s = F.interpolate(obs_s.flatten(0, 1), size=ih, mode="bilinear", align_corners=False)
            obs_s = obs_s.view(b, k, 1, ih, ih)
        i_vol = backproject(obs_s, abs_obs, geo, self.res, obs_mask)
        base = self.i_gain.clamp(0.5, 1.6) * i_vol + self.i_bias.clamp(-0.1, 0.1)
        if self.kind == "bp":
            return base.clamp(0, 1)
        f_vol = self._feat_vol(patches, abs_obs, obs_mask, geo)
        x = torch.cat([i_vol, f_vol], dim=1)
        if self.kind == "unet":
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            d2 = self.dec2(e2)
            if d2.shape[-1] != e1.shape[-1]:
                d2 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
            delta = self.dec1(torch.cat([d2, e1], dim=1))
            return (base + 0.4 * torch.tanh(delta)).clamp(0, 1)
        # triplane: volume layout [B,C,Z,Y,X]
        xy = self.plane_xy(torch.cat([i_vol.mean(2), f_vol.mean(2)], dim=1))
        xz = self.plane_xz(torch.cat([i_vol.mean(3), f_vol.mean(3)], dim=1))
        yz = self.plane_yz(torch.cat([i_vol.mean(4), f_vol.mean(4)], dim=1))
        r = self.res
        fxy = xy.unsqueeze(2).expand(-1, -1, r, -1, -1)
        fxz = xz.unsqueeze(3).expand(-1, -1, -1, r, -1)
        fyz = yz.unsqueeze(4).expand(-1, -1, -1, -1, r)
        h = torch.cat([fxy, fxz, fyz], dim=1).permute(0, 2, 3, 4, 1)
        delta = self.mlp(h).permute(0, 4, 1, 2, 3)
        return (base + 0.4 * torch.tanh(delta)).clamp(0, 1)


def _fourier_xyz(pts: torch.Tensor, n_freq: int = 6) -> torch.Tensor:
    """NeRF-style positional encoding, pts [..., 3] in [-1, 1] → [..., 6*n_freq]."""
    freq = (2.0 ** torch.arange(n_freq, device=pts.device, dtype=pts.dtype)) * 3.14159265
    ang = pts.unsqueeze(-1) * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


class RayWorld(nn.Module):
    """View-invariant attenuation μ(x) by pooling K observed measurements at x, then P_a(μ).

    This is the CT-OrbitWorld world: a 3D point is explained by every detector that sees it.
    Unlike intensity backprojection, μ is inferred from multi-view evidence and only then
    integrated along the *query* cone-beam ray (same quadrature as preprocess).
    """

    def __init__(self, dim=768, feat_ch=16, ray_hw=48, n_samples=24, vol_preview=32):
        super().__init__()
        self.feat_ch = feat_ch
        self.ray_hw = ray_hw
        self.n_samples = n_samples
        self.vol_preview = vol_preview
        self.feat_proj = nn.Sequential(nn.Linear(dim, feat_ch), nn.GELU())
        in_ch = feat_ch * 2 + 4 + 6 * 6  # mean/max feat, I mean/std/vis/count, xyz PE
        self.mu_mlp = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.proj_scale = nn.Parameter(torch.ones(2) * 6.0)
        self.proj_bias = nn.Parameter(torch.zeros(2))

    def _obs_maps(self, patches, obs_views):
        b, k, n, c = patches.shape
        g = int(n ** 0.5)
        fmap = self.feat_proj(patches).view(b, k, g, g, self.feat_ch).permute(0, 1, 4, 2, 3)
        hw = self.ray_hw
        if fmap.shape[-1] != hw:
            fmap = F.interpolate(fmap.flatten(0, 1), size=hw, mode="bilinear", align_corners=False)
            fmap = fmap.view(b, k, self.feat_ch, hw, hw)
        inten = obs_views
        if inten.shape[-1] != hw:
            inten = F.interpolate(inten.flatten(0, 1), size=hw, mode="bilinear", align_corners=False)
            inten = inten.view(b, k, 1, hw, hw)
        return fmap, inten

    def mu_at(self, pts, fmap, inten, abs_obs, obs_mask, geo) -> torch.Tensor:
        """pts [B,N,3] → μ [B,N,1] ≥ 0."""
        b, n, _ = pts.shape
        k = fmap.size(1)
        acc_f = pts.new_zeros(b, self.feat_ch, n)
        acc_w = pts.new_zeros(b, 1, n)
        acc_i = pts.new_zeros(b, 1, n)
        acc_i2 = pts.new_zeros(b, 1, n)
        max_f = pts.new_full((b, self.feat_ch, n), -1e4)
        for ki in range(k):
            if obs_mask is not None and not bool((obs_mask[:, ki] > 0.5).any()):
                continue
            uv, valid = world_to_uv(pts, abs_obs[:, ki], geo, out_hw=self.ray_hw)
            if obs_mask is not None:
                valid = valid & (obs_mask[:, ki] > 0.5)[:, None]
            w = valid.to(dtype=pts.dtype)[:, None, :]
            grid = uv.view(b, n, 1, 2)
            sf = F.grid_sample(fmap[:, ki], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            si = F.grid_sample(inten[:, ki], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sf = sf.view(b, self.feat_ch, n)
            si = si.view(b, 1, n)
            acc_f = acc_f + sf * w
            acc_i = acc_i + si * w
            acc_i2 = acc_i2 + (si * si) * w
            acc_w = acc_w + w
            max_f = torch.maximum(max_f, sf + (w - 1.0) * 1e4)
        wsum = acc_w.clamp_min(1e-4)
        mean_f = acc_f / wsum
        mean_i = acc_i / wsum
        var_i = (acc_i2 / wsum - mean_i * mean_i).clamp_min(0).sqrt()
        vis = (acc_w > 0.5).to(dtype=pts.dtype)
        cnt = (acc_w / max(k, 1)).clamp(0, 1)
        max_f = torch.where(acc_w > 0.5, max_f, mean_f)
        pe = _fourier_xyz(pts, n_freq=6)
        h = torch.cat(
            [
                mean_f.transpose(1, 2),
                max_f.transpose(1, 2),
                mean_i.transpose(1, 2),
                var_i.transpose(1, 2),
                vis.transpose(1, 2),
                cnt.transpose(1, 2),
                pe,
            ],
            dim=-1,
        )
        return F.softplus(self.mu_mlp(h))

    def render(self, patches, obs_views, abs_obs, obs_mask, query_abs, geo, geo_id=0, leave_one_out=False):
        """Integrate μ along query rays. query_abs [B,Q]. Returns [B,Q,1,H,W] in [0,1]."""
        if patches.size(1) != obs_views.size(1):
            k = min(patches.size(1), obs_views.size(1))
            patches = patches[:, :k]
            obs_views = obs_views[:, :k]
            abs_obs = abs_obs[:, :k]
            if obs_mask is not None:
                obs_mask = obs_mask[:, :k]
        fmap, inten = self._obs_maps(patches, obs_views)
        b, q = query_abs.shape
        hw, m = self.ray_hw, self.n_samples
        pts, dt, valid = sample_cone_rays(
            query_abs.float(), geo, hw, m, patches.device, torch.float32
        )
        outs = []
        for qi in range(q):
            mask_k = obs_mask
            if leave_one_out:
                if mask_k is None:
                    mask_k = abs_obs.new_ones(b, abs_obs.size(1))
                else:
                    mask_k = mask_k.clone()
                d = (query_abs[:, qi : qi + 1] - abs_obs).abs()
                d = torch.minimum(d, d.new_tensor(2 * 3.14159265) - d)
                drop = d.argmin(dim=1)
                mask_k[torch.arange(b, device=mask_k.device), drop] = 0
            mu = self.mu_at(
                pts[:, qi].reshape(b, hw * hw * m, 3), fmap.float(), inten.float(), abs_obs.float(), mask_k, geo
            )
            mu = mu.view(b, 1, hw, hw, m)
            line = (
                mu * valid[:, qi, None, ..., None].to(mu.dtype) * dt[:, qi, None, ..., None].to(mu.dtype)
            ).sum(-1)
            outs.append(line)
        pred = torch.stack(outs, 1)
        pred = torch.flip(pred, dims=[-2])
        gid = int(geo_id)
        pred = pred * self.proj_scale[gid].clamp(0.5, 20.0) + self.proj_bias[gid].clamp(-0.2, 0.2)
        return pred.clamp(0, 1)

    def volume_preview(self, patches, obs_views, abs_obs, obs_mask, geo) -> torch.Tensor:
        fmap, inten = self._obs_maps(patches, obs_views)
        b = patches.size(0)
        r = self.vol_preview
        pts = voxel_centers(r, patches.device, patches.dtype).expand(b, -1, -1)
        mu = self.mu_at(pts, fmap, inten, abs_obs, obs_mask, geo)
        mx = mu.amax(dim=1, keepdim=True).clamp_min(1e-4)
        return (mu / mx).view(b, 1, r, r, r)


class ViewPredictor(nn.Module):
    """X-WIN g_v, adapted to K observed views + explicit query mask tokens.

    X-WIN forward_view: action token + source patches → transformed patches (then they mean-pool).
    We keep the action+context transformer, but also decode 14×14 query tokens with 2D sincos
    so a pixel head can emit a measurement instead of a pooled embedding.
    """

    def __init__(self, dim=768, pred_dim=384, depth=4, n_patches=196, n_heads=12):
        super().__init__()
        self.n_patches = n_patches
        self.pred_dim = pred_dim
        self.ctx_proj = nn.Linear(dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, dim)
        self.action_mlp = nn.Sequential(nn.Linear(64, pred_dim), nn.GELU(), nn.Linear(pred_dim, pred_dim))
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, pred_dim))
        pos = sincos_2d(n_patches, pred_dim, device="cpu", dtype=torch.float32)
        self.register_buffer("pos2d", pos.view(1, n_patches, pred_dim), persistent=False)
        self.blocks = nn.ModuleList([Block(pred_dim, n_heads, 4.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        nn.init.trunc_normal_(self.mask_tok, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        query_rel: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        obs_rel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """patches [B,V,N,C]; query_rel [B,Q] → [B,Q,N,C_enc]"""
        b, v, n, c = patches.shape
        q = query_rel.shape[1]
        ctx = self.ctx_proj(patches) + self.pos2d.to(dtype=patches.dtype)
        if obs_mask is not None:
            ctx = ctx * obs_mask[:, :, None, None]
        ctx = ctx.reshape(b, v * n, self.pred_dim)
        ctx = ctx.unsqueeze(1).expand(b, q, v * n, self.pred_dim).reshape(b * q, v * n, self.pred_dim)
        act = self.action_mlp(fourier_angle(query_rel.reshape(b * q)))
        tgt = self.mask_tok.expand(b * q, self.n_patches, -1) + self.pos2d.to(dtype=patches.dtype)
        x = torch.cat([act.unsqueeze(1), ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_proj(x[:, -self.n_patches :]).view(b, q, self.n_patches, -1)


class XWinViewPredictor(nn.Module):
    """Paper g_v: Linear(a) ⊕ source patches, transform in place (forward_view).

    Action is the signed yaw from the nearest observed view to the query, as in X-WIN.
    Extra observed views are appended as context (incomplete orbit). Output tokens are
    the transformed source patches, not a separate mask stream.
    """

    def __init__(self, dim=768, pred_dim=384, depth=6, n_patches=196, n_heads=12):
        super().__init__()
        self.n_patches = n_patches
        self.pred_dim = pred_dim
        self.ctx_proj = nn.Linear(dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, dim)
        self.action_encoder = nn.Linear(1, pred_dim)
        pos = sincos_2d(n_patches, pred_dim, device="cpu", dtype=torch.float32)
        self.register_buffer("pos2d", pos.view(1, n_patches, pred_dim), persistent=False)
        self.blocks = nn.ModuleList([Block(pred_dim, n_heads, 4.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)

    def forward(
        self,
        patches: torch.Tensor,
        query_rel: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        obs_rel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, v, n, c = patches.shape
        q = query_rel.shape[1]
        if obs_rel is None:
            obs_rel = query_rel.new_zeros(b, v)
        dist = (query_rel.unsqueeze(-1) - obs_rel.unsqueeze(1) + 3.14159265) % (2 * 3.14159265) - 3.14159265
        adist = dist.abs()
        if obs_mask is not None:
            adist = adist.masked_fill(obs_mask.unsqueeze(1) < 0.5, 1e6)
        nn_idx = adist.argmin(dim=-1)
        act = dist.gather(2, nn_idx.unsqueeze(-1)).squeeze(-1)
        src = torch.gather(
            patches.reshape(b, v, n * c),
            1,
            nn_idx.unsqueeze(-1).expand(b, q, n * c),
        ).view(b, q, n, c)
        src_e = self.ctx_proj(src) + self.pos2d.to(dtype=patches.dtype)
        src_e = src_e.reshape(b * q, n, self.pred_dim)
        ctx = self.ctx_proj(patches) + self.pos2d.to(dtype=patches.dtype)
        if obs_mask is not None:
            ctx = ctx * obs_mask[:, :, None, None]
        ctx = ctx.reshape(b, 1, v * n, self.pred_dim).expand(b, q, v * n, self.pred_dim)
        ctx = ctx.reshape(b * q, v * n, self.pred_dim)
        a = self.action_encoder(act.reshape(b * q, 1)).unsqueeze(1)
        x = torch.cat([a, src_e, ctx], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_proj(x[:, 1 : 1 + n]).view(b, q, n, -1)

    def forward_one(self, patches_1: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Official forward_view: one source [B,N,C] + signed yaw [B] → [B,N,C]."""
        if patches_1.dim() == 4:
            patches_1 = patches_1[:, 0]
        n = patches_1.size(1)
        x = self.ctx_proj(patches_1) + self.pos2d.to(dtype=patches_1.dtype)
        a = self.action_encoder(action.reshape(-1, 1)).unsqueeze(1)
        x = torch.cat([a, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_proj(x[:, 1 : 1 + n])


class VolUNet(nn.Module):
    """Legacy light residual UNet (kept for loading old s2 ckpts)."""

    def __init__(self, in_ch=2, base=16):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv3d(a, b, 3, padding=1),
                nn.InstanceNorm3d(b),
                nn.GELU(),
                nn.Conv3d(b, b, 3, padding=1),
                nn.InstanceNorm3d(b),
                nn.GELU(),
            )

        self.e1 = block(in_ch, base)
        self.p1 = nn.MaxPool3d(2)
        self.e2 = block(base, base * 2)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = block(base * 2, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, fdk_obs: torch.Tensor, fdk_fill: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fdk_obs, fdk_fill], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        y = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        y = self.d1(torch.cat([self.u1(y), e1], dim=1))
        return (fdk_fill + self.out(y)).clamp(0, 1)


class VolDetailRefiner(nn.Module):
    """OrbitRefine Stage-2: global multi-scale refine + local residual enhancer.

    Framed as our volumetric detail refinement (not an off-the-shelf 2D GAN name).
    Input: FDK_obs, FDK_fill → sharp volume in [0,1].
    Uses GroupNorm (AMP-safe) and float32 refine path — InstanceNorm3d on near-constant
    FDK backgrounds often NaNs under fp16 autocast.
    """

    def __init__(self, in_ch=2, base=16):
        super().__init__()

        def gn(ch: int) -> nn.GroupNorm:
            # Prefer 8 groups; fall back so ch % g == 0.
            for g in (8, 4, 2, 1):
                if ch % g == 0:
                    return nn.GroupNorm(g, ch)
            return nn.GroupNorm(1, ch)

        def block(a, b):
            return nn.Sequential(
                nn.Conv3d(a, b, 3, padding=1),
                gn(b),
                nn.GELU(),
                nn.Conv3d(b, b, 3, padding=1),
                gn(b),
                nn.GELU(),
            )

        # global pathway (coarser context)
        self.e1 = block(in_ch, base)
        self.p1 = nn.MaxPool3d(2)
        self.e2 = block(base, base * 2)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = block(base * 2, base * 4)
        self.p3 = nn.MaxPool3d(2)
        self.e4 = block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)
        self.g_out = nn.Conv3d(base, 1, 1)
        nn.init.zeros_(self.g_out.weight)
        nn.init.zeros_(self.g_out.bias)

        # local enhancer on full-res residual (high-frequency)
        self.local = nn.Sequential(
            nn.Conv3d(in_ch + 1, base, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(base, base, 3, padding=1),
            gn(base),
            nn.GELU(),
            nn.Conv3d(base, 1, 3, padding=1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, fdk_obs: torch.Tensor, fdk_fill: torch.Tensor) -> torch.Tensor:
        # Refine in fp32 for numerical stability under autocast.
        out_dtype = fdk_fill.dtype
        fdk_obs = torch.nan_to_num(fdk_obs.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        fdk_fill = torch.nan_to_num(fdk_fill.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        x = torch.cat([fdk_obs, fdk_fill], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))
        y = self.d3(torch.cat([self.u3(e4), e3], dim=1))
        y = self.d2(torch.cat([self.u2(y), e2], dim=1))
        y = self.d1(torch.cat([self.u1(y), e1], dim=1))
        coarse = (fdk_fill + self.g_out(y)).clamp(0, 1)
        detail = self.local(torch.cat([x, coarse], dim=1))
        return (coarse + 0.5 * detail).clamp(0, 1).to(out_dtype)


class SlicePatchDisc(nn.Module):
    """2.5D slice Patch discriminator for volumetric detail (ax/cor/sag crops)."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTOrbitWorld(nn.Module):
    def __init__(
        self,
        mae_ckpt: str,
        encoder_name: str = "mae",
        img_size: int = 224,
        fuse_depth: int = 4,
        vol_res: int = 64,
        proj_hw: int = 224,
        vol_hw: int = 128,
        n_samples: int = 32,
        geos: Optional[dict[str, dict]] = None,
        pred_depth: int = 4,
        pred_dim: int = 384,
        ema_m: float = 0.996,
        max_obs_proj: int = 4,
        use_world: bool = True,
        skip_mode: str = "lerp",
        pixel_aux: bool = False,
        mix_wide: bool = False,
        world_kind: str = "bp",
        pred_kind: str = "mask",
        fuse_tau: float = 0.10,
        fuse_topk: int = 2,
        drop_obs_p: float = 0.0,
        near_mix: bool = True,
        use_vol_refine: bool = False,
        vol_detach_2d: bool = False,
        vol_refine_kind: str = "detail",
        use_vol_disc: bool = False,
        use_fiber: bool = False,
        fdk_invert: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.proj_hw = proj_hw
        self.vol_hw = vol_hw
        self.n_samples = n_samples
        self.vol_res = vol_res
        self.ema_m = ema_m
        self.max_obs_proj = max_obs_proj
        self.use_world = use_world
        self.skip_mode = skip_mode
        self.pixel_aux = pixel_aux
        self.world_kind = world_kind
        self.pred_kind = pred_kind
        self.fuse_topk = int(fuse_topk)
        self.drop_obs_p = float(drop_obs_p)
        self.near_mix = bool(near_mix)
        self.use_fiber = bool(use_fiber)
        self.fuse_tau = nn.Parameter(torch.tensor(float(fuse_tau)))
        self.encoder = build_encoder(encoder_name, mae_ckpt=mae_ckpt, img_size=img_size)
        dim = int(self.encoder.embed_dim)
        self.n_patches = (img_size // 16) ** 2
        self.view_angle_mlp = nn.Sequential(nn.Linear(64, dim), nn.GELU(), nn.Linear(dim, dim))
        pred_cls = XWinViewPredictor if pred_kind in ("xwin", "ensemble") else ViewPredictor
        self.predictor = pred_cls(
            dim=dim, pred_dim=pred_dim, depth=pred_depth, n_patches=self.n_patches
        )
        self.pixel = PixelHead(dim=dim, out_size=proj_hw, use_aux=pixel_aux)
        self.flow = FlowHead(dim=dim, out_size=proj_hw)
        self.scale_head = ScaleHead(dim=dim, out_size=proj_hw) if self.use_fiber else None
        if use_world:
            if world_kind == "ray":
                self.world = RayWorld(dim=dim, feat_ch=16, ray_hw=vol_hw, n_samples=n_samples)
            else:
                self.world = WorldUnproject(dim=dim, feat_ch=8, res=vol_res, kind=world_kind)
            self.proj_scale = nn.Parameter(torch.ones(2) * 3.3)
            self.proj_bias = nn.Parameter(torch.zeros(2))
            self.geo_mix = GeoMixWide() if mix_wide else GeoMixHead()
            self.mix_wide = mix_wide
            self.blend_lo = 15.0
            self.blend_hi = 35.0
        else:
            self.world = None
        self.use_vol_refine = bool(use_vol_refine)
        self.vol_detach_2d = bool(vol_detach_2d)
        self.vol_refine_kind = str(vol_refine_kind)
        self.fdk_invert = bool(fdk_invert)
        if self.use_vol_refine:
            if self.vol_refine_kind == "unet":
                self.vol_unet = VolUNet()
            else:
                self.vol_unet = VolDetailRefiner()
            self.vol_disc = SlicePatchDisc() if use_vol_disc else None
        else:
            self.vol_unet = None
            self.vol_disc = None
        self.geos = geos or {}
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def freeze_2d(self) -> None:
        for name in ("encoder", "predictor", "pixel", "flow", "view_angle_mlp", "scale_head"):
            mod = getattr(self, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
        self._2d_frozen = True

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.view_angle_mlp.parameters():
            p.requires_grad = False
        self._enc_frozen = True

    def freeze_world(self) -> None:
        if self.world is not None:
            for p in self.world.parameters():
                p.requires_grad = False
            self.proj_scale.requires_grad = False
            self.proj_bias.requires_grad = False
        self._world_frozen = True

    @torch.no_grad()
    def update_ema(self):
        m = self.ema_m
        for t, s in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1 - m)

    def _prep(self, views: torch.Tensor) -> torch.Tensor:
        b, v, _, h, w = views.shape
        x = views.reshape(b * v, 1, h, w)
        if h != self.img_size or w != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x

    def encode_obs(self, views, rel, mask):
        b, v = views.shape[:2]
        enc_frozen = getattr(self, "_2d_frozen", False) or getattr(self, "_enc_frozen", False)
        ctx = torch.no_grad() if enc_frozen else torch.enable_grad()
        with ctx:
            tok = self.encoder(self._prep(views))
            tok = tok + self.view_angle_mlp(fourier_angle(rel)).reshape(b * v, 1, -1)
        tok = tok.view(b, v, tok.size(1), -1)
        if mask is not None:
            tok = tok * mask[:, :, None, None]
        patches = tok[:, :, 1:, :]
        return patches

    def project_volume(self, vol: torch.Tensor, angles: torch.Tensor, geo_ids: torch.Tensor) -> torch.Tensor:
        b, q = angles.shape
        vol_f, ang_f = vol.float(), angles.float()
        if not (geo_ids == geo_ids[:1]).all():
            return torch.cat(
                [self.project_volume(vol[i : i + 1], angles[i : i + 1], geo_ids[i : i + 1]) for i in range(b)],
                dim=0,
            )
        gid = int(geo_ids[0].item())
        key = "luna" if gid == 0 else "tooth"
        pred = cone_project(
            vol_f, ang_f, self.geos[key], n_samples=self.n_samples, out_hw=self.vol_hw
        )
        pred = pred * self.proj_scale[gid].clamp(1.8, 5.5) + self.proj_bias[gid].clamp(-0.15, 0.15)
        pred = pred.clamp(0, 1)
        if pred.shape[-1] != self.proj_hw:
            pq = pred.flatten(0, 1)
            pq = F.interpolate(pq, size=self.proj_hw, mode="bilinear", align_corners=False)
            pred = pq.view(b, q, 1, self.proj_hw, self.proj_hw)
        return pred.to(dtype=vol.dtype)

    def _upsample_proj(self, pred: torch.Tensor) -> torch.Tensor:
        b, q, _, h, w = pred.shape
        if h == self.proj_hw and w == self.proj_hw:
            return pred
        y = F.interpolate(pred.flatten(0, 1), size=self.proj_hw, mode="bilinear", align_corners=False)
        return y.view(b, q, 1, self.proj_hw, self.proj_hw)

    def _world_one(self, patches, obs_views, abs_obs, obs_mask, abs_q, gid: int):
        key = "luna" if gid == 0 else "tooth"
        geo = self.geos[key]
        ids = abs_q.new_full((patches.size(0),), gid).long()
        if self.world_kind == "ray":
            pred_q = self.world.render(patches, obs_views, abs_obs, obs_mask, abs_q, geo, geo_id=gid)
            vol = self.world.volume_preview(patches, obs_views, abs_obs, obs_mask, geo)
            ov, _, oa, om = self._subset_obs(obs_views, abs_obs, abs_obs, obs_mask)
            kcap = ov.size(1)
            pred_o = self.world.render(
                patches[:, :kcap], ov, oa, om, oa, geo, geo_id=gid, leave_one_out=True
            )
            return self._upsample_proj(pred_q), vol, self._upsample_proj(pred_o)
        vol = self.world(patches, obs_views, abs_obs, obs_mask, geo)
        pred_q = self.project_volume(vol, abs_q, ids)
        ov, _, oa, om = self._subset_obs(obs_views, abs_obs, abs_obs, obs_mask)
        pred_o = self.project_volume(vol, oa, ids)
        return pred_q, vol, pred_o

    def _world_forward(self, patches, obs_views, abs_obs, obs_mask, abs_q, geo_ids):
        if (geo_ids == geo_ids[:1]).all():
            return self._world_one(patches, obs_views, abs_obs, obs_mask, abs_q, int(geo_ids[0]))
        qs, vs, os = [], [], []
        for i in range(patches.size(0)):
            m_i = None if obs_mask is None else obs_mask[i : i + 1]
            pq, vol, po = self._world_one(
                patches[i : i + 1],
                obs_views[i : i + 1],
                abs_obs[i : i + 1],
                m_i,
                abs_q[i : i + 1],
                int(geo_ids[i]),
            )
            qs.append(pq)
            vs.append(vol)
            os.append(po)
        return torch.cat(qs, 0), torch.cat(vs, 0), torch.cat(os, 0)

    def _subset_obs(self, views, rel, abs_ang, mask):
        kcap = min(self.max_obs_proj, views.size(1))
        if views.size(1) <= kcap:
            return views, rel, abs_ang, mask
        m = None if mask is None else mask[:, :kcap]
        return views[:, :kcap], rel[:, :kcap], abs_ang[:, :kcap], m

    def _fdk_one(self, views, angles, mask, gid: int):
        key = "luna" if gid == 0 else "tooth"
        return fdk_backproject(views, angles, self.geos[key], self.vol_res, mask)

    def _vol_refine(self, obs_views, pred_pix, abs_obs, abs_q, obs_mask, geo_ids):
        """FDK(obs) and FDK(obs+pred views) → 3D UNet. Two-channel de-artifacting."""
        b = obs_views.size(0)
        if not (geo_ids == geo_ids[:1]).all():
            outs = [
                self._vol_refine(
                    obs_views[i : i + 1],
                    pred_pix[i : i + 1],
                    abs_obs[i : i + 1],
                    abs_q[i : i + 1],
                    None if obs_mask is None else obs_mask[i : i + 1],
                    geo_ids[i : i + 1],
                )
                for i in range(b)
            ]
            return tuple(torch.cat(xs, 0) for xs in zip(*outs))
        gid = int(geo_ids[0])
        if obs_mask is None:
            obs_mask = obs_views.new_ones(b, obs_views.size(1))
        pred_for_fdk = pred_pix.detach() if self.vol_detach_2d else pred_pix
        with torch.amp.autocast("cuda", enabled=False):
            ov = obs_views.float()
            pv = pred_for_fdk.float()
            if pv.shape[-2:] != ov.shape[-2:]:
                pq = F.interpolate(pv.flatten(0, 1), size=ov.shape[-2:], mode="bilinear", align_corners=False)
                pv = pq.view(pv.size(0), pv.size(1), 1, ov.size(-2), ov.size(-1))
            ao = abs_obs.float()
            aq = abs_q.float()
            om = obs_mask.float()
            fdk_obs = unit_range(self._fdk_one(ov, ao, om, gid))
            fill_v = torch.cat([ov, pv], dim=1)
            fill_a = torch.cat([ao, aq], dim=1)
            fill_m = torch.cat([om, pv.new_ones(b, pv.size(1))], dim=1)
            fdk_fill = unit_range(self._fdk_one(fill_v, fill_a, fill_m, gid))
            if self.fdk_invert:
                fdk_obs = 1.0 - fdk_obs
                fdk_fill = 1.0 - fdk_fill
        vol = self.vol_unet(fdk_obs, fdk_fill)
        return vol, fdk_obs, fdk_fill

    def _maybe_drop_obs(self, obs_mask, obs_rel, query_rel):
        """Train-time K=1 dropout: force the official single-view X-WIN path."""
        if (not self.training) or self.drop_obs_p <= 0 or obs_mask is None:
            return obs_mask
        b = obs_mask.size(0)
        dist = (query_rel[:, :1] - obs_rel.unsqueeze(1) + math.pi) % (2 * math.pi) - math.pi
        adist = dist.abs().squeeze(1).masked_fill(obs_mask < 0.5, 1e6)
        nn_idx = adist.argmin(dim=-1)
        k1 = torch.zeros_like(obs_mask)
        k1.scatter_(1, nn_idx.unsqueeze(1), 1.0)
        use = (torch.rand(b, device=obs_mask.device) < self.drop_obs_p).float().unsqueeze(1)
        return obs_mask * (use * k1 + (1.0 - use))

    def _ensemble_xwin(self, patches, obs_views, obs_rel, query_rel, obs_mask):
        """Independent X-WIN g_v per observed view, then angular softmax fusion.

        K=1 reduces to official X-WIN (one source + signed yaw). Extra views cannot
        be ignored: each branch is a full single-view transform of that measurement.
        """
        b, k, n, c = patches.shape
        q = query_rel.shape[1]
        delta = (query_rel.unsqueeze(-1) - obs_rel.unsqueeze(1) + math.pi) % (2 * math.pi) - math.pi
        adist = delta.abs()
        if obs_mask is not None:
            invalid = obs_mask.unsqueeze(1) < 0.5
            adist = adist.masked_fill(invalid, 1e6)
            delta = delta.masked_fill(invalid, 0.0)
        bqk = b * q * k
        src = patches[:, None].expand(b, q, k, n, c).reshape(bqk, n, c)
        act = delta.reshape(bqk)
        tok = self.predictor.forward_one(src, act)
        tok_q = tok.view(bqk, 1, n, c)
        hw = self.proj_hw
        skip = obs_views
        if skip.shape[-1] != hw or skip.shape[-2] != hw:
            skip = F.interpolate(skip.flatten(0, 1), size=hw, mode="bilinear", align_corners=False)
            skip = skip.view(b, k, 1, hw, hw)
        skip = skip[:, None].expand(b, q, k, 1, hw, hw).reshape(bqk, 1, 1, hw, hw)
        gap = (adist * (180.0 / math.pi)).reshape(bqk, 1)
        if self.skip_mode != "none":
            skip = warp_imgs(skip, self.flow(tok_q, gap_deg=gap))
        imgs = self.pixel(
            tok_q,
            skip=skip if self.skip_mode != "none" else None,
            gap_deg=gap,
        )
        imgs = imgs.view(b, q, k, 1, hw, hw)
        tok = tok.view(b, q, k, n, c)
        # Direct tau in radians (NOT softplus — v8's softplus(0.35)≈30° mixed away near-angle).
        tau = self.fuse_tau.abs().clamp(0.04, 0.22)
        fuse_d = adist
        kk = min(self.fuse_topk, k) if self.fuse_topk > 0 else k
        if kk < k:
            _, idx = adist.topk(kk, dim=-1, largest=False)
            topm = torch.zeros_like(adist)
            topm.scatter_(-1, idx, 1.0)
            fuse_d = adist.masked_fill(topm < 0.5, 1e6)
        w = torch.softmax(-fuse_d / tau, dim=-1)
        if obs_mask is not None:
            w = w * obs_mask.unsqueeze(1)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pred = (imgs * w[..., None, None, None]).sum(dim=2)
        pred_patch = (tok * w[..., None, None]).sum(dim=2)
        skip_f = (skip.view(b, q, k, 1, hw, hw) * w[..., None, None, None]).sum(dim=2)
        return pred, pred_patch, skip_f, imgs

    def forward(
        self,
        obs_views,
        obs_rel,
        query_rel,
        geo_ids,
        obs_mask=None,
        abs_obs=None,
        a_ref=None,
        query_views=None,
    ):
        frozen = getattr(self, "_2d_frozen", False)
        obs_mask = self._maybe_drop_obs(obs_mask, obs_rel, query_rel)
        patches = self.encode_obs(obs_views, obs_rel, obs_mask)
        skip_pack = nearest_and_lerp(obs_views, obs_rel, query_rel, obs_mask, self.proj_hw)
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        pred_pix_ready = None
        view_imgs = None
        with ctx:
            if self.pred_kind == "ensemble":
                pred_pix_ready, pred_patch, skip, view_imgs = self._ensemble_xwin(
                    patches, obs_views, obs_rel, query_rel, obs_mask
                )
            else:
                pred_patch = self.predictor(patches, query_rel, obs_mask, obs_rel=obs_rel)
                skip = skip_pack["lerp"] if self.skip_mode == "lerp" else skip_pack["nn"]
                if self.skip_mode != "none":
                    skip = warp_imgs(skip, self.flow(pred_patch, gap_deg=skip_pack["gap_deg"]))
        if frozen:
            pred_patch = pred_patch.detach()
            skip = skip.detach()
            if pred_pix_ready is not None:
                pred_pix_ready = pred_pix_ready.detach()
        out = {
            "pred_patch": pred_patch,
            "skip": skip,
            "gap_deg": skip_pack["gap_deg"],
        }
        if view_imgs is not None and not self.training:
            out["view_preds"] = view_imgs
        pred_3d = None
        vol = None
        if self.use_world:
            if abs_obs is None:
                raise ValueError("abs_obs is required when use_world=True")
            abs_q = query_rel + a_ref.unsqueeze(1) if a_ref is not None else query_rel
            wctx = torch.no_grad() if getattr(self, "_world_frozen", False) else torch.enable_grad()
            with wctx:
                pred_3d, vol, pred_obs = self._world_forward(
                    patches, obs_views, abs_obs, obs_mask, abs_q, geo_ids
                )
            if getattr(self, "_world_frozen", False):
                pred_3d = pred_3d.detach()
                vol = vol.detach()
                pred_obs = pred_obs.detach()
            out["pred_query_vol"] = pred_3d
            out["volume"] = vol
            out["pred_obs"] = pred_obs
            _ov, _or, oa, om = self._subset_obs(obs_views, obs_rel, abs_obs, obs_mask)
            out["obs_proj_mask"] = om
            if pred_obs.size(1) != om.size(1):
                out["obs_proj_mask"] = om[:, : pred_obs.size(1)]
        else:
            b = obs_views.size(0)
            out["pred_query_vol"] = skip
            out["volume"] = obs_views.new_zeros(b, 1, 8, 8, 8)
            out["pred_obs"] = skip[:, :1]
            out["obs_proj_mask"] = obs_views.new_ones(b, 1)
        if pred_pix_ready is not None:
            if self.near_mix:
                g = ((skip_pack["gap_deg"] - 8.0) / 12.0).clamp(0, 1)
                g = g.view(*pred_pix_ready.shape[:2], 1, 1, 1)
                pred_pix = (1.0 - g) * skip_pack["lerp"] + g * pred_pix_ready
            else:
                pred_pix = pred_pix_ready
            out["skip"] = skip_pack["lerp"]
        else:
            pix_ctx = torch.no_grad() if frozen else torch.enable_grad()
            with pix_ctx:
                pred_pix = self.pixel(
                    pred_patch,
                    skip=skip if self.skip_mode != "none" else None,
                    gap_deg=skip_pack["gap_deg"],
                    aux=pred_3d if self.pixel_aux else None,
                )
            if frozen:
                pred_pix = pred_pix.detach()
        out["pred_2d"] = pred_pix
        if self.use_world and (not self.pixel_aux) and hasattr(self, "geo_mix"):
            mixed = (
                self.geo_mix(pred_pix, pred_3d, skip, skip_pack["gap_deg"])
                if getattr(self, "mix_wide", False)
                else self.geo_mix(pred_pix, pred_3d, skip_pack["gap_deg"])
            )
            out["pred_query"] = mixed
            out["mix"] = ((skip_pack["gap_deg"] - 12.0) / 28.0).clamp(0, 1)
        else:
            out["pred_query"] = pred_pix
            out["mix"] = skip_pack["gap_deg"] * 0
        if self.use_fiber and self.scale_head is not None:
            tok = pred_patch.detach() if frozen else pred_patch
            log_s = self.scale_head(tok)
            scale = log_s.exp()
            out["log_scale"] = log_s
            out["scale"] = scale
            out["fiber_width"] = scale.flatten(2).mean(-1)
        if self.use_vol_refine:
            if abs_obs is None:
                raise ValueError("abs_obs is required when use_vol_refine=True")
            abs_q = query_rel + a_ref.unsqueeze(1) if a_ref is not None else query_rel
            vol, fdk_obs, fdk_fill = self._vol_refine(
                obs_views, out["pred_query"], abs_obs, abs_q, obs_mask, geo_ids
            )
            out["volume"] = vol
            out["fdk_obs"] = fdk_obs
            out["fdk_fill"] = fdk_fill
        if query_views is not None:
            with torch.no_grad():
                b, qv = query_views.shape[:2]
                qt = self.target_encoder(self._prep(query_views))
                qt = qt[:, 1:, :]
                qt = F.layer_norm(qt, (qt.size(-1),))
            out["target_patch"] = qt.view(b, qv, self.n_patches, -1)
        return out
