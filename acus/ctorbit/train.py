"""DDP train: query P(V̂) + obs reprojection + volume aux. Resume-safe."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctorbit.baselines import nearest_and_lerp, psnr_torch, summarize_psnr
from ctorbit.data import CBCTOrbitDataset, collate_cbct
from ctorbit.model import CTOrbitWorld
from ctorbit.projector import affine_match, cone_project, geo_from_yaml
from ctorbit.vis import plot_curves, prune_old_vis, save_case_panel


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred, gt = pred.flatten(0, 1), gt.flatten(0, 1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(gt, 3, 1, 1)
    sig_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sig_y = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_y ** 2
    sig_xy = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def ssim_value(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(1.0 - ssim_loss(pred, gt).detach())


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / F.mse_loss(pred, gt).clamp_min(1e-10)))


def slice_ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt [B,1,H,W]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(gt, 3, 1, 1)
    sig_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sig_y = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_y ** 2
    sig_xy = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def vol_ssim_loss(pred: torch.Tensor, gt: torch.Tensor, n_slices: int = 8) -> torch.Tensor:
    """Approximate volumetric SSIM via random ax/cor/sag 2D slices. pred/gt [B,1,D,H,W]."""
    b, _, d, h, w = pred.shape
    losses = []
    for _ in range(n_slices):
        axis = int(torch.randint(0, 3, (1,)).item())
        if axis == 0:
            i = int(torch.randint(0, d, (1,)).item())
            p, g = pred[:, :, i], gt[:, :, i]
        elif axis == 1:
            i = int(torch.randint(0, h, (1,)).item())
            p, g = pred[:, :, :, i], gt[:, :, :, i]
        else:
            i = int(torch.randint(0, w, (1,)).item())
            p, g = pred[:, :, :, :, i], gt[:, :, :, :, i]
        losses.append(slice_ssim_loss(p, g))
    return torch.stack(losses).mean()


def sample_vol_slices(vol: torch.Tensor, n: int = 8) -> torch.Tensor:
    """vol [B,1,D,H,W] -> [B*n,1,128,128] random orthoslices."""
    b, _, d, h, w = vol.shape
    outs = []
    for bi in range(b):
        for _ in range(n):
            axis = int(torch.randint(0, 3, (1,), device=vol.device).item())
            if axis == 0:
                i = int(torch.randint(0, d, (1,), device=vol.device).item())
                sl = vol[bi : bi + 1, :, i]
            elif axis == 1:
                i = int(torch.randint(0, h, (1,), device=vol.device).item())
                sl = vol[bi : bi + 1, :, :, i]
            else:
                i = int(torch.randint(0, w, (1,), device=vol.device).item())
                sl = vol[bi : bi + 1, :, :, :, i]
            outs.append(F.interpolate(sl, size=(128, 128), mode="bilinear", align_corners=False))
    return torch.cat(outs, dim=0)


def hf_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    blur_p = F.avg_pool3d(pred, 3, 1, 1)
    blur_g = F.avg_pool3d(gt, 3, 1, 1)
    return F.l1_loss(pred - blur_p, gt - blur_g)


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def resize_views(x: torch.Tensor, hw: int) -> torch.Tensor:
    b, v, c, h, w = x.shape
    if h == hw and w == hw:
        return x
    y = F.interpolate(x.flatten(0, 1), size=(hw, hw), mode="bilinear", align_corners=False)
    return y.view(b, v, c, hw, hw)


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    last = ckpt_dir / "last.pt"
    if last.is_file():
        return last
    pts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def gather_obs_query(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One acquired view, used as a query so the fiber must collapse onto it."""
    mask = batch["obs_mask"]
    views = batch["obs_views"]
    b, k = mask.shape
    score = torch.rand(b, k, device=mask.device).masked_fill(mask < 0.5, -1.0)
    idx = score.argmax(dim=1)
    take = idx.view(b, 1, 1, 1, 1).expand(-1, 1, views.size(2), views.size(3), views.size(4))
    qv = views.gather(1, take)
    qrel = batch["obs_rel"].gather(1, idx.view(b, 1))
    return qv, qrel


def fiber_split(pred: torch.Tensor, gt: torch.Tensor, scale: torch.Tensor, gap: torch.Tensor) -> dict[str, float]:
    """90% Gaussian interval coverage and mean scale, split by angular gap."""
    z = 1.64485
    hit = ((pred - gt).abs() <= z * scale).float().mean(dim=(2, 3, 4))
    width = scale.mean(dim=(2, 3, 4))
    near = gap < 20.0
    far = gap >= 40.0

    def _mean(val: torch.Tensor, m: torch.Tensor) -> float:
        if int(m.sum()) == 0:
            return float("nan")
        return float(val[m].mean())

    return {
        "cover": float(hit.mean()),
        "cover_near": _mean(hit, near),
        "cover_far": _mean(hit, far),
        "width": float(width.mean()),
        "width_near": _mean(width, near),
        "width_far": _mean(width, far),
    }


def prune_ckpts(ckpt_dir: Path, keep: int = 2) -> None:
    pts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in pts[:-keep]:
        p.unlink(missing_ok=True)


@torch.no_grad()
def run_eval(model, loader, device, cfg, vis_dir: Path, step: int, max_batches: int = 32):
    model.eval()
    hw = cfg["model"]["proj_hw"]
    use_world = bool(cfg["model"].get("use_world", True))
    acc = {
        "val_psnr_query": 0.0,
        "val_ssim_query": 0.0,
        "val_psnr_nn": 0.0,
        "val_psnr_lerp": 0.0,
        "val_psnr_far": 0.0,
        "val_psnr_near": 0.0,
        "val_psnr_obs": 0.0,
        "val_psnr_vol": 0.0,
        "val_psnr_3d": 0.0,
        "val_psnr_fdk": 0.0,
        "val_cover": 0.0,
        "val_cover_near": 0.0,
        "val_cover_far": 0.0,
        "val_width": 0.0,
        "val_width_near": 0.0,
        "val_width_far": 0.0,
        "val_obs_width": 0.0,
        "val_obs_psnr": 0.0,
    }
    n_far = n_near = n = vis_n = 0
    n_cnear = n_cfar = n_obs = 0
    use_fiber = bool(cfg["model"].get("use_fiber", False))
    latest = vis_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        qgt = resize_views(batch_t["query_views"], hw)
        bl = nearest_and_lerp(batch_t["obs_views"], batch_t["obs_rel"], batch_t["query_rel"], batch_t["obs_mask"], hw)
        out = raw(
            batch_t["obs_views"],
            batch_t["obs_rel"],
            batch_t["query_rel"],
            batch_t["geo_id"],
            obs_mask=batch_t["obs_mask"],
            abs_obs=batch_t["obs_abs"] if (use_world or bool(cfg["model"].get("use_vol_refine", False))) else None,
            a_ref=batch_t["a_ref"],
            query_views=batch_t["query_views"],
        )
        pred = out["pred_query"].clamp(0, 1)
        if use_fiber and "scale" in out:
            fs = fiber_split(pred, qgt, out["scale"], bl["gap_deg"])
            acc["val_cover"] += fs["cover"]
            acc["val_width"] += fs["width"]
            if fs["cover_near"] == fs["cover_near"]:
                acc["val_cover_near"] += fs["cover_near"]
                acc["val_width_near"] += fs["width_near"]
                n_cnear += 1
            if fs["cover_far"] == fs["cover_far"]:
                acc["val_cover_far"] += fs["cover_far"]
                acc["val_width_far"] += fs["width_far"]
                n_cfar += 1
            qv, qrel = gather_obs_query(batch_t)
            need_abs = use_world or bool(cfg["model"].get("use_vol_refine", False))
            out_c = raw(
                batch_t["obs_views"],
                batch_t["obs_rel"],
                qrel,
                batch_t["geo_id"],
                obs_mask=batch_t["obs_mask"],
                abs_obs=batch_t["obs_abs"] if need_abs else None,
                a_ref=batch_t["a_ref"],
                query_views=None,
            )
            gt_c = resize_views(qv, hw)
            acc["val_obs_width"] += float(out_c["scale"].mean())
            acc["val_obs_psnr"] += float(psnr_torch(out_c["pred_query"].clamp(0, 1), gt_c).mean())
            n_obs += 1
        sm = summarize_psnr(pred, qgt, bl["gap_deg"])
        acc["val_psnr_query"] += sm["psnr"]
        acc["val_ssim_query"] += ssim_value(pred, qgt)
        acc["val_psnr_nn"] += float(psnr_torch(bl["nn"], qgt).mean())
        acc["val_psnr_lerp"] += float(psnr_torch(bl["lerp"], qgt).mean())
        if sm["n_far"]:
            acc["val_psnr_far"] += sm["psnr_far"] * sm["n_far"]
            n_far += sm["n_far"]
        if sm["n_near"]:
            acc["val_psnr_near"] += sm["psnr_near"] * sm["n_near"]
            n_near += sm["n_near"]
        if use_world:
            ogt = resize_views(batch_t["obs_views"], hw)
            pred_o = out["pred_obs"].clamp(0, 1)
            om = out.get("obs_proj_mask", batch_t["obs_mask"])
            ogt_o = ogt[:, : pred_o.size(1)]
            mask = om[:, :, None, None, None]
            acc["val_psnr_obs"] += psnr(pred_o * mask, ogt_o * mask)
            if "pred_query_vol" in out:
                acc["val_psnr_3d"] = acc.get("val_psnr_3d", 0.0) + float(psnr_torch(out["pred_query_vol"].clamp(0, 1), qgt).mean())
        if out["volume"].ndim == 5 and batch_t["gt_vol"].ndim >= 3 and float(batch_t["gt_vol"].max()) > 1e-4:
            gt = batch_t["gt_vol"][:, None] if batch_t["gt_vol"].dim() == 4 else batch_t["gt_vol"].unsqueeze(1)
            if out["volume"].shape[-1] == gt.shape[-1]:
                acc["val_psnr_vol"] += psnr(out["volume"][:, 0], gt[:, 0])
                if "fdk_obs" in out:
                    acc["val_psnr_fdk"] = acc.get("val_psnr_fdk", 0.0) + psnr(out["fdk_obs"][:, 0], gt[:, 0])
        n += 1
        qpv = out.get("pred_query_vol")
        if vis_n < 4:
            b0 = 0
            title = f"step {step} {batch['domain'][b0]} {batch['orbit'][b0]} {batch['name'][b0][:24]}"
            kw = dict(
                obs=batch_t["obs_views"][b0],
                obs_mask=batch_t["obs_mask"][b0],
                q_gt=qgt[b0],
                q_pred=pred[b0],
                vol=out["volume"][b0],
                gt_vol=batch_t["gt_vol"][b0],
                title=title,
                q_pred_vol=None if qpv is None or (not use_world) else qpv[b0],
                fdk_obs=None if "fdk_obs" not in out else out["fdk_obs"][b0],
                fdk_fill=None if "fdk_fill" not in out else out["fdk_fill"][b0],
            )
            save_case_panel(vis_dir / f"step{step:06d}_{batch['domain'][b0]}_{batch['orbit'][b0]}_{vis_n}.png", **kw)
            save_case_panel(latest / f"{vis_n:02d}_{batch['orbit'][b0]}_{batch['domain'][b0]}.png", **kw)
            vis_n += 1
    model.train()
    prune_old_vis(vis_dir, keep_evals=2)
    outm = {k: v / max(n, 1) for k, v in acc.items()}
    outm["val_psnr_far"] = acc["val_psnr_far"] / max(n_far, 1)
    outm["val_psnr_near"] = acc["val_psnr_near"] / max(n_near, 1)
    outm["val_cover_near"] = acc["val_cover_near"] / max(n_cnear, 1)
    outm["val_cover_far"] = acc["val_cover_far"] / max(n_cfar, 1)
    outm["val_width_near"] = acc["val_width_near"] / max(n_cnear, 1)
    outm["val_width_far"] = acc["val_width_far"] / max(n_cfar, 1)
    outm["val_obs_width"] = acc["val_obs_width"] / max(n_obs, 1)
    outm["val_obs_psnr"] = acc["val_obs_psnr"] / max(n_obs, 1)
    outm["val_delta_lerp"] = outm["val_psnr_query"] - outm["val_psnr_lerp"]
    xwin_ref = float(cfg.get("eval", {}).get("xwin_ref", 26.39))
    outm["val_delta_xwin"] = outm["val_psnr_query"] - xwin_ref
    outm["n_eval"] = n
    if outm["val_psnr_query"] >= 30.0:
        outm["gate"] = "PASS"
    elif outm["val_delta_xwin"] > 0.5:
        outm["gate"] = "ABOVE_XWIN"
    elif outm["val_delta_lerp"] > 0.15:
        outm["gate"] = "ABOVE_LERP"
    elif outm["val_delta_lerp"] < -0.15:
        outm["gate"] = "BELOW_BASELINE"
    else:
        outm["gate"] = "TIED_BASELINE"
    return outm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/cbct.yaml"))
    ap.add_argument("--resume", default="auto")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out = Path(cfg["train"]["outdir"])
    ckpt_dir, vis_dir, log_dir = out / "ckpt", out / "vis", out / "logs"
    if is_main():
        for d in (ckpt_dir, vis_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)

    geos = {
        "luna": geo_from_yaml(yaml.safe_load((ROOT / "data_meta/LUNA16/config.yaml").read_text())),
        "tooth": geo_from_yaml(yaml.safe_load((ROOT / "data_meta/ToothFairy/config.yaml").read_text())),
    }
    model = CTOrbitWorld(
        mae_ckpt=str(ROOT / cfg["model"]["mae_ckpt"]),
        encoder_name=cfg["model"]["encoder"],
        img_size=cfg["model"]["img_size"],
        fuse_depth=cfg["model"]["fuse_depth"],
        vol_res=cfg["model"]["vol_res"],
        proj_hw=cfg["model"]["proj_hw"],
        vol_hw=int(cfg["model"].get("vol_hw", 128)),
        n_samples=cfg["model"]["n_samples"],
        geos=geos,
        use_world=bool(cfg["model"].get("use_world", True)),
        skip_mode=str(cfg["model"].get("skip_mode", "lerp")),
        pixel_aux=bool(cfg["model"].get("pixel_aux", False)),
        mix_wide=bool(cfg["model"].get("mix_wide", False)),
        world_kind=str(cfg["model"].get("world_kind", "bp")),
        pred_kind=str(cfg["model"].get("pred_kind", "mask")),
        pred_depth=int(cfg["model"].get("pred_depth", 4)),
        pred_dim=int(cfg["model"].get("pred_dim", 384)),
        fuse_tau=float(cfg["model"].get("fuse_tau", 0.10)),
        fuse_topk=int(cfg["model"].get("fuse_topk", 2)),
        drop_obs_p=float(cfg["model"].get("drop_obs_p", 0.0)),
        near_mix=bool(cfg["model"].get("near_mix", True)),
        use_fiber=bool(cfg["model"].get("use_fiber", False)),
        use_vol_refine=bool(cfg["model"].get("use_vol_refine", False)),
        vol_detach_2d=bool(cfg["model"].get("vol_detach_2d", False)),
        vol_refine_kind=str(cfg["model"].get("vol_refine_kind", "detail")),
        use_vol_disc=bool(cfg["model"].get("use_vol_disc", False)),
        fdk_invert=bool(cfg["model"].get("fdk_invert", False)),
    ).to(device)
    raw_m = model.module if world > 1 else model
    init_from = cfg["train"].get("init_from")
    if init_from:
        ip = Path(init_from)
        if ip.is_file():
            pack = torch.load(ip, map_location="cpu", weights_only=False)
            src = pack["model"]
            dst = raw_m.state_dict()
            filtered = {k: v for k, v in src.items() if k in dst and tuple(v.shape) == tuple(dst[k].shape)}
            missing, unexpected = raw_m.load_state_dict(filtered, strict=False)
            if is_main():
                skipped = [k for k, v in src.items() if k in dst and tuple(v.shape) != tuple(dst[k].shape)]
                print(f"[init_from] {ip} loaded={len(filtered)} missing={len(missing)} skipped_shape={skipped[:6]}")
            raw_m.target_encoder.load_state_dict(raw_m.encoder.state_dict())
    init_vol = cfg["train"].get("init_from_vol")
    if init_vol:
        vp = Path(init_vol)
        if vp.is_file():
            pack = torch.load(vp, map_location="cpu", weights_only=False)
            src = pack["model"] if "model" in pack else pack
            dst = raw_m.state_dict()
            vol_keys = {
                k: v for k, v in src.items()
                if k.startswith("vol_unet") and k in dst and tuple(v.shape) == tuple(dst[k].shape)
            }
            raw_m.load_state_dict(vol_keys, strict=False)
            if is_main():
                print(f"[init_from_vol] {vp} vol_unet_keys={len(vol_keys)}", flush=True)
    if bool(cfg["model"].get("freeze_2d", False)):
        raw_m.freeze_2d()
        if is_main():
            print("[freeze_2d] encoder/predictor/pixel/flow frozen")
    elif bool(cfg["model"].get("freeze_encoder", False)):
        raw_m.freeze_encoder()
        if is_main():
            print("[freeze_encoder] MAE frozen; predictor/pixel/flow trainable")
    if bool(cfg["model"].get("freeze_world", False)):
        raw_m.freeze_world()
        if is_main():
            print("[freeze_world] unproject/proj_scale frozen")
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_m = model.module if world > 1 else model
    enc_ids = set(id(p) for p in raw_m.encoder.parameters())
    p_enc = [p for p in raw_m.encoder.parameters() if p.requires_grad]
    p_rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    # keep disc out of G optimizer when present
    if getattr(raw_m, "vol_disc", None) is not None:
        disc_ids = set(id(p) for p in raw_m.vol_disc.parameters())
        p_rest = [p for p in p_rest if id(p) not in disc_ids]
    groups = []
    if p_enc:
        groups.append({"params": p_enc, "lr": cfg["train"]["lr_encoder"]})
    if p_rest:
        groups.append({"params": p_rest, "lr": cfg["train"]["lr"]})
    if not groups:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])
    opt_d = None
    if getattr(raw_m, "vol_disc", None) is not None:
        opt_d = torch.optim.AdamW(raw_m.vol_disc.parameters(), lr=float(cfg["train"].get("lr_disc", 1e-4)), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and opt_d is not None)

    use_world = bool(cfg["model"].get("use_world", True))
    use_vol = bool(cfg["model"].get("use_vol_refine", False))
    use_fiber = bool(cfg["model"].get("use_fiber", False))
    need_vol = use_world or use_vol
    train_set = CBCTOrbitDataset(
        cfg["data"]["roots"],
        split="train",
        num_query=cfg["data"]["num_query"],
        vol_res=cfg["model"]["vol_res"],
        p_sparse=cfg["data"]["p_sparse"],
        k_min=cfg["data"]["k_min"],
        k_max=cfg["data"]["k_max"],
        arc_min=cfg["data"]["arc_min"],
        arc_max=cfg["data"]["arc_max"],
        load_gt_vol=need_vol,
        far_frac=float(cfg["data"].get("far_frac", 0.5)),
        use_multi_arc=bool(cfg["data"].get("use_multi_arc", True)),
        n_arcs_min=int(cfg["data"].get("n_arcs_min", 2)),
        n_arcs_max=int(cfg["data"].get("n_arcs_max", 3)),
    )
    eval_set = CBCTOrbitDataset(
        cfg["data"]["roots"],
        split="eval",
        num_query=cfg["data"]["num_query"],
        vol_res=cfg["model"]["vol_res"],
        p_sparse=cfg["data"]["p_sparse"],
        k_min=int(cfg["data"].get("eval_k_min", max(int(cfg["data"]["k_min"]), 3))),
        k_max=int(cfg["data"].get("eval_k_max", cfg["data"]["k_max"])),
        arc_min=cfg["data"]["arc_min"],
        arc_max=cfg["data"]["arc_max"],
        load_gt_vol=need_vol,
        far_frac=float(cfg["data"].get("far_frac", 0.5)),
        use_multi_arc=bool(cfg["data"].get("use_multi_arc", True)),
        n_arcs_min=int(cfg["data"].get("n_arcs_min", 2)),
        n_arcs_max=int(cfg["data"].get("n_arcs_max", 3)),
    )
    sampler = DistributedSampler(train_set, shuffle=True) if world > 1 else None
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_cbct,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_cbct)

    step = 0
    best = -1.0
    best_vol = -1.0
    resume_path = None
    if args.resume == "auto":
        resume_path = latest_ckpt(ckpt_dir)
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path and resume_path.is_file():
        pack = torch.load(resume_path, map_location="cpu", weights_only=False)
        (model.module if world > 1 else model).load_state_dict(pack["model"], strict=False)
        try:
            opt.load_state_dict(pack["opt"])
            scaler.load_state_dict(pack["scaler"])
            step = int(pack.get("step", 0))
            best = float(pack.get("best", -1.0))
            best_vol = float(pack.get("best_vol", -1.0))
        except Exception:
            step, best, best_vol = 0, -1.0, -1.0
            if is_main():
                print("[resume] weights only (opt mismatch), start step=0")
        if is_main():
            print(f"[resume] {resume_path} step={step}")

    hw = cfg["model"]["proj_hw"]
    stop = Path(cfg["train"].get("stop_file", str(ROOT / "STOP")))
    log_path = log_dir / "train.jsonl"
    it = iter(loader)
    epoch = 0
    model.train()
    while step < cfg["train"]["max_steps"]:
        if stop.is_file():
            if is_main():
                print("STOP file present, exit")
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            it = iter(loader)
            batch = next(it)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        raw_m = model.module if world > 1 else model
        qgt = resize_views(batch["query_views"], hw)
        ogt = resize_views(batch["obs_views"], hw)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(
                batch["obs_views"],
                batch["obs_rel"],
                batch["query_rel"],
                batch["geo_id"],
                obs_mask=batch["obs_mask"],
                abs_obs=batch["obs_abs"] if (use_world or use_vol) else None,
                a_ref=batch["a_ref"],
                query_views=None if bool(cfg["model"].get("freeze_2d", False)) else batch["query_views"],
            )
            pred_q = out["pred_query"]
            if bool(cfg["model"].get("freeze_2d", False)) and not bool(cfg["model"].get("mix_wide", False)):
                gap_w = (out["gap_deg"] / 30.0).clamp(0.5, 1.5).view(pred_q.size(0), pred_q.size(1), 1, 1, 1)
            else:
                gap_w = (out["gap_deg"] / 25.0).clamp(0.3, 4.0).view(pred_q.size(0), pred_q.size(1), 1, 1, 1)
            l_q = ((pred_q - qgt).abs() * gap_w).mean() + cfg["loss"]["lambda_ssim"] * ssim_loss(pred_q, qgt)
            gx = pred_q[..., 1:, :] - pred_q[..., :-1, :]
            gy = pred_q[..., :, 1:] - pred_q[..., :, :-1]
            l_grad = F.l1_loss(gx, qgt[..., 1:, :]) + F.l1_loss(gy, qgt[..., :, 1:])
            l_nll = pred_q.new_zeros(())
            l_col = pred_q.new_zeros(())
            l_cover = pred_q.new_zeros(())
            l_vol_reproj = pred_q.new_zeros(())
            l_vol_fiber = pred_q.new_zeros(())
            width_now = 0.0
            if use_fiber and "log_scale" in out:
                log_var = (2.0 * out["log_scale"]).clamp(-8.0, math.log(0.35) * 2)
                inv = torch.exp(-log_var)
                l_nll = (0.5 * (log_var + (pred_q - qgt).pow(2) * inv)).mean()
                width_now = float(out["scale"].detach().mean())
                if float(cfg["loss"].get("lambda_collapse", 0.0)) > 0:
                    qv, qrel = gather_obs_query(batch)
                    out_c = model(
                        batch["obs_views"],
                        batch["obs_rel"],
                        qrel,
                        batch["geo_id"],
                        obs_mask=batch["obs_mask"],
                        abs_obs=batch["obs_abs"] if use_vol else None,
                        a_ref=batch["a_ref"],
                        query_views=None,
                    )
                    gt_c = resize_views(qv, hw)
                    l_col = (out_c["pred_query"] - gt_c).abs().mean() + out_c["scale"].mean()
                # Soft 90% coverage: widen scale when the interval is too tight.
                if float(cfg["loss"].get("lambda_cover", 0.0)) > 0:
                    z = 1.64485
                    slack = 8.0 * (z * out["scale"] - (pred_q - qgt).abs())
                    soft = torch.sigmoid(slack).mean()
                    l_cover = torch.relu(pred_q.new_tensor(0.90) - soft)
            pred_aln = F.layer_norm(out["pred_patch"], (out["pred_patch"].size(-1),))
            l_jepa = (
                F.mse_loss(pred_aln, out["target_patch"])
                if "target_patch" in out
                else pred_q.new_zeros(())
            )
            if use_world:
                # Ray world: always supervise P_a(μ). Old BP path kept mix_w so the
                # 15 dB blob could not overwrite near-angle 2D.
                if str(cfg["model"].get("world_kind", "bp")) == "ray":
                    far_w = (0.6 + 1.8 * (out["gap_deg"] / 40.0).clamp(0, 1)).view_as(gap_w)
                    l_proj = ((out["pred_query_vol"] - qgt).abs() * far_w).mean()
                else:
                    gap_w_3d = (out["gap_deg"] / 40.0).clamp(0.4, 1.2).view_as(gap_w)
                    mix_w = out["mix"].detach().reshape_as(gap_w_3d)
                    l_proj = ((out["pred_query_vol"] - qgt).abs() * gap_w_3d * mix_w).mean()
                pred_o = out["pred_obs"]
                om = out.get("obs_proj_mask", batch["obs_mask"])
                ogt_o = ogt[:, : pred_o.size(1)]
                mask = om[:, :, None, None, None]
                err_o = F.l1_loss(pred_o, ogt_o, reduction="none") * mask
                l_o = err_o.sum() / mask.expand_as(pred_o).sum().clamp_min(1)
                l_v = pred_q.new_zeros(())
                l_sp = out["volume"].mean()
            else:
                l_proj = pred_q.new_zeros(())
                l_o = pred_q.new_zeros(())
                l_v = pred_q.new_zeros(())
                l_sp = pred_q.new_zeros(())
                pred_o = out["skip"]
                mask = batch["obs_mask"][:, :1, None, None, None]
                ogt_o = qgt
            l_gan = pred_q.new_zeros(())
            gan_on = False
            if use_vol and "volume" in out:
                gt = batch["gt_vol"]
                if gt.dim() == 4:
                    gt = gt.unsqueeze(1)
                vol = torch.nan_to_num(out["volume"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                l_v = F.l1_loss(vol, gt)
                l_v = l_v + float(cfg["loss"].get("lambda_vol_ssim", 0.0)) * vol_ssim_loss(vol, gt)
                l_v = l_v + float(cfg["loss"].get("lambda_vol_hf", 0.0)) * hf_l1(vol, gt)
                gan_warm = int(cfg["loss"].get("gan_warmup", 500))
                gan_on = (
                    getattr(raw_m, "vol_disc", None) is not None
                    and float(cfg["loss"].get("lambda_gan", 0.0)) > 0
                    and step >= gan_warm
                )
                if gan_on:
                    fake_s = sample_vol_slices(vol, n=int(cfg["loss"].get("gan_slices", 6)))
                    l_gan = torch.relu(1.0 - raw_m.vol_disc(fake_s)).mean()
                # Keep V inside the fiber: match acquired views and land in p(x|a').
                if float(cfg["loss"].get("lambda_vol_reproj", 0.0)) > 0 or float(cfg["loss"].get("lambda_vol_fiber", 0.0)) > 0:
                    gid = int(batch["geo_id"][0])
                    geo = geos["luna" if gid == 0 else "tooth"]
                    ns = int(cfg["model"].get("proj_samples", cfg["model"].get("n_samples", 32)))
                    po = cone_project(vol.float(), batch["obs_abs"].float(), geo, n_samples=ns, out_hw=hw)
                    om = batch["obs_mask"][:, :, None, None, None]
                    po_m = affine_match(po, ogt, om)
                    l_vol_reproj = (F.l1_loss(po_m, ogt, reduction="none") * om).sum() / om.expand_as(po).sum().clamp_min(1)
                    pq = cone_project(vol.float(), batch["query_abs"].float(), geo, n_samples=ns, out_hw=hw)
                    tgt = pred_q.detach()
                    pq_m = affine_match(pq, tgt)
                    l_vol_fiber = F.smooth_l1_loss(pq_m, tgt)
            loss = (
                float(cfg["loss"].get("lambda_point", 1.0)) * l_q
                + cfg["loss"]["lambda_grad"] * l_grad
                + cfg["loss"]["lambda_jepa"] * l_jepa
                + float(cfg["loss"].get("lambda_nll", 0.0)) * l_nll
                + float(cfg["loss"].get("lambda_collapse", 0.0)) * l_col
                + float(cfg["loss"].get("lambda_cover", 0.0)) * l_cover
            )
            if use_world:
                loss = (
                    loss
                    + cfg["loss"]["lambda_proj"] * l_proj
                    + cfg["loss"]["lambda_reproj"] * l_o
                    + cfg["loss"]["lambda_vol"] * l_v
                    + float(cfg["loss"].get("lambda_sparse", 0.01)) * l_sp
                )
            elif use_vol:
                loss = (
                    loss
                    + float(cfg["loss"].get("lambda_vol", 1.0)) * l_v
                    + float(cfg["loss"].get("lambda_gan", 0.0)) * l_gan
                    + float(cfg["loss"].get("lambda_vol_reproj", 0.0)) * l_vol_reproj
                    + float(cfg["loss"].get("lambda_vol_fiber", 0.0)) * l_vol_fiber
                )
            if not torch.isfinite(loss):
                # skip bad batch rather than poisoning GradScaler
                opt.zero_grad(set_to_none=True)
                if is_main() and step % cfg["train"]["log_every"] == 0:
                    print({"step": step, "skip_nonfinite": True}, flush=True)
                continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()

        # discriminator step (slice PatchGAN); warm up after G has stabilized
        if use_vol and opt_d is not None and "volume" in out and gan_on and torch.isfinite(out["volume"]).all():
            gt = batch["gt_vol"]
            if gt.dim() == 4:
                gt = gt.unsqueeze(1)
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                real_s = sample_vol_slices(gt.detach(), n=int(cfg["loss"].get("gan_slices", 6)))
                fake_s = sample_vol_slices(out["volume"].detach(), n=int(cfg["loss"].get("gan_slices", 6)))
                d_real = raw_m.vol_disc(real_s)
                d_fake = raw_m.vol_disc(fake_s)
                loss_d = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
            if torch.isfinite(loss_d):
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                opt_d.zero_grad(set_to_none=True)
                loss_d = pred_q.new_zeros(())
        else:
            loss_d = pred_q.new_zeros(())

        raw_m = model.module if world > 1 else model
        raw_m.update_ema()
        step += 1

        if is_main() and step % cfg["train"]["log_every"] == 0:
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "l_q": float(l_q.detach()),
                "l_nll": float(l_nll.detach()),
                "l_col": float(l_col.detach()),
                "l_cover": float(l_cover.detach()) if torch.is_tensor(l_cover) else 0.0,
                "l_vol_reproj": float(l_vol_reproj.detach()) if torch.is_tensor(l_vol_reproj) else 0.0,
                "l_vol_fiber": float(l_vol_fiber.detach()) if torch.is_tensor(l_vol_fiber) else 0.0,
                "width": width_now,
                "l_o": float(l_o.detach()),
                "l_v": float(l_v.detach()),
                "l_gan": float(l_gan.detach()) if torch.is_tensor(l_gan) else 0.0,
                "l_proj": float(l_proj.detach()),
                "l_jepa": float(l_jepa.detach()),
                "psnr_q": float(psnr_torch(pred_q.detach().float(), qgt).mean()),
                "ssim_q": ssim_value(pred_q.detach(), qgt),
                "psnr_lerp": float(psnr_torch(out["skip"].detach().float(), qgt).mean()),
                "psnr_o": 0.0 if not use_world else psnr((pred_o.detach() * mask), (ogt_o * mask)),
                "psnr_3d": 0.0 if not use_world else float(psnr_torch(out["pred_query_vol"].detach().float(), qgt).mean()),
                "psnr_vol": 0.0 if not use_vol else float(psnr_torch(out["volume"].detach().float(), batch["gt_vol"][:, None] if batch["gt_vol"].dim()==4 else batch["gt_vol"].unsqueeze(1)).mean()),
            }
            print(rec, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            latest = vis_dir / "latest"
            latest.mkdir(parents=True, exist_ok=True)
            save_case_panel(
                latest / "current.png",
                batch["obs_views"][0],
                batch["obs_mask"][0],
                qgt[0],
                pred_q[0].detach(),
                out["volume"][0].detach(),
                batch["gt_vol"][0],
                title=f"train step {step} {batch['domain'][0]} {batch['orbit'][0]}",
                q_pred_vol=None if not use_world else out["pred_query_vol"][0].detach(),
                fdk_obs=None if "fdk_obs" not in out else out["fdk_obs"][0].detach(),
                fdk_fill=None if "fdk_fill" not in out else out["fdk_fill"][0].detach(),
            )
            plot_curves(log_path, latest / "curves.png")

        if is_main() and step % cfg["train"]["eval_every"] == 0:
            val = run_eval(
                model, eval_loader, device, cfg, vis_dir, step,
                max_batches=int(cfg["train"].get("eval_batches", 32)),
            )
            val["step"] = step
            print(val, flush=True)
            print(
                f"[gate] step={step} model={val['val_psnr_query']:.2f} "
                f"nn={val['val_psnr_nn']:.2f} lerp={val['val_psnr_lerp']:.2f} "
                f"near={val['val_psnr_near']:.2f} far={val['val_psnr_far']:.2f} "
                f"3d={val.get('val_psnr_3d', 0):.2f} "
                f"vol={val.get('val_psnr_vol', 0):.2f} fdk={val.get('val_psnr_fdk', 0):.2f} "
                f"delta_lerp={val['val_delta_lerp']:+.2f} delta_xwin={val.get('val_delta_xwin', 0):+.2f} "
                f"width={val.get('val_width', 0):.3f}/{val.get('val_width_near', 0):.3f}/{val.get('val_width_far', 0):.3f} "
                f"cover={val.get('val_cover', 0):.2f} obs_w={val.get('val_obs_width', 0):.4f} "
                f"{val['gate']} target=30",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(val) + "\n")
            plot_curves(log_path, vis_dir / "latest" / "curves.png")
            raw = model.module if world > 1 else model
            vps = float(val["val_psnr_query"])
            payload = {
                "step": step,
                "best": best,
                "best_vol": best_vol,
                "model": raw.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg,
            }
            if vps > best:
                best = vps
                payload["best"] = best
                torch.save(payload, ckpt_dir / "best.pt")
            vvol = float(val.get("val_psnr_vol", -1.0))
            if use_vol and vvol > best_vol and vps >= float(cfg["train"].get("abort_overall", 25.5)):
                best_vol = vvol
                payload["best_vol"] = best_vol
                torch.save(payload, ckpt_dir / "best_vol.pt")
            torch.save(payload, ckpt_dir / "last.pt")
            slim = {"step": step, "best": best, "model": raw.state_dict(), "cfg": cfg}
            torch.save(slim, ckpt_dir / f"step_{step:06d}.pt")
            prune_ckpts(ckpt_dir, keep=cfg["train"]["keep_step_ckpts"])
            abort_after = int(cfg["train"].get("abort_after", 0))
            if abort_after and step >= abort_after:
                near_floor = float(cfg["train"].get("abort_near", 29.0))
                overall_floor = float(cfg["train"].get("abort_overall", 25.5))
                reason = None
                if val["val_psnr_near"] < near_floor:
                    reason = f"near={val['val_psnr_near']:.2f}<{near_floor}"
                elif val["val_psnr_query"] < overall_floor:
                    reason = f"overall={val['val_psnr_query']:.2f}<{overall_floor}"
                if reason:
                    msg = f"[abort] step={step} {reason}"
                    print(msg, flush=True)
                    (log_dir / "ABORT").write_text(msg + "\n")
                    break

        if world > 1:
            dist.barrier()

    if is_main():
        raw = model.module if world > 1 else model
        torch.save(
            {"step": step, "best": best, "model": raw.state_dict(), "opt": opt.state_dict(),
             "scaler": scaler.state_dict(), "cfg": cfg},
            ckpt_dir / "last.pt",
        )
        print("done step", step)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
