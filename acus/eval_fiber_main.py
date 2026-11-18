"""OrbitRefine main results: Table 1 (K=8) and Fig. 2/3.

Protocol: held-out eval cases, all unobserved views as queries.
Near < 20°, far >= 40°. 90% Gaussian interval uses z = 1.64485.
Lerp/NN widths are a single residual-calibrated constant (unlearned fiber).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ctorbit.baselines import nearest_and_lerp, psnr_torch
from ctorbit.data import CBCTOrbitDataset, collate_cbct
from ctorbit.model import CTOrbitWorld
from ctorbit.projector import geo_from_yaml

Z90 = 1.64485
NEAR, FAR = 20.0, 40.0
BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 180)]


def resize_views(x: torch.Tensor, hw: int) -> torch.Tensor:
    b, v, c, h, w = x.shape
    if h == hw and w == hw:
        return x
    y = F.interpolate(x.flatten(0, 1), size=(hw, hw), mode="bilinear", align_corners=False)
    return y.view(b, v, c, hw, hw)


def mean_std(xs: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([x for x in xs if x == x], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), int(arr.size)


def cover_hit(pred, gt, scale) -> torch.Tensor:
    return ((pred - gt).abs() <= Z90 * scale).float().mean(dim=(2, 3, 4))


def build_model(cfg: dict, device, use_fiber: bool, near_mix: bool) -> CTOrbitWorld:
    geos = {
        "luna": geo_from_yaml(yaml.safe_load((ROOT / "data_meta/LUNA16/config.yaml").read_text())),
        "tooth": geo_from_yaml(yaml.safe_load((ROOT / "data_meta/ToothFairy/config.yaml").read_text())),
    }
    m = CTOrbitWorld(
        mae_ckpt=str(ROOT / cfg["model"]["mae_ckpt"]),
        encoder_name=cfg["model"]["encoder"],
        img_size=cfg["model"]["img_size"],
        fuse_depth=cfg["model"]["fuse_depth"],
        vol_res=cfg["model"]["vol_res"],
        proj_hw=cfg["model"]["proj_hw"],
        vol_hw=int(cfg["model"].get("vol_hw", 32)),
        n_samples=cfg["model"]["n_samples"],
        geos=geos,
        use_world=False,
        skip_mode=str(cfg["model"].get("skip_mode", "lerp")),
        pred_kind=str(cfg["model"].get("pred_kind", "ensemble")),
        pred_depth=int(cfg["model"].get("pred_depth", 6)),
        pred_dim=int(cfg["model"].get("pred_dim", 384)),
        fuse_tau=float(cfg["model"].get("fuse_tau", 0.10)),
        fuse_topk=int(cfg["model"].get("fuse_topk", 2)),
        drop_obs_p=0.0,
        near_mix=near_mix,
        use_fiber=use_fiber,
    ).to(device)
    m.eval()
    return m


def load_weights(model, path: Path) -> None:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    src = pack["model"] if "model" in pack else pack
    dst = model.state_dict()
    filtered = {k: v for k, v in src.items() if k in dst and tuple(v.shape) == tuple(dst[k].shape)}
    model.load_state_dict(filtered, strict=False)
    print(f"[load] {path} keys={len(filtered)}/{len(dst)}", flush=True)


@torch.no_grad()
def predict_chunks(model, batch, hw: int, device, q_chunk: int = 6):
    obs = batch["obs_views"].to(device)
    obs_rel = batch["obs_rel"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    qrel = batch["query_rel"].to(device)
    gid = batch["geo_id"].to(device)
    a_ref = batch["a_ref"].to(device)
    qgt = resize_views(batch["query_views"].to(device), hw)
    q = qrel.size(1)
    preds, scales = [], []
    for s in range(0, q, q_chunk):
        e = min(q, s + q_chunk)
        out = model(
            obs, obs_rel, qrel[:, s:e], gid, obs_mask=obs_mask,
            abs_obs=None, a_ref=a_ref, query_views=None,
        )
        preds.append(out["pred_query"].clamp(0, 1))
        if "scale" in out:
            scales.append(out["scale"])
    pred = torch.cat(preds, dim=1)
    scale = torch.cat(scales, dim=1) if scales else None
    bl = nearest_and_lerp(obs, obs_rel, qrel, obs_mask, hw)
    # collapse: query the first valid observed angle
    idx = obs_mask[0].argmax()
    qrel_c = obs_rel[:, idx : idx + 1]
    out_c = model(obs, obs_rel, qrel_c, gid, obs_mask=obs_mask, abs_obs=None, a_ref=a_ref, query_views=None)
    ogt = resize_views(obs[:, idx : idx + 1], hw)
    collapse = {
        "psnr": float(psnr_torch(out_c["pred_query"].clamp(0, 1), ogt).mean()),
        "width": float(out_c["scale"].mean()) if "scale" in out_c else float("nan"),
    }
    return pred, scale, qgt, bl, collapse


def make_loader(cfg, orbit: str, k: int, n_cases: int) -> DataLoader:
    ds = CBCTOrbitDataset(
        cfg["data"]["roots"],
        split="eval",
        num_query=4,
        vol_res=64,
        p_sparse=1.0 if orbit == "sparse" else 0.0,
        k_min=k,
        k_max=k,
        arc_min=25.0,
        arc_max=45.0,
        load_gt_vol=False,
        far_frac=1.0,
        use_multi_arc=True,
        n_arcs_min=2,
        n_arcs_max=3,
        force_k=k,
        force_orbit=orbit,
        query_mode="all_unobs",
    )
    n = min(n_cases, len(ds))
    return DataLoader(
        torch.utils.data.Subset(ds, list(range(n))),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_cbct,
    )


def fit_cover_alpha(err: np.ndarray, scale: np.ndarray, target: float = 0.90) -> float:
    """Scalar temperature so 90% Gaussian intervals hit `target` coverage on cal pixels."""
    lo, hi = 0.4, 6.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        c = float((err <= Z90 * mid * scale).mean())
        if c < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def run_split(fiber, point, cfg, device, orbit: str, k: int, n_cases: int, n_cal: int):
    hw = cfg["model"]["proj_hw"]
    loader = make_loader(cfg, orbit, k, n_cases + n_cal)
    recs = []
    cal_e, cal_s = [], []
    for i, batch in enumerate(loader):
        pred_f, scale_f, qgt, bl, col = predict_chunks(fiber, batch, hw, device)
        pred_p, _, _, _, col_p = predict_chunks(point, batch, hw, device)
        gap = bl["gap_deg"]
        psnr_f = psnr_torch(pred_f, qgt)[0]
        psnr_p = psnr_torch(pred_p, qgt)[0]
        psnr_nn = psnr_torch(bl["nn"], qgt)[0]
        psnr_lp = psnr_torch(bl["lerp"], qgt)[0]
        width_f = scale_f.mean(dim=(2, 3, 4))[0]
        hit_f = cover_hit(pred_f, qgt, scale_f)[0]
        abs_lp = (bl["lerp"] - qgt).abs().mean(dim=(2, 3, 4))[0]
        abs_nn = (bl["nn"] - qgt).abs().mean(dim=(2, 3, 4))[0]
        abs_p = (pred_p - qgt).abs().mean(dim=(2, 3, 4))[0]
        e16 = F.interpolate((pred_f - qgt).abs().flatten(0, 1), size=16, mode="area")
        s16 = F.interpolate(scale_f.flatten(0, 1), size=16, mode="area")
        recs.append({
            "i": i,
            "name": batch["name"][0],
            "gap": gap[0].cpu().numpy(),
            "psnr_f": psnr_f.cpu().numpy(),
            "psnr_p": psnr_p.cpu().numpy(),
            "psnr_nn": psnr_nn.cpu().numpy(),
            "psnr_lp": psnr_lp.cpu().numpy(),
            "width_f": width_f.cpu().numpy(),
            "hit_f": hit_f.cpu().numpy(),
            "e16": e16.cpu().numpy(),
            "s16": s16.cpu().numpy(),
            "abs_lp": abs_lp.cpu().numpy(),
            "abs_nn": abs_nn.cpu().numpy(),
            "abs_p": abs_p.cpu().numpy(),
            "collapse_psnr": col["psnr"],
            "collapse_width": col["width"],
            "point_collapse_psnr": col_p["psnr"],
        })
        if i < n_cal:
            cal_e.append(e16.cpu().numpy().reshape(-1))
            cal_s.append(s16.cpu().numpy().reshape(-1))
        if (i + 1) % 5 == 0:
            print(f"  {orbit} {i+1}/{n_cases+n_cal} {batch['name'][0]} Q={gap.size(1)}", flush=True)
        if i + 1 >= n_cases + n_cal:
            break

    cal, test = recs[:n_cal], recs[n_cal:]
    alpha = 1.0
    if cal_e:
        alpha = fit_cover_alpha(np.concatenate(cal_e), np.concatenate(cal_s), 0.90)
        print(f"  {orbit} cover-temperature alpha={alpha:.3f}", flush=True)
        for r in recs:
            r["width_f"] = r["width_f"] * alpha
            r["collapse_width"] = float(r["collapse_width"]) * alpha
            r["hit_f"] = (r["e16"] <= (Z90 * alpha * r["s16"])).mean(axis=(1, 2, 3)).astype(np.float32)
            r["exceed_f"] = np.maximum(r["e16"] - Z90 * alpha * r["s16"], 0).mean(axis=(1, 2, 3)).astype(np.float32)
            r["acus_f"] = (3.29 * r["width_f"] + 20.0 * r["exceed_f"]).astype(np.float32)
    s_lp = float(np.concatenate([r["abs_lp"] for r in cal]).mean()) / Z90
    s_nn = float(np.concatenate([r["abs_nn"] for r in cal]).mean()) / Z90
    s_p = float(np.concatenate([r["abs_p"] for r in cal]).mean()) / Z90
    s_lp, s_nn, s_p = max(s_lp, 1e-4), max(s_nn, 1e-4), max(s_p, 1e-4)

    def pack_method(name, psnr_key, width_fn, hit_fn, collapse_w, collapse_p):
        near_w, far_w, near_c, far_c, near_p, far_p, obs_w, obs_p = [], [], [], [], [], [], [], []
        bins = {b: {"w": [], "c": [], "p": []} for b in BINS}
        for r in test:
            gap = r["gap"]
            psnr = r[psnr_key]
            width = width_fn(r)
            hit = hit_fn(r)
            near = gap < NEAR
            far = gap >= FAR
            if near.any():
                near_w.append(float(width[near].mean()))
                near_c.append(float(hit[near].mean()))
                near_p.append(float(psnr[near].mean()))
            if far.any():
                far_w.append(float(width[far].mean()))
                far_c.append(float(hit[far].mean()))
                far_p.append(float(psnr[far].mean()))
            obs_w.append(collapse_w(r))
            obs_p.append(collapse_p(r))
            for lo, hi in BINS:
                m = (gap >= lo) & (gap < hi)
                if m.any():
                    bins[(lo, hi)]["w"].append(float(width[m].mean()))
                    bins[(lo, hi)]["c"].append(float(hit[m].mean()))
                    bins[(lo, hi)]["p"].append(float(psnr[m].mean()))
        row = {
            "method": name,
            "n": len(test),
            "width_near": mean_std(near_w),
            "width_far": mean_std(far_w),
            "cover_near": mean_std(near_c),
            "cover_far": mean_std(far_c),
            "psnr_near": mean_std(near_p),
            "psnr_far": mean_std(far_p),
            "obs_width": mean_std(obs_w),
            "obs_psnr": mean_std(obs_p),
            "curve": {
                f"{lo}-{hi}": {
                    "width": mean_std(v["w"]),
                    "cover": mean_std(v["c"]),
                    "psnr": mean_std(v["p"]),
                }
                for (lo, hi), v in bins.items()
            },
        }
        return row

    def hit_const(r, psnr_key, s):
        # reconstruct coverage from stored psnr is wrong; recompute from abs
        key = {"psnr_lp": "abs_lp", "psnr_nn": "abs_nn", "psnr_p": "abs_p"}[psnr_key]
        return (r[key] <= Z90 * s).astype(np.float32)

    table = [
        pack_method(
            "NN", "psnr_nn",
            lambda r: np.full_like(r["gap"], s_nn, dtype=np.float32),
            lambda r: hit_const(r, "psnr_nn", s_nn),
            lambda r: s_nn, lambda r: r["point_collapse_psnr"],
        ),
        pack_method(
            "Lerp", "psnr_lp",
            lambda r: np.full_like(r["gap"], s_lp, dtype=np.float32),
            lambda r: hit_const(r, "psnr_lp", s_lp),
            lambda r: s_lp, lambda r: r["point_collapse_psnr"],
        ),
        pack_method(
            "Point completion", "psnr_p",
            lambda r: np.full_like(r["gap"], s_p, dtype=np.float32),
            lambda r: hit_const(r, "psnr_p", s_p),
            lambda r: s_p, lambda r: r["point_collapse_psnr"],
        ),
        pack_method(
            "OrbitRefine", "psnr_f",
            lambda r: r["width_f"],
            lambda r: r["hit_f"],
            lambda r: r["collapse_width"],
            lambda r: r["collapse_psnr"],
        ),
    ]
    def binned_width(abs_key, global_s):
        scales = {}
        for lo, hi in BINS:
            chunks = []
            for r in cal:
                m = (r["gap"] >= lo) & (r["gap"] < hi)
                if np.any(m):
                    chunks.append(r[abs_key][m])
            scales[(lo, hi)] = max(float(np.concatenate(chunks).mean()) / Z90, 1e-4) if chunks else global_s

        def width_of(r, scales=scales, fallback=global_s):
            w = np.full_like(r["gap"], fallback, dtype=np.float32)
            for (lo, hi), s in scales.items():
                m = (r["gap"] >= lo) & (r["gap"] < hi)
                w[m] = s
            return w

        return width_of

    for label, abs_key, psnr_key, global_s in (
        ("NN-bin", "abs_nn", "psnr_nn", s_nn),
        ("Lerp-bin", "abs_lp", "psnr_lp", s_lp),
        ("Point-bin", "abs_p", "psnr_p", s_p),
    ):
        width_of = binned_width(abs_key, global_s)
        err_key = {"psnr_lp": "abs_lp", "psnr_nn": "abs_nn", "psnr_p": "abs_p"}[psnr_key]
        table.append(pack_method(
            label, psnr_key, width_of,
            lambda r, width_of=width_of, err_key=err_key: (r[err_key] <= Z90 * width_of(r)).astype(np.float32),
            lambda r, width_of=width_of: float(np.mean(width_of(r))),
            lambda r: r["point_collapse_psnr"],
        ))
    acus_vals = []
    for r in test:
        if "acus_f" in r and r["acus_f"].size:
            acus_vals.append(float(np.mean(r["acus_f"])))
    if table:
        table[-4]["acus"] = mean_std(acus_vals) if acus_vals else (float("nan"), float("nan"), 0)
    # Fig. 3: farthest query of first test case
    fig3 = None
    if test:
        r0 = test[0]
        j = int(np.argmax(r0["gap"]))
        fig3 = {"name": r0["name"], "gap": float(r0["gap"][j]), "q_index": j}
    return {
        "orbit": orbit,
        "k": k,
        "n_cal": n_cal,
        "n_test": len(test),
        "calib_scale": {"nn": s_nn, "lerp": s_lp, "point": s_p, "fiber_alpha": alpha},
        "table": table,
        "fig3_hint": fig3,
    }


def fmt_ms(ms):
    m, s, n = ms
    if n == 0 or m != m:
        return "—"
    return f"{m:.3f}±{s:.3f}" if "width" else f"{m:.2f}±{s:.2f}"


def write_markdown(all_out: dict, path: Path) -> None:
    lines = ["# OrbitRefine main results", "", f"Cases per split: test after {all_out.get('n_cal', 8)} calibration cases.", ""]
    for split in all_out["splits"]:
        lines.append(f"## {split['orbit']}, K={split['k']}, N={split['n_test']}")
        lines.append("")
        lines.append("| method | near width | far width | cover near/far | PSNR near/far | obs width | obs PSNR |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in split["table"]:
            def f(key, nd=3):
                m, s, n = row[key]
                if n == 0 or m != m:
                    return "—"
                if nd == 2:
                    return f"{m:.2f}±{s:.2f}"
                return f"{m:.3f}±{s:.3f}"
            lines.append(
                f"| {row['method']} | {f('width_near')} | {f('width_far')} | "
                f"{f('cover_near')}/{f('cover_far')} | {f('psnr_near', 2)}/{f('psnr_far', 2)} | "
                f"{f('obs_width')} | {f('obs_psnr', 2)} |"
            )
        lines.append("")
        lines.append("Fig. 2 bins (OrbitRefine):")
        lines.append("")
        orow = next(r for r in split["table"] if r["method"] == "OrbitRefine")
        lines.append("| gap ° | width | cover | PSNR |")
        lines.append("|---|---|---|---|")
        for k, v in orow["curve"].items():
            def g(t, nd=3):
                m, s, n = t
                if n == 0 or m != m:
                    return "—"
                return f"{m:.2f}±{s:.2f}" if nd == 2 else f"{m:.3f}±{s:.3f}"
            lines.append(f"| {k} | {g(v['width'])} | {g(v['cover'])} | {g(v['psnr'], 2)} |")
        lines.append("")
    path.write_text("\n".join(lines))


def plot_fig2(splits: list, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(splits), 3, figsize=(12, 3.6 * max(len(splits), 1)), squeeze=False)
    xs_lab = [f"{a}-{b}" for a, b in BINS]
    for i, split in enumerate(splits):
        orow = next(r for r in split["table"] if r["method"] == "OrbitRefine")
        xs = np.arange(len(BINS))
        w = [orow["curve"][f"{a}-{b}"]["width"][0] for a, b in BINS]
        c = [orow["curve"][f"{a}-{b}"]["cover"][0] for a, b in BINS]
        p = [orow["curve"][f"{a}-{b}"]["psnr"][0] for a, b in BINS]
        axes[i, 0].plot(xs, w, marker="o")
        axes[i, 1].plot(xs, c, marker="o")
        axes[i, 1].axhline(0.90, color="gray", ls="--", lw=0.8)
        axes[i, 2].plot(xs, p, marker="o")
        axes[i, 0].set_ylabel("width")
        axes[i, 1].set_ylabel("90% cover")
        axes[i, 2].set_ylabel("point PSNR")
        axes[i, 0].set_title(f"{split['orbit']} K={split['k']}")
        for ax in axes[i]:
            ax.set_xticks(xs)
            ax.set_xticklabels(xs_lab, rotation=30, ha="right")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


@torch.no_grad()
def plot_fig3(fiber, cfg, device, orbit: str, k: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = cfg["model"]["proj_hw"]
    loader = make_loader(cfg, orbit, k, 4)
    batch = next(iter(loader))
    pred, scale, qgt, bl, _ = predict_chunks(fiber, batch, hw, device)
    gap = bl["gap_deg"][0]
    # Prefer a far view; fall back to the widest gap in this case.
    far = torch.where(gap >= 40.0)[0]
    j = int(far[0].item()) if far.numel() else int(torch.argmax(gap).item())
    mu = pred[0, j, 0].cpu()
    sig = scale[0, j, 0].cpu()
    gt = qgt[0, j, 0].cpu()
    lp = bl["lerp"][0, j, 0].cpu()
    lo = (mu - 1.64485 * sig).clamp(0, 1)
    hi = (mu + 1.64485 * sig).clamp(0, 1)
    err = (mu - gt).abs()
    sc = sig / sig.max().clamp_min(1e-6)
    imgs = [gt, lp, mu, sc, lo, hi, err, (err > 1.64485 * sig).float()]
    titles = ["GT", "Lerp", "Mean", "Scale (norm)", "Mean−1.64σ", "Mean+1.64σ", "|Mean−GT|", "Outside 90%"]
    obs = batch["obs_views"].to(device)
    obs_rel = batch["obs_rel"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    qrel = batch["query_rel"].to(device)[:, j : j + 1]
    gid = batch["geo_id"].to(device)
    a_ref = batch["a_ref"].to(device)
    one = fiber(obs, obs_rel, qrel, gid, obs_mask=obs_mask, abs_obs=None, a_ref=a_ref, query_views=None)
    if "view_preds" in one:
        vp = one["view_preds"][0, 0]  # [K,1,H,W]
        kshow = min(3, vp.size(0))
        for ki in range(kshow):
            imgs.append(vp[ki, 0].detach().cpu().clamp(0, 1))
            titles.append(f"View branch {ki}")
    n = len(imgs)
    cols = 4
    rows = int(math.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(12, 3.0 * rows))
    ax = np.atleast_2d(ax)
    for i, (im, t) in enumerate(zip(imgs, titles)):
        a = ax[i // cols, i % cols]
        a.imshow(im.numpy() if hasattr(im, "numpy") else im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=8)
        a.axis("off")
    for i in range(n, rows * cols):
        ax[i // cols, i % cols].axis("off")
    fig.suptitle(f"{batch['name'][0]} {orbit} gap={float(gap[j]):.1f}°")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiber_cfg", required=True)
    ap.add_argument("--fiber_ckpt", required=True)
    ap.add_argument("--point_cfg", required=True)
    ap.add_argument("--point_ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n_cases", type=int, default=40)
    ap.add_argument("--n_cal", type=int, default=8)
    ap.add_argument("--orbits", default="sparse,multi_arc")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fiber_cfg = yaml.safe_load(Path(args.fiber_cfg).read_text())
    point_cfg = yaml.safe_load(Path(args.point_cfg).read_text())
    fiber = build_model(
        fiber_cfg, device, use_fiber=True,
        near_mix=bool(fiber_cfg["model"].get("near_mix", False)),
    )
    point = build_model(point_cfg, device, use_fiber=False, near_mix=bool(point_cfg["model"].get("near_mix", True)))
    load_weights(fiber, Path(args.fiber_ckpt))
    load_weights(point, Path(args.point_ckpt))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = []
    for orbit in [x.strip() for x in args.orbits.split(",") if x.strip()]:
        print(f"[split] {orbit} K={args.k}", flush=True)
        splits.append(run_split(fiber, point, fiber_cfg, device, orbit, args.k, args.n_cases, args.n_cal))
        plot_fig3(fiber, fiber_cfg, device, orbit, args.k, outdir / f"fig3_{orbit}.png")
    payload = {"k": args.k, "n_cases": args.n_cases, "n_cal": args.n_cal, "splits": splits}
    (outdir / "table1.json").write_text(json.dumps(payload, indent=2))
    write_markdown(payload, outdir / "TABLE1.md")
    plot_fig2(splits, outdir / "fig2.png")
    print("wrote", outdir, flush=True)


if __name__ == "__main__":
    main()
