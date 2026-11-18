from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

def load_pickle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    projs = np.asarray(data["projs"])
    if projs.dtype == np.uint8:
        projs = projs.astype(np.float32) / 255.0
    else:
        projs = projs.astype(np.float32)
        mx = float(projs.max()) or 1.0
        if mx > 1.5:
            projs = projs / mx
    angles = np.asarray(data["angles"], dtype=np.float32)
    if projs.ndim == 3:
        projs = projs[:, None]
    return projs, angles

def load_volume(path: Path, res: int = 64) -> np.ndarray:
    if sitk is None or not path.is_file():
        return np.zeros((res, res, res), np.float32)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    t = torch.from_numpy(arr)[None, None]
    t = torch.nn.functional.interpolate(t, size=(res, res, res), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()

def sample_sparse(n: int, k: int, rng: random.Random, npr: np.random.RandomState) -> np.ndarray:
    if rng.random() < 0.3:
        return np.sort(npr.choice(n, size=min(k, n), replace=False))
    idx = np.linspace(0, n, k, endpoint=False).astype(int)
    max_off = max(n - idx[-1] - 1, 0)
    off = rng.randint(0, max_off) if max_off > 0 else 0
    return idx + off

def sample_limited(n: int, angles: np.ndarray, arc_deg: float, rng: random.Random) -> np.ndarray:
    span = float(angles.max() - angles.min()) or np.pi
    arc = np.deg2rad(arc_deg)
    start = rng.uniform(0.0, max(span - arc, 1e-6))
    rel = (angles - angles.min()) % (span + 1e-6)
    idx = np.nonzero((rel >= start) & (rel <= start + arc))[0]
    if len(idx) < 3:
        k = max(int(round(n * arc_deg / 180.0)), 3)
        s = rng.randint(0, max(n - k, 0))
        idx = np.arange(s, s + k)
    return idx

def sample_multi_arc(
    n: int,
    angles: np.ndarray,
    k: int,
    rng: random.Random,
    npr: np.random.RandomState,
    n_arcs: int = 2,
    arc_min_deg: float = 25.0,
    arc_max_deg: float = 45.0,
) -> np.ndarray:

    n_arcs = int(max(2, min(n_arcs, 3)))
    span = float(angles.max() - angles.min()) or np.pi
    rel = (angles - angles.min()) % (span + 1e-6)
    centers = (np.arange(n_arcs) + rng.random()) / n_arcs * span
    per = max(k // n_arcs, 2)
    picked = []
    for c in centers:
        arc = np.deg2rad(rng.uniform(arc_min_deg, arc_max_deg))
        half = 0.5 * arc
        d = np.abs(((rel - c + 0.5 * span) % span) - 0.5 * span)
        cand = np.nonzero(d <= half)[0]
        if len(cand) == 0:
            cand = np.argsort(d)[: max(per, 3)]
        take = min(per, len(cand))
        picked.append(npr.choice(cand, size=take, replace=len(cand) < take))
    idx = np.unique(np.concatenate(picked))
    if len(idx) < max(3, k // 2):
        return sample_sparse(n, k, rng, npr)
    if len(idx) > k:
        idx = np.sort(npr.choice(idx, size=k, replace=False))
    return np.sort(idx)

def pick_queries(rest: np.ndarray, angles: np.ndarray, obs_idx: np.ndarray, num_query: int, npr, far_frac: float) -> np.ndarray:
    if len(rest) < num_query:
        return npr.choice(np.arange(len(angles)), size=num_query, replace=True)
    if far_frac <= 0:
        return npr.choice(rest, size=num_query, replace=False)
    obs_ang = angles[obs_idx]
    d = np.min(
        np.abs(((angles[rest, None] - obs_ang[None, :] + np.pi) % (2 * np.pi)) - np.pi),
        axis=1,
    )
    order = np.argsort(-d)
    n_far = min(num_query, max(1, int(round(num_query * far_frac))))
    far = rest[order[: max(len(rest) // 3, n_far)]]
    n_far = min(n_far, len(far))
    chosen_far = npr.choice(far, size=n_far, replace=len(far) < n_far)
    remain = np.setdiff1d(rest, chosen_far)
    n_rest = num_query - len(np.atleast_1d(chosen_far))
    if n_rest <= 0:
        return np.atleast_1d(chosen_far)[:num_query]
    if len(remain) == 0:
        remain = rest
    chosen_rest = npr.choice(remain, size=n_rest, replace=len(remain) < n_rest)
    return np.concatenate([np.atleast_1d(chosen_far), np.atleast_1d(chosen_rest)])[:num_query]

class CBCTOrbitDataset(Dataset):
    def __init__(
        self,
        roots: list[str],
        split: str = "train",
        num_query: int = 4,
        vol_res: int = 64,
        p_sparse: float = 0.5,
        k_min: int = 3,
        k_max: int = 12,
        arc_min: float = 25.0,
        arc_max: float = 45.0,
        load_gt_vol: bool = True,
        far_frac: float = 0.0,
        use_multi_arc: bool = True,
        n_arcs_min: int = 2,
        n_arcs_max: int = 3,
        force_k: int | None = None,
        force_orbit: str | None = None,
        query_mode: str = "sample",
    ):
        super().__init__()
        self.split = split
        self.num_query = num_query
        self.vol_res = vol_res
        self.p_sparse = p_sparse
        self.k_min, self.k_max = k_min, k_max
        self.arc_min, self.arc_max = arc_min, arc_max
        self.load_gt_vol = load_gt_vol
        self.far_frac = far_frac
        self.use_multi_arc = bool(use_multi_arc)
        self.n_arcs_min = int(n_arcs_min)
        self.n_arcs_max = int(n_arcs_max)
        self.force_k = None if force_k is None else int(force_k)
        self.force_orbit = None if force_orbit is None else str(force_orbit)
        self.query_mode = str(query_mode)
        self.by_domain: dict[str, list[tuple[Path, str, str]]] = {}
        self.items: list[tuple[Path, str, str]] = []
        for root in roots:
            self._index(Path(root))
        if split == "train":
            print("[data] train domains", {k: len(v) for k, v in self.by_domain.items()})
        else:
            print(f"[data] split={split} n={len(self.items)}")

    def _index(self, root: Path) -> None:
        proj = root / "processed" / "projections"
        img = root / "processed" / "images"
        meta_path = root / "meta_info.json"
        names: list[str] = []
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            names = list(meta.get(self.split) or [])
        if not names and proj.is_dir():
            names = [p.stem for p in sorted(proj.glob("*.pickle"))]
        domain = root.name
        geo = "luna" if domain.upper().startswith("LUNA") else "tooth"
        for name in names:
            p = proj / f"{name}.pickle"
            if p.is_file():
                rec = (p, name, geo)
                self.items.append(rec)
                self.by_domain.setdefault(domain, []).append(rec)

    def __len__(self) -> int:
        if self.split == "train" and self.by_domain:
            per = max(len(v) for v in self.by_domain.values())
            return max(per * len(self.by_domain), 1)
        return max(len(self.items), 1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.split == "eval":
            rng = random.Random(10007 + idx)
            npr = np.random.RandomState(10007 + idx)
            path, name, geo = self.items[idx % len(self.items)]
        elif self.split == "train" and self.by_domain:
            rng, npr = random, np.random
            dom = rng.choice(list(self.by_domain.keys()))
            path, name, geo = rng.choice(self.by_domain[dom])
        else:
            rng, npr = random, np.random
            path, name, geo = self.items[idx % len(self.items)]
        projs, angles = load_pickle(path)
        n = len(projs)
        hi = max(self.k_min, min(self.k_max, n - self.num_query - 1))
        k = self.force_k if self.force_k is not None else rng.randint(self.k_min, hi)
        k = max(3, min(int(k), n - 2))
        orbit_roll = self.force_orbit
        if orbit_roll is None:
            orbit_roll = "sparse" if rng.random() < self.p_sparse else ("multi_arc" if self.use_multi_arc else "limited")
        if orbit_roll == "sparse":
            obs_idx = sample_sparse(n, k, rng, npr)
            orbit = "sparse"
        elif orbit_roll == "multi_arc" or (orbit_roll != "limited" and self.use_multi_arc):
            n_arcs = rng.randint(self.n_arcs_min, self.n_arcs_max)
            obs_idx = sample_multi_arc(
                n, angles, k, rng, npr,
                n_arcs=n_arcs,
                arc_min_deg=self.arc_min,
                arc_max_deg=self.arc_max,
            )
            orbit = "multi_arc"
        else:
            arc = rng.uniform(max(self.arc_min, 50.0), max(self.arc_max, 120.0))
            obs_idx = sample_limited(n, angles, arc, rng)
            orbit = "limited"
        cap_k = self.force_k if self.force_k is not None else self.k_max
        if len(obs_idx) > cap_k:
            obs_idx = np.sort(npr.choice(obs_idx, cap_k, replace=False))
        rest = np.setdiff1d(np.arange(n), obs_idx)
        if self.query_mode == "all_unobs":
            q_idx = rest if len(rest) else np.array([0], dtype=np.int64)
        else:
            q_idx = pick_queries(rest, angles, obs_idx, self.num_query, npr, self.far_frac)
        obs_ang = angles[obs_idx]
        a_ref = float(obs_ang[0])
        vol_path = path.parent.parent / "images" / f"{name}.nii.gz"
        gt_vol = load_volume(vol_path, self.vol_res) if self.load_gt_vol else np.zeros((self.vol_res,) * 3, np.float32)
        return {
            "obs_views": torch.from_numpy(projs[obs_idx]),
            "obs_abs": torch.from_numpy(obs_ang.copy()),
            "obs_rel": torch.from_numpy((obs_ang - a_ref).astype(np.float32)),
            "query_views": torch.from_numpy(projs[q_idx]),
            "query_abs": torch.from_numpy(angles[q_idx].copy()),
            "query_rel": torch.from_numpy((angles[q_idx] - a_ref).astype(np.float32)),
            "a_ref": torch.tensor(a_ref, dtype=torch.float32),
            "gt_vol": torch.from_numpy(gt_vol),
            "geo_id": 0 if geo == "luna" else 1,
            "orbit": orbit,
            "name": name,
            "domain": path.parents[2].name if path.parents[2].name else geo,
        }

def collate_cbct(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_v = max(x["obs_views"].shape[0] for x in batch)
    b = len(batch)
    _, c, h, w = batch[0]["obs_views"].shape
    q = batch[0]["query_views"].shape[0]
    obs = torch.zeros(b, max_v, c, h, w)
    obs_abs = torch.zeros(b, max_v)
    obs_rel = torch.zeros(b, max_v)
    mask = torch.zeros(b, max_v)
    for i, x in enumerate(batch):
        v = x["obs_views"].shape[0]
        obs[i, :v] = x["obs_views"]
        obs_abs[i, :v] = x["obs_abs"]
        obs_rel[i, :v] = x["obs_rel"]
        mask[i, :v] = 1
    return {
        "obs_views": obs,
        "obs_abs": obs_abs,
        "obs_rel": obs_rel,
        "obs_mask": mask,
        "query_views": torch.stack([x["query_views"] for x in batch]),
        "query_abs": torch.stack([x["query_abs"] for x in batch]),
        "query_rel": torch.stack([x["query_rel"] for x in batch]),
        "a_ref": torch.stack([x["a_ref"] for x in batch]),
        "gt_vol": torch.stack([x["gt_vol"] for x in batch]),
        "geo_id": torch.tensor([x["geo_id"] for x in batch], dtype=torch.long),
        "orbit": [x["orbit"] for x in batch],
        "name": [x["name"] for x in batch],
        "domain": [x["domain"] for x in batch],
    }
