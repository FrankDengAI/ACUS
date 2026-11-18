from __future__ import annotations

from typing import Optional

from .mae_vit import load_mae_encoder, vit_base_patch16


def build_encoder(name: str, mae_ckpt: Optional[str] = None, img_size: int = 224):
    name = name.lower()
    if name in {"mae", "mae_gray"}:
        if not mae_ckpt:
            raise ValueError("mae_ckpt required")
        return load_mae_encoder(mae_ckpt, img_size=img_size, in_chans=1, strict=False)
    if name == "scratch":
        return vit_base_patch16(img_size=img_size, in_chans=1)
    raise ValueError(f"unknown encoder {name}")
