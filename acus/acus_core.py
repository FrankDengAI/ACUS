import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Z90 = 1.64485
WIDTH_FACTOR = 3.29
MISS_FACTOR = 20.0
SCALE_FLOOR = math.exp(-6.0)
SCALE_CEIL = 0.35

class ScaleHead(nn.Module):
    def __init__(self, dim=768, out_size=224):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(dim, 128, 1)
        chs = [128, 64, 32]
        layers = []
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

    def forward(self, tokens):
        b, q, n, c = tokens.shape
        g = int(n ** 0.5)
        x = tokens.reshape(b * q, g, g, c).permute(0, 3, 1, 2)
        x = self.up(self.proj(x))
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x.view(b, q, 1, self.out_size, self.out_size).clamp(-6.0, math.log(SCALE_CEIL))

def gaussian_nll(mu, log_scale, target):
    log_var = (2.0 * log_scale).clamp(-8.0, math.log(SCALE_CEIL) * 2)
    inv = torch.exp(-log_var)
    return (0.5 * (log_var + (mu - target).pow(2) * inv)).mean()

def collapse_loss(mu_acquired, scale_acquired, target_acquired):
    return (mu_acquired - target_acquired).abs().mean() + scale_acquired.mean()

def coverage_loss(mu, scale, target, rate=0.90):
    slack = 8.0 * (Z90 * scale - (mu - target).abs())
    soft = torch.sigmoid(slack).mean()
    return torch.relu(mu.new_tensor(rate) - soft)

def interval(mu, scale):
    half = Z90 * scale
    return mu - half, mu + half

def coverage(mu, scale, target):
    lo, hi = interval(mu, scale)
    return ((target >= lo) & (target <= hi)).float().mean()

def exceedance(mu, scale, target):
    return torch.relu((mu - target).abs() - Z90 * scale).mean()

def acus(scale, mu, target):
    return WIDTH_FACTOR * scale.mean() + MISS_FACTOR * exceedance(mu, scale, target)

def fiber_loss(mu, log_scale, scale, target, mu_acquired, scale_acquired, target_acquired, lambda_nll=1.0, lambda_collapse=1.0, lambda_cover=0.0, lambda_point=1.0, lambda_grad=0.05):
    l_point = (mu - target).abs().mean()
    gx = mu[..., 1:, :] - mu[..., :-1, :]
    gy = mu[..., :, 1:] - mu[..., :, :-1]
    tgx = target[..., 1:, :] - target[..., :-1, :]
    tgy = target[..., :, 1:] - target[..., :, :-1]
    l_grad = F.l1_loss(gx, tgx) + F.l1_loss(gy, tgy)
    loss = lambda_point * l_point + lambda_grad * l_grad
    loss = loss + lambda_nll * gaussian_nll(mu, log_scale, target)
    loss = loss + lambda_collapse * collapse_loss(mu_acquired, scale_acquired, target_acquired)
    if lambda_cover > 0:
        loss = loss + lambda_cover * coverage_loss(mu, scale, target)
    return loss
