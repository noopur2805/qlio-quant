"""Adapters turning a torch model into the filter's measurement callable."""

import numpy as np
import torch


class TorchPredictor:
    """Callable used by the EKF: (6, W) window -> (displacement, variance).

    sigma_scale applies post-hoc variance recalibration (per-axis multiplier on
    sigma), which is how a quantized covariance head is repaired without retraining.
    """

    def __init__(self, model, sigma_scale=None, min_sigma=1e-3, device="cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.sigma_scale = None if sigma_scale is None else np.asarray(sigma_scale, dtype=np.float64)
        self.min_sigma = min_sigma
        self.calls = 0

    @torch.no_grad()
    def __call__(self, x):
        self.calls += 1
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.device)
        mean, logvar = self.model(t)
        disp = mean[0].cpu().numpy().astype(np.float64)
        sigma = np.exp(0.5 * logvar[0].cpu().numpy().astype(np.float64))
        if self.sigma_scale is not None:
            sigma = sigma * self.sigma_scale
        sigma = np.maximum(sigma, self.min_sigma)
        return disp, sigma**2


class OracleDisplacement:
    """Ground-truth displacement with a fixed sigma; isolates filter bugs from model error."""

    wants_context = True

    def __init__(self, seq, sigma=0.02, noise=True, seed=0):
        self.seq = seq
        self.sigma = sigma
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, ctx):
        R_g, i0, i = ctx["R_g"], ctx["i0"], ctx["i"]
        d = R_g.T @ (self.seq.p_w[i] - self.seq.p_w[i0])
        if self.noise:
            d = d + self.rng.normal(0, self.sigma, 3)
        return d, np.full(3, self.sigma**2)
