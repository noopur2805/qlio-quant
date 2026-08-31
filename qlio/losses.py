"""Training objectives for displacement + uncertainty regression."""

import torch


def mse_loss(pred, logvar, target):
    return ((pred - target) ** 2).sum(dim=1).mean()


def gaussian_nll(pred, logvar, target):
    """Diagonal Gaussian MLE loss, TLIO eq. (5) up to a constant."""
    err2 = (pred - target) ** 2
    return (0.5 * (err2 * torch.exp(-logvar) + logvar)).sum(dim=1).mean()


def loss_fn(name):
    return {"mse": mse_loss, "nll": gaussian_nll}[name]
