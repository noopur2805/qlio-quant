"""Two-stage training (MSE warm-up, then Gaussian MLE) and network-level evaluation."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import gaussian_nll, mse_loss


def train(model, train_ds, val_ds=None, epochs_mse=2, epochs_nll=6, batch_size=64,
          lr=1e-3, weight_decay=1e-5, device="cpu", num_workers=0, log_every=1, seed=0):
    torch.manual_seed(seed)
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs_mse + epochs_nll
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    history = []

    for ep in range(total):
        stage = "mse" if ep < epochs_mse else "nll"
        crit = mse_loss if stage == "mse" else gaussian_nll
        model.train()
        run, nb = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            mean, logvar = model(x)
            loss = crit(mean, logvar, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.detach().item()
            nb += 1
        sched.step()
        rec = {"epoch": ep, "stage": stage, "train_loss": run / max(nb, 1)}
        if val_ds is not None and (ep % log_every == 0 or ep == total - 1):
            err, sig = evaluate(model, val_ds, device=device, batch_size=batch_size)
            rec["val_rmse_m"] = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
            rec["val_mean_z2"] = float(np.mean((err / np.maximum(sig, 1e-9)) ** 2))
        history.append(rec)
    return history


@torch.no_grad()
def evaluate(model, ds, device="cpu", batch_size=128):
    """Return (errors, sigmas) over the dataset in prediction order."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errs, sigs = [], []
    for x, y in dl:
        mean, logvar = model(x.to(device))
        errs.append((mean.cpu() - y).numpy())
        sigs.append(torch.exp(0.5 * logvar).cpu().numpy())
    return np.concatenate(errs), np.concatenate(sigs)
