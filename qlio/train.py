"""Two-stage training (MSE warm-up, then Gaussian MLE) and network-level evaluation."""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import gaussian_nll, mse_loss


def train(model, train_ds, val_ds=None, epochs_mse=2, epochs_nll=6, batch_size=64,
          lr=1e-3, weight_decay=1e-5, device="cpu", num_workers=0, log_every=1, seed=0,
          progress=True, progress_every_s=15.0, ckpt_path=None):
    """progress: print live within-epoch throughput/ETA to stdout every
    `progress_every_s` seconds, so long (multi-hour) runs aren't silent.
    ckpt_path: if set, save {model_state, epoch, history} after every epoch,
    so an interrupted multi-hour run doesn't lose all its progress.
    """
    torch.manual_seed(seed)
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    n_batches = len(tl)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs_mse + epochs_nll
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    history = []

    for ep in range(total):
        stage = "mse" if ep < epochs_mse else "nll"
        crit = mse_loss if stage == "mse" else gaussian_nll
        model.train()
        run, nb = 0.0, 0
        t_epoch, t_last = time.time(), time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            mean, logvar = model(x)
            loss = crit(mean, logvar, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
            nb += 1
            if progress and (time.time() - t_last > progress_every_s or nb == n_batches):
                elapsed = time.time() - t_epoch
                rate = nb / max(elapsed, 1e-9)
                eta_epoch = (n_batches - nb) / max(rate, 1e-9)
                print(f"\r  epoch {ep+1}/{total} [{stage}] batch {nb}/{n_batches} "
                     f"({100*nb/n_batches:.0f}%) loss={run/nb:.4f} "
                     f"{rate:.1f} batch/s  ETA(epoch)={eta_epoch/60:.1f} min",
                     end="", flush=True)
                t_last = time.time()
        if progress:
            print()
        sched.step()
        epoch_time = time.time() - t_epoch
        rec = {"epoch": ep, "stage": stage, "train_loss": run / max(nb, 1), "epoch_s": epoch_time}
        if val_ds is not None and (ep % log_every == 0 or ep == total - 1):
            err, sig = evaluate(model, val_ds, device=device, batch_size=batch_size)
            rec["val_rmse_m"] = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
            rec["val_mean_z2"] = float(np.mean((err / np.maximum(sig, 1e-9)) ** 2))
        history.append(rec)
        if progress:
            remaining = (total - ep - 1) * epoch_time
            msg = f"  epoch {ep+1}/{total} done in {epoch_time/60:.1f} min, " \
                 f"train_loss={rec['train_loss']:.4f}"
            if "val_rmse_m" in rec:
                msg += f", val_rmse_m={rec['val_rmse_m']:.4f}"
            msg += f"  (ETA remaining: {remaining/60:.1f} min)"
            print(msg, flush=True)
        if ckpt_path is not None:
            torch.save({"model_state": model.state_dict(), "epoch": ep, "history": history}, ckpt_path)
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
