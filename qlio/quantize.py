"""Simulated and real post-training quantization, with per-scope control.

The simulated path (fake quant) is what the study uses: it lets us quantize the
trunk, the displacement head and the covariance head independently, which is the
ablation that isolates what precision does to the *uncertainty* output.
"""

import copy
import time

import torch
import torch.nn as nn

SCOPES = ("stem", "blocks", "mean_head", "cov_head")


def quantize_weight(w, bits=8):
    """Symmetric per-output-channel weight quantization."""
    qmax = 2 ** (bits - 1) - 1
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(w / scale).clamp(-qmax - 1, qmax) * scale


class ActObserver(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))
        self.momentum = momentum

    def observe(self, x):
        lo, hi = x.min().detach(), x.max().detach()
        self.lo = torch.minimum(self.lo, lo) if torch.isfinite(self.lo) else lo
        self.hi = torch.maximum(self.hi, hi) if torch.isfinite(self.hi) else hi

    def fake_quant(self, x, bits=8):
        if not torch.isfinite(self.lo) or self.hi <= self.lo:
            return x
        levels = 2**bits - 1
        scale = (self.hi - self.lo) / levels
        zp = torch.round(-self.lo / scale)
        q = torch.clamp(torch.round(x / scale) + zp, 0, levels)
        return (q - zp) * scale


class FakeQuantWrapper(nn.Module):
    """Wraps a Conv1d/Linear with int-N weight and input-activation simulation."""

    def __init__(self, mod, w_bits=8, a_bits=8):
        super().__init__()
        self.mod = mod
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.obs = ActObserver()
        self.mode = "observe"
        self._wq = None

    def freeze(self):
        self.mode = "quant"
        self._wq = quantize_weight(self.mod.weight.data, self.w_bits)

    def forward(self, x):
        if self.mode == "observe":
            self.obs.observe(x)
            return self.mod(x)
        xq = self.obs.fake_quant(x, self.a_bits)
        if isinstance(self.mod, nn.Conv1d):
            return nn.functional.conv1d(
                xq, self._wq, self.mod.bias, self.mod.stride,
                self.mod.padding, self.mod.dilation, self.mod.groups,
            )
        return nn.functional.linear(xq, self._wq, self.mod.bias)


def _wrap_scope(module, w_bits, a_bits):
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv1d, nn.Linear)):
            setattr(module, name, FakeQuantWrapper(child, w_bits, a_bits))
        else:
            _wrap_scope(child, w_bits, a_bits)


def apply_fake_quant(model, scopes=SCOPES, w_bits=8, a_bits=8):
    """Return a copy of `model` with the named scopes wrapped for fake quant."""
    m = copy.deepcopy(model).eval()
    for s in scopes:
        if not hasattr(m, s):
            raise AttributeError(f"model has no scope '{s}'")
        sub = getattr(m, s)
        if isinstance(sub, (nn.Conv1d, nn.Linear)):
            setattr(m, s, FakeQuantWrapper(sub, w_bits, a_bits))
        else:
            _wrap_scope(sub, w_bits, a_bits)
    return m


def calibrate(model, batches):
    """Run observer passes, then freeze weight scales."""
    model.eval()
    with torch.no_grad():
        for x in batches:
            model(x)
    for m in model.modules():
        if isinstance(m, FakeQuantWrapper):
            m.freeze()
    return model


def quantized_scopes(model):
    return [n for n, m in model.named_modules() if isinstance(m, FakeQuantWrapper)]


def model_size_bytes(model, w_bits=8):
    """Bytes of parameter storage, counting wrapped weights at w_bits."""
    total = 0
    wrapped = set()
    for _, m in model.named_modules():
        if isinstance(m, FakeQuantWrapper):
            total += m.mod.weight.numel() * w_bits / 8
            wrapped.add(id(m.mod.weight))
            if m.mod.bias is not None:
                total += m.mod.bias.numel() * 4
                wrapped.add(id(m.mod.bias))
    for p in model.parameters():
        if id(p) not in wrapped:
            total += p.numel() * 4
    return int(total)


@torch.no_grad()
def benchmark_latency(model, x, warmup=5, iters=50):
    model.eval()
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = torch.tensor(ts)
    return {"p50_ms": float(ts.median()), "p99_ms": float(ts.quantile(0.99)), "mean_ms": float(ts.mean())}


def real_static_ptq(model, batches, backend="fbgemm"):
    """Native int8 static PTQ; returns None if the backend is unavailable."""
    try:
        m = copy.deepcopy(model).eval()
        m.fuse()
        torch.backends.quantized.engine = backend
        m.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(m, inplace=True)
        with torch.no_grad():
            for x in batches:
                m(x)
        return torch.ao.quantization.convert(m, inplace=False)
    except Exception as exc:  # backend or op coverage gaps
        return {"error": repr(exc)}
