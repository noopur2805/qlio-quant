"""TLIO-style 1D ResNet regressing displacement and its log-variance."""

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu2 = nn.ReLU()
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)
            )
        self.add = nn.quantized.FloatFunctional()

    def forward(self, x):
        idt = x if self.skip is None else self.skip(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(self.add.add(out, idt))


class ResNet1D(nn.Module):
    """Trunk shared by both heads; heads are separate modules so precision can differ."""

    def __init__(self, in_ch=6, base=64, layers=(2, 2, 2, 2), out_dim=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks, cin = [], base
        for i, n in enumerate(layers):
            cout = base * (2**i)
            for j in range(n):
                blocks.append(BasicBlock1d(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.mean_head = nn.Linear(cin, out_dim)
        self.cov_head = nn.Sequential(nn.Linear(cin, cin // 4), nn.ReLU(), nn.Linear(cin // 4, out_dim))
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant_mean = torch.ao.quantization.DeQuantStub()
        self.dequant_cov = torch.ao.quantization.DeQuantStub()

    def features(self, x):
        return self.flatten(self.pool(self.blocks(self.stem(x))))

    def forward(self, x):
        x = self.quant(x)
        f = self.features(x)
        mean = self.dequant_mean(self.mean_head(f))
        logvar = self.dequant_cov(self.cov_head(f))
        return mean, torch.clamp(logvar, -10.0, 6.0)

    def fuse(self):
        """Fold conv+bn(+relu) for static quantization."""
        torch.ao.quantization.fuse_modules(
            self.stem, [["0", "1", "2"]], inplace=True
        )
        for b in self.blocks:
            torch.ao.quantization.fuse_modules(b, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if b.skip is not None:
                torch.ao.quantization.fuse_modules(b.skip, [["0", "1"]], inplace=True)
        return self


def small_resnet(**kw):
    """CPU-sized variant used for fast studies."""
    kw.setdefault("base", 16)
    kw.setdefault("layers", (1, 1, 1))
    return ResNet1D(**kw)


def tlio_resnet(**kw):
    kw.setdefault("base", 64)
    kw.setdefault("layers", (2, 2, 2, 2))
    return ResNet1D(**kw)
