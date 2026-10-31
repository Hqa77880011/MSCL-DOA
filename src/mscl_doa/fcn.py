"""Dai et al. proxy-spectrum FCN: complex proxy channels, skip convolutions and BCE."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import find_peaks
from .data import load_case
from .common import load, table


def proxy_spectrum(blocks, masks, positions, grid):
    """B^H r from jointly observed off-diagonal covariance entries (Eq. 10)."""
    proxy = np.zeros(len(grid), complex)
    for block, mask in zip(blocks, masks):
        ids = np.flatnonzero(mask)
        covariance = block[ids] @ block[ids].conj().T / block.shape[1]
        i, j = np.triu_indices(len(ids), 1)
        lag = np.asarray(positions)[ids[i]] - np.asarray(positions)[ids[j]]
        dictionary = np.exp(2j * np.pi * lag[:, None] * np.sin(np.deg2rad(grid))[None])
        proxy += dictionary.conj().T @ covariance[i, j]
    return np.stack([proxy.real, proxy.imag, np.angle(proxy)]).astype(np.float32)


def convolution(inputs, outputs):
    return nn.Sequential(nn.Conv1d(inputs, outputs, 31, padding=15, padding_mode="reflect"),
                         nn.BatchNorm1d(outputs), nn.ReLU())


class SpectrumFCN(nn.Module):
    """Length-preserving convolutions with shallow/deep concatenation skips."""
    def __init__(self, width=64):
        super().__init__()
        self.first = convolution(3, width)
        self.second = convolution(width, 2 * width)
        self.third = convolution(2 * width, 2 * width)
        self.deep = convolution(4 * width, 2 * width)
        self.shallow = convolution(3 * width, width)
        self.output = nn.Sequential(nn.Conv1d(width, 1, 1), nn.BatchNorm1d(1))

    def forward(self, x):
        first = self.first(x); second = self.second(first); third = self.third(second)
        deep = self.deep(torch.cat([second, third], 1))
        return self.output(self.shallow(torch.cat([first, deep], 1)))[:, 0]


def train(data, output, c):
    """Use explicitly labelled simulation training horizons, with test held out."""
    torch.manual_seed(c["seed"])
    grid = np.arange(-90., 90. + c.get("fcn_grid_step", 1.) / 2, c.get("fcn_grid_step", 1.))
    records = [r for r in load(Path(data) / "manifest.json") if r["split"] == "train"]
    if not records:
        raise ValueError("FCN needs independent labelled train horizons; do not train on validation/test")
    inputs, targets = [], []
    for record in records:
        case = load_case(record["path"])
        inputs.append(proxy_spectrum(case["x"], case["masks"], case["positions"], grid))
        labels = np.zeros(len(grid), np.float32)
        labels[np.argmin(abs(grid[:, None] - case["angles"][None]), axis=0)] = 1
        targets.append(labels)
    inputs, targets = torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))
    model = SpectrumFCN(c.get("fcn_width", 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=c.get("fcn_lr", 1e-4))
    epochs, batch = c.get("fcn_epochs", 200), c.get("fcn_batch_size", 32)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=c.get("fcn_max_lr", .01),
                                                    total_steps=max(4, epochs * int(np.ceil(len(inputs) / batch))),
                                                    pct_start=.3)
    history = []
    for epoch in range(epochs):
        model.train(); total = 0.
        for selected in torch.randperm(len(inputs)).split(batch):
            loss = nn.functional.binary_cross_entropy_with_logits(model(inputs[selected]), targets[selected])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            total += float(loss.detach()) * len(selected)
        history.append(dict(epoch=epoch + 1, loss=total / len(inputs)))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), width=c.get("fcn_width", 64), grid=torch.tensor(grid), config=c), output / "fcn.pt")
    table(output / "history.csv", history)


def restore(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SpectrumFCN(state["width"]); model.load_state_dict(state["model"]); model.eval()
    return model, state["grid"].numpy()


def infer(case, c, predictor):
    model, grid = predictor
    x = proxy_spectrum(case["x"], case["masks"], case["positions"], grid)
    with torch.no_grad():
        spectrum = model(torch.from_numpy(x)[None]).sigmoid()[0].numpy()
    peaks = find_peaks(spectrum)[0]
    if len(peaks) < c["sources"]:
        peaks = np.arange(len(grid))
    chosen = peaks[np.argsort(spectrum[peaks])[-c["sources"]:]]
    lags = np.round(case["positions"][:, None] - case["positions"][None], 8)
    observed = np.any(case["masks"][:, :, None] & case["masks"][:, None], axis=0)
    return dict(angles=np.sort(grid[chosen]), quality=float(spectrum[chosen].min()),
                information_rank=c["sources"], lag_coverage=len(np.unique(lags[observed])) / len(np.unique(lags)),
                spectrum=spectrum, grid=grid)
