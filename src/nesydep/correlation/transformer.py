"""AttrFinder: Transformer-guided correlation extraction (SCFDM, paper §6).

The model learns to reconstruct a *masked* attribute from the remaining ones;
attributes whose masking degrades the reconstruction of a target Y carry
dependency information about Y.

Note on fidelity: the shipped prototype's entropy-adaptive threshold code
(``configuration.py``/``adaptive_thresholds.py``) simulated the attention
matrix with ``np.random`` — it was demo scaffolding. This module implements
the mechanism for real: attribute importance is measured by leave-one-out
masking of the trained model's reconstruction accuracy, and the paper's
normalised-Shannon-entropy rule (γ = max(γ₀, 1−avg(H)), individual weight ≥
z/m) selects the minimal antecedent set per target.

Requires the ``transformer`` extra (torch).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nesydep.core.config import TransformerCorrelationConfig
from nesydep.core.registry import CORRELATORS
from nesydep.core.stages import CorrelatedSet, CorrelationGraph


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "TransformerCorrelation requires the 'transformer' extra: "
            "pip install 'nesydep[transformer]' (pulls in torch)"
        ) from e


class AttrFinderModel:
    """Masked-attribute reconstruction Transformer (prototype-faithful).

    Built lazily via :func:`train_attrfinder`; kept torch-free at import.
    """

    def __init__(self, frame: pd.DataFrame, cfg: TransformerCorrelationConfig) -> None:
        torch = _require_torch()
        import torch.nn as nn

        self.torch = torch
        self.columns = [str(c) for c in frame.columns]
        self.cardinalities: dict[str, int] = {}
        codes = np.zeros((len(frame), len(self.columns)), dtype=np.int64)
        for j, col in enumerate(self.columns):
            cat = pd.Categorical(frame[col])
            self.cardinalities[col] = len(cat.categories)
            codes[:, j] = cat.codes
        self.codes = torch.tensor(codes)

        d_model, nhead, num_layers = cfg.d_model, 8, 3
        embeddings = nn.ModuleDict(
            {f"c{j}": nn.Embedding(max(2, self.cardinalities[c]), d_model)
             for j, c in enumerate(self.columns)}
        )
        reconstructors = nn.ModuleDict(
            {f"c{j}": nn.Linear(d_model, max(2, self.cardinalities[c]))
             for j, c in enumerate(self.columns)}
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.embeddings = embeddings
        self.reconstructors = reconstructors
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available()
            else cfg.device if cfg.device != "auto" else "cpu"
        )

    def forward(self, batch, mask_idx: int | None = None):
        torch = self.torch
        embeds = []
        for j in range(batch.shape[1]):
            e = self.embeddings[f"c{j}"](batch[:, j])
            if j == mask_idx:
                e = torch.zeros_like(e)
            embeds.append(e.unsqueeze(1))
        h = self.encoder(torch.cat(embeds, dim=1))
        return [self.reconstructors[f"c{j}"](h[:, j, :]) for j in range(batch.shape[1])]

    def parameters(self):
        for module in (self.embeddings, self.reconstructors, self.encoder):
            yield from module.parameters()

    def to(self, device):
        self.embeddings.to(device)
        self.reconstructors.to(device)
        self.encoder.to(device)
        self.codes = self.codes.to(device)
        return self

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()


def train_attrfinder(frame: pd.DataFrame, cfg: TransformerCorrelationConfig) -> AttrFinderModel:
    """Train with random masked-attribute reconstruction (prototype §6)."""
    torch = _require_torch()
    import torch.nn as nn

    model = AttrFinderModel(frame, cfg)
    model.to(model.device)
    device = model.device
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    n, m = model.codes.shape
    batch_size = min(512, n)

    model.train_mode()
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            batch = model.codes[perm[start:start + batch_size]]
            mask_idx = int(rng.integers(m))
            optimizer.zero_grad()
            logits = model.forward(batch, mask_idx=mask_idx)
            loss = criterion(logits[mask_idx], batch[:, mask_idx])
            loss.backward()
            optimizer.step()
    return model


def reconstruction_accuracy(model: AttrFinderModel, target: int, masked: int | None) -> float:
    """Accuracy of reconstructing ``target`` with attribute ``masked`` zeroed."""
    torch = model.torch
    model.eval_mode()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, model.codes.shape[0], 4096):
            batch = model.codes[start:start + 4096]
            logits = model.forward(batch, mask_idx=masked)
            pred = logits[target].argmax(dim=1)
            correct += int((pred == batch[:, target]).sum())
            total += batch.shape[0]
    return correct / max(1, total)


def adaptive_thresholds(importance: np.ndarray, gamma0: float, z: int) -> tuple[np.ndarray, float]:
    """Paper §7.3: entropy-adaptive retention threshold.

    ``importance[j, i]`` = accuracy drop on target j when attribute i is
    masked. Row-normalise, compute the normalised Shannon entropy per target,
    set γ = max(γ₀, 1 − avg(H)), and return the boolean keep-matrix with
    cumulative weight ≥ γ and individual weight ≥ z/m.
    """
    m = importance.shape[0]
    normed = importance / np.maximum(importance.sum(axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(normed > 0, np.log(normed), 0.0)
    entropy = -(normed * logp).sum(axis=1)
    max_entropy = np.log(max(2, m - 1))
    avg_h = float(entropy.mean() / max_entropy)
    gamma = max(gamma0, 1.0 - avg_h)

    keep = np.zeros_like(importance, dtype=bool)
    for j in range(m):
        order = np.argsort(-importance[j])
        cumulative = 0.0
        total = importance[j].sum()
        for i in order:
            if i == j or importance[j, i] < (z / m) * max(total, 1e-12):
                continue
            keep[j, i] = True
            cumulative += importance[j, i]
            if total > 0 and cumulative / total >= gamma:
                break
    return keep, gamma


@CORRELATORS.decorator("transformer")
class TransformerCorrelation:
    """AttrFinder correlation extractor (requires the ``transformer`` extra)."""

    def extract(
        self, frame: pd.DataFrame, config: TransformerCorrelationConfig | None = None
    ) -> CorrelationGraph:
        cfg = config or TransformerCorrelationConfig()
        model = train_attrfinder(frame, cfg)
        m = len(model.columns)

        # Leave-one-out masking importance matrix.
        importance = np.zeros((m, m))
        base_acc = [reconstruction_accuracy(model, j, None) for j in range(m)]
        for j in range(m):
            for i in range(m):
                if i != j:
                    importance[j, i] = max(0.0, base_acc[j] - reconstruction_accuracy(model, j, i))

        if cfg.threshold == "auto":
            keep, _gamma = adaptive_thresholds(importance, cfg.gamma0, cfg.z)
        else:
            keep = importance >= float(cfg.threshold)
            np.fill_diagonal(keep, False)

        sets = [
            CorrelatedSet(
                antecedents=tuple(model.columns[i] for i in range(m) if keep[j, i]),
                target=model.columns[j],
            )
            for j in range(m)
            if keep[j].any()
        ]
        return CorrelationGraph(sets)
