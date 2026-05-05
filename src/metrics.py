from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import hmean
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    labels = np.arange(logits.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "nll": float(log_loss(y_true, probs, labels=labels)),
        "ece": float(ece(y_true, probs)),
        "brier": float(brier_score(y_true, probs)),
    }


def domain_metrics(y_true: np.ndarray, logits: np.ndarray, domains: np.ndarray, id_f1: float | None = None) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    f1s = [f1_score(y_true[domains == d], pred[domains == d], average="macro", zero_division=0) for d in sorted(set(domains))]
    out = {
        "mean_domain_f1": float(np.mean(f1s)),
        "worst_domain_f1": float(np.min(f1s)),
        "harmonic_domain_f1": float(hmean(np.maximum(f1s, 1e-12))),
    }
    if id_f1 is not None and id_f1 > 0:
        out["relative_drop"] = float((id_f1 - out["mean_domain_f1"]) / id_f1)
    return out


def ece(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            score += mask.mean() * abs((pred[mask] == y_true[mask]).mean() - conf[mask].mean())
    return float(score)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def msp_score(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return -temperature * logsumexp(logits / temperature, axis=1)


def ood_metrics(id_oodness: np.ndarray, out_oodness: np.ndarray) -> dict[str, float]:
    y = np.r_[np.zeros_like(id_oodness), np.ones_like(out_oodness)]
    s = np.r_[id_oodness, out_oodness]
    return {
        "auroc": float(roc_auc_score(y, s)),
        "fpr95tpr": float(fpr_at_95_ood_tpr(id_oodness, out_oodness)),
        "aupr_out": float(average_precision_score(y, s)),
    }


def fpr_at_95_ood_tpr(id_oodness: np.ndarray, out_oodness: np.ndarray) -> float:
    threshold = float(np.quantile(out_oodness, 0.05))
    return float(np.mean(id_oodness >= threshold))
