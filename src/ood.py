from __future__ import annotations

import numpy as np

from .metrics import energy_score, msp_score, ood_metrics


def semantic_ood_report(id_logits: np.ndarray, unknown_logits: np.ndarray) -> dict[str, float]:
    report = {}
    for name, id_score, out_score in (
        ("msp", -msp_score(id_logits), -msp_score(unknown_logits)),
        ("energy", energy_score(id_logits), energy_score(unknown_logits)),
    ):
        for k, v in ood_metrics(id_score, out_score).items():
            report[f"{name}_{k}"] = v
    return report

