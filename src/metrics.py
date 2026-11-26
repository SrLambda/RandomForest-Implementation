"""Incident-level ranking and winner-based classification metrics."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingResult:
    incidents: pd.DataFrame
    confusion: pd.DataFrame

    @property
    def summary(self) -> pd.Series:
        ranks = self.incidents["root_rank"].dropna()
        tn, fp, fn, tp = self.confusion.to_numpy().ravel()
        return pd.Series({
            "attempted": len(self.incidents),
            "associated": int(self.incidents["association_status"].eq("associated").sum()),
            "scored": len(ranks),
            "tied_first": int(self.incidents["top_tie_size"].gt(1).sum()),
            "false_alerts_before_injection": int(self.incidents["false_alerts_before_injection"].sum()),
            "top_1": ranks.le(1).mean() if len(ranks) else np.nan,
            "top_3": ranks.le(3).mean() if len(ranks) else np.nan,
            "mrr": (1 / ranks).mean() if len(ranks) else np.nan,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        })


def evaluate_ranking(
    X: pd.DataFrame, y: pd.Series, audit: pd.DataFrame, score_row: Callable[[pd.Series], float],
) -> RankingResult:
    """Score every candidate or omit the entire incident; report omissions separately."""
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Candidate features and labels must be aligned and unique.")
    if audit.empty or audit.incident_id.duplicated().any():
        raise ValueError("Provide a nonempty audit with unique incident IDs.")
    if set(X.index.get_level_values("incident_id")) != set(
        audit.loc[audit.status.eq("associated"), "incident_id"]
    ):
        raise ValueError("Candidate incidents must match the associated audit incidents.")
    records = []
    confusion = np.zeros((2, 2), dtype=int)
    for case in audit.itertuples(index=False):
        entry = dict(incident_id=case.incident_id, association_status=case.status,
                     false_alerts_before_injection=case.false_alerts_before_injection,
                     scoring_status="not_associated", candidates=None, predicted_service=None,
                     root_rank=None, top_tie_size=None)
        records.append(entry)
        if case.status != "associated":
            continue
        rows = X.xs(case.incident_id, level="incident_id")
        truth = y.xs(case.incident_id, level="incident_id")
        if not truth.isin([0, 1]).all() or truth.sum() != 1:
            raise ValueError("Each associated incident must have exactly one root cause.")
        entry["candidates"] = len(rows)
        if not np.isfinite(rows.to_numpy(dtype=float, na_value=np.nan)).all():
            entry["scoring_status"] = "nonfinite_features"
            continue
        scores = [(service, score_row(row)) for service, row in rows.iterrows()]
        if not all(np.isfinite(score) for _, score in scores):
            entry["scoring_status"] = "nonfinite_scores"
            continue
        # Break ties by service name and count all exact ties at the top.
        ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
        winner, best_score = ranked[0]
        entry.update(scoring_status="scored", predicted_service=winner,
                     root_rank=next(rank for rank, (service, _) in enumerate(ranked, 1)
                                    if truth.at[service] == 1),
                     top_tie_size=sum(score == best_score for _, score in ranked))
        for service, label in truth.items():
            confusion[int(label), int(service == winner)] += 1
    incidents = pd.DataFrame(records).astype(
        {"candidates": "Int64", "root_rank": "Int64", "top_tie_size": "Int64"}
    )
    matrix = pd.DataFrame(confusion, index=pd.Index([0, 1], name="Actual"),
                          columns=pd.Index([0, 1], name="Predicted"))
    return RankingResult(incidents, matrix)
