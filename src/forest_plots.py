"""Optional Matplotlib figures for development ranking results."""

import numpy as np

from .metrics import RankingResult


def plot_validation(result: RankingResult):
    """Return the confusion and incident rank figures without displaying them."""
    import matplotlib.pyplot as plt

    confusion = result.confusion.to_numpy()
    confusion_figure, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (scored candidates)")
    for (actual, predicted), count in np.ndenumerate(confusion):
        color = "white" if image.norm(count) > 0.5 else "black"
        ax.text(predicted, actual, str(count), ha="center", va="center", color=color)

    ranks = result.incidents["root_rank"].dropna()
    total = len(result.incidents)
    counts = [(f"Rank {rank}", count) for rank, count in ranks.value_counts().sort_index().items()]
    counts.append(("Not scorable", total - len(ranks)))
    labels, values = zip(*counts)
    rank_figure, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = ax.bar(labels, values)
    bars[-1].set_color("tab:gray")
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(1, max(values)) * 1.2)
    ax.set(ylabel="Incidents", title=f"True root rank across {total} validation incidents")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return confusion_figure, rank_figure
