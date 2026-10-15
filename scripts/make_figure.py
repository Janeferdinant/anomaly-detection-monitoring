"""Draw the alert-fatigue figure (test recall against the alert budget) from the benchmark CSVs.

Usage::

    python scripts/make_figure.py        # reads results/benchmark_*/budget.csv.gz, writes docs/img/alert_fatigue.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from anomaly_monitoring import opsreport as R  # noqa: E402
from anomaly_monitoring.detectors import DETECTOR_LABELS  # noqa: E402

SHOW = [("zscore", "#555555", "--"), ("seasonal", "#1f5fbf", "-"), ("ocsvm", "#c2410c", "-"),
        ("ensemble", "#0f7b6c", "-")]
TITLES = {"uci": "UCI household power (injected incidents)", "smd": "Server Machine Dataset (real labels)"}


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds in zip(axes, ("uci", "smd")):
        b = R.prepare(pd.read_csv(ROOT / "results" / f"benchmark_{ds}" / "budget.csv.gz"))
        curve = R.budget_curve(b, k=1, rules="none", methods=[m for m, _, _ in SHOW])
        for m, color, ls in SHOW:
            c = curve[curve["method"] == m]
            ax.plot(c["budget"], c["recall"], ls, color=color, marker="o", ms=3, label=DETECTOR_LABELS[m])
            ax.fill_between(c["budget"], c["recall_lo"], c["recall_hi"], color=color, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_title(TITLES[ds], fontsize=10)
        ax.set_xlabel("alert budget on the calibration window (alerts per day)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test-window event recall")
    axes[0].set_ylim(0, 1)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Alert-fatigue curves: recall gained per extra alert per day (detector alone, k=1, 95% CI over series)",
                 fontsize=10)
    fig.tight_layout()
    out = ROOT / "docs" / "img" / "alert_fatigue.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
