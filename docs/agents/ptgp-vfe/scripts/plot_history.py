"""Plot a 6-panel VFEDiagnostics history PNG.

CLI: ``python plot_history.py --history-pickle <path> [--out <png>]``

Accepts the three pickle shapes documented in `_common.load_history`. Tier
D output (with non-trivial phase_labels) gets per-phase coloring with
`tab10`; Tier B/C output (single label "run") renders in one color.
"""

import argparse
import sys

from pathlib import Path

import numpy as np

from _common import load_history, phase_sort_key

_FIELDS = (
    "elbo",
    "sigma",
    "nystrom_residual",
    "trace_penalty",
    "fit_per_n",
    "excess_fit_per_n",
)


def _plot_history(history, phase_labels, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_iter = len(history)
    iters = np.arange(n_iter)
    unique_phases = sorted(set(phase_labels), key=phase_sort_key)
    is_staged = unique_phases != ["run"]
    cmap = plt.get_cmap("tab10")
    color_for = {p: cmap(i % 10) for i, p in enumerate(unique_phases)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()

    for ax, field in zip(axes, _FIELDS):
        y = np.asarray([getattr(d, field) for d in history], dtype=float)
        if is_staged:
            for p in unique_phases:
                mask = np.asarray([lab == p for lab in phase_labels], dtype=bool)
                ax.plot(iters[mask], y[mask], "o-", ms=2, color=color_for[p], label=p)
        else:
            ax.plot(iters, y, "o-", ms=2, color=cmap(0))
        ax.set_title(field)
        ax.set_xlabel("iter")
        ax.grid(True, alpha=0.3)

    if is_staged:
        axes[0].legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-pickle", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    history, phase_labels = load_history(args.history_pickle)
    if not history:
        print("Empty history; nothing to plot.", file=sys.stderr)
        return 1

    out = args.out or args.history_pickle.with_suffix(".png")
    _plot_history(history, phase_labels, out)
    print(f"Wrote {out}  ({len(history)} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
