"""Shared helpers for ptgp-vfe skill scripts."""

import argparse
import pickle
import sys

from collections import namedtuple
from pathlib import Path

from ptgp.optim import phase_sort_key  # noqa: F401  re-exported; used by plot_history.py

Verdict = namedtuple("Verdict", ["pitfall", "status", "evidence", "next_steps"])
# status ∈ {"OK", "SUSPECT", "CONFIRMED"}


def load_history(path):
    """Load a training pickle and normalise to (history, phase_labels).

    Accepts three shapes from VFE training:
    - bare list of VFEDiagnostics  (Tier B/C `tracked_minimize`)
    - tuple (history, phase_labels)              (Tier D)
    - tuple (history, phase_labels, *_extras)    (Tier D full return)
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        history = obj
        phase_labels = ["run"] * len(history)
    elif isinstance(obj, tuple) and len(obj) >= 2:
        history = list(obj[0])
        phase_labels = list(obj[1])
        if len(history) != len(phase_labels):
            raise ValueError(
                f"history ({len(history)}) and phase_labels ({len(phase_labels)}) "
                f"length mismatch in {path}"
            )
    else:
        raise TypeError(f"Unrecognised pickle shape at {path}: {type(obj).__name__}")

    return history, phase_labels


def cli_main(detect_fn):
    """Thin CLI wrapper for a `detect_fn(history, phase_labels) -> [Verdict]`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-pickle", required=True, type=Path)
    args = parser.parse_args()
    history, phase_labels = load_history(args.history_pickle)
    verdicts = detect_fn(history, phase_labels)
    print_verdicts(verdicts)
    return 0


def print_verdicts(verdicts):
    """Print a sorted Verdict table to stdout."""
    rank = {"CONFIRMED": 0, "SUSPECT": 1, "OK": 2}
    sorted_v = sorted(verdicts, key=lambda v: (rank.get(v.status, 9), v.pitfall))
    width = max(len(v.pitfall) for v in sorted_v) if sorted_v else 0
    for v in sorted_v:
        marker = {"CONFIRMED": "!!", "SUSPECT": "? ", "OK": "  "}[v.status]
        print(f"{marker} {v.pitfall:<{width}}  {v.status:<9}  {v.evidence}")
        if v.status != "OK" and v.next_steps:
            print(f"      -> {v.next_steps}")


if __name__ == "__main__":
    sys.exit(0)
