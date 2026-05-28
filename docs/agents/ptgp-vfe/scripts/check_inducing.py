"""Pre-fit health check on a GreedyVarianceDiagnostics pickle.

Importable: ``check_inducing(kernel, X, Z, jitter=1e-6) -> dict``
CLI:        ``python check_inducing.py --diag-pickle <path> [--out <png>]``

CLI mode reads a saved ``GreedyVarianceDiagnostics``, prints repr +
per-Kuu-field verdicts, and writes the canonical 3-panel inducing PNG
(trace_curve, fraction-unexplained with 1% threshold, d_final scatter).
"""

import argparse
import pickle
import sys

from pathlib import Path

import numpy as np

from _common import Verdict, print_verdicts

_KUU_COND_BAD = 1e8
_KUU_COND_WARN = 1e5


def check_inducing(kernel, X, Z, jitter: float = 1e-6, eig_threshold: float = 1e-4):
    """Inducing-point health check at a given (kernel, X, Z).

    Thin wrapper around :func:`ptgp.inducing.compute_inducing_diagnostics`.
    Returns a :class:`ptgp.inducing.KernelHealthDiagnostics` with the Kuu
    eigenvalue stats, per-point ``d_final``, ``total_variance``, and
    ``nystrom_residual``. (No ``trace_curve`` — that's only available
    from the progressive selection in :func:`greedy_variance_init`.)
    """
    from ptgp.inducing import compute_inducing_diagnostics

    return compute_inducing_diagnostics(
        kernel,
        X,
        Z,
        jitter=jitter,
        eig_threshold=eig_threshold,
    )


def _verdicts_from_diag(diag) -> list:
    cond = diag.kuu_condition_number
    if cond > _KUU_COND_BAD:
        v_cond = Verdict(
            "kuu_ill_conditioned",
            "CONFIRMED",
            f"condition number {cond:.2g} > {_KUU_COND_BAD:.0g}",
            "audit priors; check for duplicate Z; see pitfalls/kuu_ill_conditioned.md",
        )
    elif cond > _KUU_COND_WARN:
        v_cond = Verdict(
            "kuu_ill_conditioned",
            "SUSPECT",
            f"condition number {cond:.2g} > {_KUU_COND_WARN:.0g}",
            "watch for non-finite values during training",
        )
    else:
        v_cond = Verdict("kuu_ill_conditioned", "OK", f"condition number {cond:.2g}", "")

    if diag.kuu_n_small_eigenvalues > 0:
        v_small = Verdict(
            "inducing_collapse",
            "SUSPECT",
            f"{diag.kuu_n_small_eigenvalues} eigenvalue(s) below {diag.kuu_eig_threshold:.0e}",
            "check pairwise Z distances; see pitfalls/inducing_collapse.md",
        )
    else:
        v_small = Verdict(
            "inducing_collapse", "OK", f"no eigenvalues below {diag.kuu_eig_threshold:.0e}", ""
        )

    pct = 100.0 * (1.0 - diag.trace_curve[-1] / diag.total_variance)
    if pct < 95.0:
        v_M = Verdict(
            "M_too_small",
            "SUSPECT",
            f"variance explained {pct:.1f}% < 95% at chosen M",
            "increase M; see reference/choosing_M.md",
        )
    elif pct < 99.0:
        v_M = Verdict(
            "M_too_small",
            "OK",
            f"variance explained {pct:.1f}% (loose bound)",
            "",
        )
    else:
        v_M = Verdict("M_too_small", "OK", f"variance explained {pct:.1f}%", "")

    return [v_cond, v_small, v_M]


def _plot_3panel(diag, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    M = len(diag.trace_curve)
    m_axis = np.arange(1, M + 1)

    axes[0].plot(m_axis, diag.trace_curve)
    axes[0].set_xlabel("M")
    axes[0].set_ylabel("trace_curve (residual variance)")
    axes[0].set_title("Residual variance vs M")
    axes[0].set_yscale("log")

    frac = diag.trace_curve / diag.total_variance
    axes[1].plot(m_axis, frac)
    axes[1].axhline(0.01, color="r", linestyle="--", label="1% threshold")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("fraction unexplained")
    axes[1].set_title("Fraction unexplained vs M (knee selection)")
    axes[1].set_yscale("log")
    axes[1].legend()

    d = diag.d_final
    axes[2].scatter(np.arange(len(d)), d, s=4, alpha=0.5)
    axes[2].set_xlabel("data index")
    axes[2].set_ylabel("d_final")
    axes[2].set_title("Per-point residual conditional variance")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-pickle", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: alongside the pickle, .png suffix)",
    )
    args = parser.parse_args(argv)

    with open(args.diag_pickle, "rb") as f:
        diag = pickle.load(f)

    print(repr(diag))
    print()
    verdicts = _verdicts_from_diag(diag)
    print_verdicts(verdicts)

    out = args.out or args.diag_pickle.with_suffix(".png")
    _plot_3panel(diag, out)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
