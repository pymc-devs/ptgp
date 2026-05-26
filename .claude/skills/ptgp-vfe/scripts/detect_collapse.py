"""Run pitfall detection rules against a saved VFE training history.

CLI: ``python detect_collapse.py --history-pickle <path>``

Detection rules operate on the ``history`` list of VFEDiagnostics and
never on phase_labels, so the same rules work unchanged on Tier B/C
(`tracked_minimize`) and Tier D (`minimize_staged_vfe`) output.
"""

import sys

import numpy as np

from _common import Verdict, cli_main


def _arr(history, field):
    return np.asarray([getattr(d, field) for d in history], dtype=float)


def _verdict(name, status, evidence, next_steps):
    return Verdict(name, status, evidence, next_steps)


def detect(history, phase_labels):
    sigma = _arr(history, "sigma")
    elbo = _arr(history, "elbo")
    nystrom = _arr(history, "nystrom_residual")
    excess = _arr(history, "excess_fit_per_n")
    _arr(history, "trace_penalty")
    fit = _arr(history, "fit")
    n = len(history)
    half = max(1, n // 2)

    verdicts = []

    # sigma_collapse: sigma drops a lot and ELBO rose while nystrom didn't fall
    if (
        sigma[-1] < 1e-3
        and sigma[-1] < 0.1 * sigma[0]
        and elbo[-1] > elbo[0]
        and nystrom[-1] > 0.5 * nystrom[0]
    ):
        verdicts.append(
            _verdict(
                "sigma_collapse",
                "CONFIRMED",
                f"sigma {sigma[0]:.3g} -> {sigma[-1]:.3g}; nystrom_residual flat",
                "see pitfalls/sigma_collapse.md; escalate to Tier D",
            )
        )
    elif sigma[-1] < 0.1 * sigma[0] and sigma[-1] < 0.01:
        verdicts.append(
            _verdict(
                "sigma_collapse",
                "SUSPECT",
                f"sigma {sigma[0]:.3g} -> {sigma[-1]:.3g} (10x reduction)",
                "check predictive fit; see pitfalls/sigma_collapse.md",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "sigma_collapse",
                "OK",
                f"sigma {sigma[0]:.3g} -> {sigma[-1]:.3g}",
                "",
            )
        )

    # sigma_inflation: sigma grew, ELBO plateaued, excess_fit dropped
    elbo_late_progress = elbo[-1] - elbo[half]
    if sigma[-1] > 5.0 * sigma[0] and abs(elbo_late_progress) < 1e-2 * abs(elbo[0]):
        verdicts.append(
            _verdict(
                "sigma_inflation",
                "CONFIRMED",
                f"sigma grew {sigma[0]:.3g} -> {sigma[-1]:.3g}; ELBO plateau",
                "see pitfalls/sigma_inflation.md",
            )
        )
    elif sigma[-1] > 2.0 * sigma[0]:
        verdicts.append(
            _verdict(
                "sigma_inflation",
                "SUSPECT",
                f"sigma grew {sigma[0]:.3g} -> {sigma[-1]:.3g}",
                "check excess_fit_per_n; see pitfalls/sigma_inflation.md",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "sigma_inflation",
                "OK",
                f"sigma stable around {sigma[-1]:.3g}",
                "",
            )
        )

    # excess_fit_per_n_negative
    if excess[-1] <= 0:
        verdicts.append(
            _verdict(
                "excess_fit_per_n_negative",
                "CONFIRMED",
                f"excess_fit_per_n[-1] = {excess[-1]:.3g} <= 0 (model fits no better than noise)",
                "see pitfalls/excess_fit_per_n_negative.md",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "excess_fit_per_n_negative",
                "OK",
                f"excess_fit_per_n[-1] = {excess[-1]:.3g} > 0",
                "",
            )
        )

    # nystrom_residual_stuck (pitfall: M_too_small or inducing_layout_poor)
    nys_change = (nystrom[-1] - nystrom[0]) / max(abs(nystrom[0]), 1e-12)
    if nystrom[-1] > 0.05 and nys_change > -0.5:
        verdicts.append(
            _verdict(
                "M_too_small",
                "SUSPECT",
                f"nystrom_residual {nystrom[0]:.3g} -> {nystrom[-1]:.3g} (still large)",
                "compare trace_curve against threshold; see pitfalls/M_too_small.md "
                "or pitfalls/inducing_layout_poor.md",
            )
        )
        verdicts.append(
            _verdict(
                "inducing_layout_poor",
                "SUSPECT",
                f"nystrom_residual {nystrom[0]:.3g} -> {nystrom[-1]:.3g} (still large)",
                "if trace_curve has flattened, re-init Z with trained kernel",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "M_too_small",
                "OK",
                f"nystrom_residual {nystrom[-1]:.3g}",
                "",
            )
        )
        verdicts.append(
            _verdict(
                "inducing_layout_poor",
                "OK",
                f"nystrom_residual {nystrom[-1]:.3g}",
                "",
            )
        )

    # nystrom rising during training (lengthscale_runaway downstream signal)
    if nystrom[-1] > 2.0 * nystrom[half]:
        verdicts.append(
            _verdict(
                "lengthscale_runaway",
                "SUSPECT",
                f"nystrom_residual rising in second half: {nystrom[half]:.3g} -> {nystrom[-1]:.3g}",
                "check trained lengthscales; see pitfalls/lengthscale_runaway.md",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "lengthscale_runaway",
                "OK",
                "nystrom_residual not rising",
                "",
            )
        )

    # non_finite_at_init
    if not np.all(np.isfinite([elbo[0], fit[0], sigma[0], excess[0]])):
        verdicts.append(
            _verdict(
                "non_finite_at_init",
                "CONFIRMED",
                "non-finite values in history[0]",
                "run pg.utils.check_init; see pitfalls/non_finite_at_init.md",
            )
        )
    else:
        verdicts.append(
            _verdict(
                "non_finite_at_init",
                "OK",
                "history[0] all finite",
                "",
            )
        )

    # slow_convergence: ELBO still moving in last quartile
    last_q = max(1, n // 4)
    if n >= 8:
        late_progress = elbo[-1] - elbo[-last_q]
        if abs(late_progress) > 1e-3 * abs(elbo[0]):
            verdicts.append(
                _verdict(
                    "slow_convergence",
                    "SUSPECT",
                    f"ELBO still changing in last quartile: {late_progress:+.3g}",
                    "increase maxiter; see pitfalls/slow_convergence.md",
                )
            )
        else:
            verdicts.append(
                _verdict(
                    "slow_convergence",
                    "OK",
                    "ELBO converged",
                    "",
                )
            )
    else:
        verdicts.append(
            _verdict(
                "slow_convergence",
                "OK",
                "history too short to assess",
                "",
            )
        )

    # The remaining pitfalls require info beyond `history`:
    #   kuu_ill_conditioned  -> needs GreedyVarianceDiagnostics (use check_inducing.py)
    #   inducing_collapse    -> needs Z (post-training)
    #   categorical_inducing_dim -> needs X dtype info
    #   large_grad_at_init   -> needs check_init output, not history
    #   eta_collapse         -> needs trained_params
    #   bad_priors           -> needs prior-predictive run
    #   lbfgsb_abnormal      -> needs scipy result.status
    # Emit informational OK verdicts so the user knows they were considered.
    for unknowable in (
        "kuu_ill_conditioned",
        "inducing_collapse",
        "categorical_inducing_dim",
        "large_grad_at_init",
        "eta_collapse",
        "bad_priors",
        "lbfgsb_abnormal",
    ):
        verdicts.append(
            _verdict(
                unknowable,
                "OK",
                "not detectable from history alone — see pitfalls/<slug>.md",
                "",
            )
        )

    return verdicts


if __name__ == "__main__":
    sys.exit(cli_main(detect))
