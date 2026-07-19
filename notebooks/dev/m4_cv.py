"""M4 7-fold CV: step-4 architecture, twin composition, NGD on Gaussian blocks,
variational q on the R2D2 budget and c. 2500 steps/fold, caches watermains_cv_m4_{t}.pkl.
Ends with the pooled five-model table (m1/m2/hgb/glm from the seq97 caches)."""
# ruff: noqa: E402

import os
import pickle
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import watermains as wm

from scipy.linalg import cho_factor, cho_solve

import ptgp as pg

FOLDS = [2025, 2019, 2020, 2021, 2022, 2023, 2024]
N_STEPS = 2500
M_BASE, M_HIST = 1024, 256
S_DRAWS = 16
BATCH, EVAL_SIZE, EVAL_EVERY = 1024, 32768, 50
K_COMP = 5
JITTER = 1e-6
YEAR0 = 1997
ADAM_SWITCH = 700

breaks, mains, _ = wm.load_kitchener_data(cache_path="watermains_cache.pkl")
folds = {t: wm.build_panel(mains, breaks, float(t)) for t in FOLDS}
BASE_EPS = np.random.default_rng(42).standard_normal((S_DRAWS, 6))
lik = pg.likelihoods.Poisson()


def gamma_sched(i, cap):
    g = min(cap, 0.001 * (1.6 ** (i // 25)))
    if i > 400:
        g = min(g, cap / (1.0 + (i - 400) / 300.0))
    return g


def ngd_update_full(m_sh, S_sh, g_m, g_S, gamma):
    m, S = m_sh.get_value(), S_sh.get_value()
    g_S = 0.5 * (g_S + g_S.T)
    c, low = cho_factor(S)
    Sinv_m = cho_solve((c, low), m)
    Sinv = cho_solve((c, low), np.eye(len(m)))
    for _ in range(6):
        Sinv_new = Sinv - 2.0 * gamma * g_S
        try:
            cn = np.linalg.cholesky(Sinv_new)
        except np.linalg.LinAlgError:
            gamma *= 0.5
            continue
        S_new = cho_solve((cn, True), np.eye(len(m)))
        S_new = 0.5 * (S_new + S_new.T)
        m_new = S_new @ (Sinv_m + gamma * (g_m - 2.0 * g_S @ m))
        m_sh.set_value(m_new)
        S_sh.set_value(S_new)
        return


def ngd_update_diag(m_sh, s_sh, g_m, g_s, gamma):
    m, s = m_sh.get_value(), s_sh.get_value()
    prec_new = 1.0 / s - 2.0 * gamma * g_s
    ok = prec_new > 1e-8
    s_new = np.where(ok, 1.0 / np.clip(prec_new, 1e-8, None), s)
    m_new = np.where(ok, s_new * (m / s + gamma * (g_m - 2.0 * g_s * m)), m)
    m_sh.set_value(m_new)
    s_sh.set_value(s_new)


def invgamma_lp(log_x, a, b):
    return pt.sum(-(a + 1.0) * log_x - b * pt.exp(-log_x) + log_x)


def fit_m4(train, test, n_steps=N_STEPS, seed=0):
    Xtr, stats = wm.panel_design(train)
    Xte, _ = wm.panel_design(test, stats)
    ytr = train["y"].to_numpy(float)
    N = len(ytr)
    n_mat = int(Xtr[:, wm.MAT_COL].max()) + 1
    n_zone = int(Xtr[:, wm.ZONE_COL].max()) + 1
    n_pipes = int(max(Xtr[:, wm.ID_COL].max(), Xte[:, wm.ID_COL].max())) + 1
    n_years = int(test["year"].max()) - YEAR0 + 1
    input_dim = Xtr.shape[1]
    sigma2 = 1.0 / ytr.mean()
    log_rate0 = float(np.log(ytr.sum() / np.exp(Xtr[:, wm.EXPO_COL]).sum()))
    freq_mat = np.bincount(Xtr[:, wm.MAT_COL].astype(int), minlength=n_mat) / N
    freq_zone = np.bincount(Xtr[:, wm.ZONE_COL].astype(int), minlength=n_zone) / N
    mu_y, sd_y = float(stats[0]["year"]), float(stats[1]["year"])

    Z0_base, _, _ = wm.init_inducing(Xtr, M=M_BASE, cont_dims=list(wm.CONT_DIMS), rng=seed)
    Z0_hist, _, _ = wm.init_inducing(
        Xtr, M=M_HIST, cont_dims=[wm.HIST_COL, 0], code_cols=(wm.MAT_COL,), rng=seed
    )
    years_u = np.unique(Xtr[:, wm.YEAR_COL])
    Z_yr = np.zeros((len(years_u), input_dim))
    Z_yr[:, wm.YEAR_COL] = years_u
    M_YR = len(years_u)

    wrng = np.random.default_rng(seed)
    sh = pytensor.shared
    p_log_ls_base = sh(np.zeros(4), name="log_ls_base")
    p_log_ls_trend = sh(np.array(np.log(2.5)), name="log_ls_trend")
    p_log_ls_hist = sh(np.zeros(2), name="log_ls_hist")
    p_W_mat = sh(0.1 * wrng.standard_normal((n_mat, 2)), name="W_mat")
    p_log_kap_mat = sh(np.full(n_mat, -0.5), name="log_kap_mat")
    p_W_zone = sh(0.1 * wrng.standard_normal((n_zone, 2)), name="W_zone")
    p_log_kap_zone = sh(np.full(n_zone, -0.5), name="log_kap_zone")
    p_W_mat2 = sh(0.1 * wrng.standard_normal((n_mat, 2)), name="W_mat2")
    p_log_kap_mat2 = sh(np.full(n_mat, -0.5), name="log_kap_mat2")
    p_Z_base = sh(Z0_base.copy(), name="Z_base")
    p_Z_hist = sh(Z0_hist.copy(), name="Z_hist")

    W0 = sigma2 * 0.01 / 0.99
    p_q_loc = sh(np.concatenate([np.log(np.full(5, W0 / 5)), [log_rate0]]), name="q_loc")
    p_q_logsd = sh(np.full(6, np.log(0.1)), name="q_logsd")

    q_m_b, q_S_b = sh(np.zeros(M_BASE)), sh(np.eye(M_BASE))
    q_m_t, q_S_t = sh(np.zeros(M_YR)), sh(np.eye(M_YR))
    q_m_h, q_S_h = sh(np.zeros(M_HIST)), sh(np.eye(M_HIST))
    m_f, s_f = sh(np.zeros(n_pipes)), sh(np.full(n_pipes, 0.09))
    m_y, s_y = sh(np.zeros(n_years)), sh(np.full(n_years, 0.09))

    adam_params = [
        p_log_ls_base,
        p_log_ls_trend,
        p_log_ls_hist,
        p_W_mat,
        p_log_kap_mat,
        p_W_zone,
        p_log_kap_zone,
        p_W_mat2,
        p_log_kap_mat2,
        p_Z_base,
        p_Z_hist,
        p_q_loc,
        p_q_logsd,
    ]

    Xb = pt.matrix("Xb")
    yb = pt.vector("yb")

    k_base = (
        pg.kernels.Matern52(
            input_dim=input_dim, ls=pt.exp(p_log_ls_base), active_dims=list(wm.CONT_DIMS)
        )
        * wm.NormalizedLowRankCategorical(
            input_dim=input_dim,
            num_levels=n_mat,
            W=p_W_mat,
            kappa=pt.exp(p_log_kap_mat),
            freqs=freq_mat,
            active_dims=[wm.MAT_COL],
        )
        * wm.NormalizedLowRankCategorical(
            input_dim=input_dim,
            num_levels=n_zone,
            W=p_W_zone,
            kappa=pt.exp(p_log_kap_zone),
            freqs=freq_zone,
            active_dims=[wm.ZONE_COL],
        )
    )
    k_trend = pg.kernels.ExpQuad(
        input_dim=input_dim, ls=pt.exp(p_log_ls_trend), active_dims=[wm.YEAR_COL]
    )
    k_hist = pg.kernels.Matern52(
        input_dim=input_dim, ls=pt.exp(p_log_ls_hist), active_dims=[wm.HIST_COL, 0]
    ) * wm.NormalizedLowRankCategorical(
        input_dim=input_dim,
        num_levels=n_mat,
        W=p_W_mat2,
        kappa=pt.exp(p_log_kap_mat2),
        freqs=freq_mat,
        active_dims=[wm.MAT_COL],
    )

    def whitened_block(kernel, Z, q_m, q_S, M, X):
        Kmm = kernel(Z) + JITTER * pt.eye(M)
        L = pt.linalg.cholesky(Kmm)
        A = pt.linalg.solve_triangular(L, kernel(Z, X), lower=True)
        mu = A.T @ q_m
        SA = q_S @ A
        var = pt.clip(
            kernel.diag(X) - pt.sum(A * A, axis=0) + pt.sum(A * SA, axis=0), 1e-10, np.inf
        )
        _, logdet = pt.linalg.slogdet(q_S)
        kl = 0.5 * (pt.trace(q_S) + q_m @ q_m - M - logdet)
        return mu, var, kl

    mu_b, var_b, kl_b = whitened_block(k_base, p_Z_base, q_m_b, q_S_b, M_BASE, Xb)
    mu_t, var_t, kl_t = whitened_block(k_trend, pt.as_tensor_variable(Z_yr), q_m_t, q_S_t, M_YR, Xb)
    mu_h, var_h, kl_h = whitened_block(k_hist, p_Z_hist, q_m_h, q_S_h, M_HIST, Xb)

    pidx = Xb[:, wm.ID_COL].astype("int64")
    yidx = pt.cast(pt.round(Xb[:, wm.YEAR_COL] * sd_y + mu_y), "int64") - YEAR0
    mf_b, sf_b = m_f[pidx], s_f[pidx]
    my_b, sy_b = m_y[yidx], s_y[yidx]
    logE_b = Xb[:, wm.EXPO_COL]
    scale = N / Xb.shape[0]

    def draw_terms(eps):
        u = p_q_loc + pt.exp(p_q_logsd) * eps
        logv, c_s = u[0:5], u[5]
        v = pt.exp(logv)
        lp = wm.r2d2_log_prior(logv, K_COMP, sigma2, a=1.0, b=99.0)
        lp = lp - 0.5 * (c_s - log_rate0) ** 2
        mu = (
            c_s
            + logE_b
            + pt.sqrt(v[0]) * mu_b
            + pt.sqrt(v[1]) * mu_t
            + pt.sqrt(v[2]) * mu_h
            + mf_b
            + my_b
        )
        var = v[0] * var_b + v[1] * var_t + v[2] * var_h + sf_b + sy_b
        ve = scale * pt.sum(lik.variational_expectation(yb, mu, var))
        kl_year = 0.5 * pt.sum((s_y + m_y**2) / v[3] - 1.0 + logv[3] - pt.log(s_y))
        kl_eps = 0.5 * pt.sum((s_f + m_f**2) / v[4] - 1.0 + logv[4] - pt.log(s_f))
        return ve - kl_year - kl_eps + lp

    inner = pt.mean(pt.stack([draw_terms(pt.constant(BASE_EPS)[s]) for s in range(S_DRAWS)]))
    entropy = pt.sum(p_q_logsd) + 0.5 * 6 * (1.0 + np.log(2 * np.pi))
    lp_point = (
        invgamma_lp(p_log_ls_base, 3.0, 3.0)
        + invgamma_lp(p_log_ls_trend, 3.0, 6.0)
        + invgamma_lp(p_log_ls_hist, 3.0, 3.0)
        - 0.5 * pt.sum(p_W_mat**2)
        - 0.5 * pt.sum(p_W_zone**2)
        - 0.5 * pt.sum(p_W_mat2**2)
        - 0.5 * pt.sum(pt.exp(p_log_kap_mat) ** 2)
        + pt.sum(p_log_kap_mat)
        - 0.5 * pt.sum(pt.exp(p_log_kap_zone) ** 2)
        + pt.sum(p_log_kap_zone)
        - 0.5 * pt.sum(pt.exp(p_log_kap_mat2) ** 2)
        + pt.sum(p_log_kap_mat2)
    )
    elbo = inner + entropy - kl_b - kl_t - kl_h + lp_point
    loss = -elbo

    ngd_wrt = [q_m_b, q_S_b, q_m_t, q_S_t, q_m_h, q_S_h, m_f, s_f, m_y, s_y]
    grads_q = pt.grad(elbo, wrt=ngd_wrt)
    step_fn = pytensor.function(
        [Xb, yb],
        [loss, *grads_q],
        updates=pg.optim.optimizers.adam(loss, adam_params, learning_rate=1e-2),
    )
    step_fn2 = pytensor.function(
        [Xb, yb],
        [loss, *grads_q],
        updates=pg.optim.optimizers.adam(loss, adam_params, learning_rate=2e-3),
    )
    eval_fn = pytensor.function([Xb, yb], loss)

    all_state = adam_params + ngd_wrt
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(N, EVAL_SIZE, replace=False)
    Xev, yev = Xtr[eval_idx], ytr[eval_idx]
    best = {"loss": np.inf, "step": -1, "vals": None}
    gamma_cap = 0.02

    t0 = time.time()
    for i in range(n_steps):
        sel = rng.choice(N, BATCH, replace=False)
        out = (step_fn if i < ADAM_SWITCH else step_fn2)(Xtr[sel], ytr[sel])
        if not np.isfinite(float(out[0])):
            if best["vals"] is not None:
                for var, val in zip(all_state, best["vals"]):
                    var.set_value(val.copy())
            gamma_cap *= 0.5
            print(
                f"    step {i + 1}: non-finite loss; restored best @ {best['step']}, "
                f"gamma_cap -> {gamma_cap:.5f}",
                flush=True,
            )
            continue
        g = out[1:]
        gamma = gamma_sched(i, gamma_cap)
        ngd_update_full(q_m_b, q_S_b, g[0], g[1], gamma)
        ngd_update_full(q_m_t, q_S_t, g[2], g[3], gamma)
        ngd_update_full(q_m_h, q_S_h, g[4], g[5], gamma)
        ngd_update_diag(m_f, s_f, g[6], g[7], gamma)
        ngd_update_diag(m_y, s_y, g[8], g[9], gamma)
        if (i + 1) % EVAL_EVERY == 0 or i == n_steps - 1:
            ev = float(eval_fn(Xev, yev))
            if ev < best["loss"]:
                best = {
                    "loss": ev,
                    "step": i + 1,
                    "vals": [v.get_value().copy() for v in all_state],
                }
    fit_secs = time.time() - t0
    for var, val in zip(all_state, best["vals"]):
        var.set_value(val)

    pred_fn = pytensor.function([Xb], [mu_b, var_b, mu_t, var_t, mu_h, var_h])
    mu_b_te, v_b_te, mu_t_te, v_t_te, mu_h_te, v_h_te = pred_fn(Xte)
    pidx_te = Xte[:, wm.ID_COL].astype(int)
    yidx_te = np.rint(Xte[:, wm.YEAR_COL] * sd_y + mu_y).astype(int) - YEAR0
    mf_te, sf_te = m_f.get_value()[pidx_te], s_f.get_value()[pidx_te]
    my_te, sy_te = m_y.get_value()[yidx_te], s_y.get_value()[yidx_te]

    q_loc, q_sd = p_q_loc.get_value(), np.exp(p_q_logsd.get_value())
    prng = np.random.default_rng(7)
    n_phi, n_z = 100, 5
    u_draws = q_loc + q_sd * prng.standard_normal((n_phi, 6))
    lr_draws = np.empty((n_phi * n_z, len(Xte)))
    for j in range(n_phi):
        v = np.exp(u_draws[j, 0:5])
        mu = (
            u_draws[j, 5]
            + np.sqrt(v[0]) * mu_b_te
            + np.sqrt(v[1]) * mu_t_te
            + np.sqrt(v[2]) * mu_h_te
            + mf_te
            + my_te
        )
        var = v[0] * v_b_te + v[1] * v_t_te + v[2] * v_h_te + sf_te + sy_te
        z = prng.standard_normal((n_z, len(mu)))
        lr_draws[j * n_z : (j + 1) * n_z] = mu[None, :] + np.sqrt(var)[None, :] * z

    n_test = test["y"].to_numpy(float)
    expo = test["length_km"].to_numpy(float)
    row_elpd = wm.elpd_row_poisson(n_test, lr_draws, expo)
    lam = expo[None, :] * np.exp(lr_draws)
    EN = lam.mean(0)
    VN = EN + lam.var(0)
    prob = 1.0 - np.exp(-lam).mean(0)

    return {
        "row_elpd": row_elpd,
        "EN": EN,
        "VN": VN,
        "prob": prob,
        "q_loc": q_loc,
        "q_sd": q_sd,
        "sigma2": sigma2,
        "m_y_all": m_y.get_value(),
        "s_y_all": s_y.get_value(),
        "ls_base": np.exp(p_log_ls_base.get_value()).tolist(),
        "ls_trend": float(np.exp(p_log_ls_trend.get_value())),
        "ls_hist": np.exp(p_log_ls_hist.get_value()).tolist(),
        "best_step": best["step"],
        "fit_secs": fit_secs,
    }


def row_ll(fold, key, draws=500):
    d = fold[key]
    lr = (
        wm.latent_draws(d["fm"], d["fv"], draws, seed=0)
        if "fm" in d
        else np.log(np.clip(d["rate"], 1e-9, None))[None, :]
    )
    return wm.elpd_row_poisson(fold["n_test"], lr, fold["expo_test"])


for t in FOLDS:
    cache = Path(f"watermains_cv_m4_{t}.pkl")
    if cache.exists():
        d = pickle.loads(cache.read_bytes())
    else:
        d = fit_m4(*folds[t])
        cache.write_bytes(pickle.dumps(d))
    fold = pickle.loads(Path(f"watermains_cv_seq97_{t}.pkl").read_bytes())
    v_mean = np.exp(d["q_loc"][0:5] + 0.5 * d["q_sd"][0:5] ** 2)
    phi = v_mean / v_mean.sum()
    d2m, se2m = wm.elpd_diff_se(d["row_elpd"], row_ll(fold, "m2"))
    d2h, se2h = wm.elpd_diff_se(d["row_elpd"], row_ll(fold, "hgb"))
    print(
        f"[{t}] m4 fit {d['fit_secs']:.0f}s best@{d['best_step']} | phi={np.round(phi, 3)} "
        f"sigma_y={np.sqrt(v_mean[3]):.3f} | m4-m2: {d2m:+.1f}+-{se2m:.1f} "
        f"m4-hgb: {d2h:+.1f}+-{se2h:.1f}",
        flush=True,
    )

# ---- pooled five-model table ---------------------------------------------
rows = {}
diff_rows = {"m2": [], "hgb": []}
for key in ("m1", "m2", "hgb", "glm", "m4"):
    elpd, z2, captured = 0.0, [], 0.0
    y_pool, p_pool, tot_pred, tot_real = [], [], [], []
    for t in sorted(FOLDS):
        fold = pickle.loads(Path(f"watermains_cv_seq97_{t}.pkl").read_bytes())
        n_test, expo = fold["n_test"], fold["expo_test"]
        if key == "m4":
            d = pickle.loads(Path(f"watermains_cv_m4_{t}.pkl").read_bytes())
            EN, VN, prob = d["EN"], d["VN"], d["prob"]
            elpd += float(d["row_elpd"].sum())
            for ref in ("m2", "hgb"):
                diff_rows[ref].append(d["row_elpd"] - row_ll(fold, ref))
        else:
            dd = fold[key]
            if "fm" in dd:
                EN, VN = wm.poisson_moments(dd["fm"], dd["fv"], expo)
                prob = wm.rate_to_prob(wm.latent_to_rate(dd["fm"], dd["fv"]), expo)
                elpd += wm.elpd_poisson(
                    n_test, wm.latent_draws(dd["fm"], dd["fv"], 500, seed=0), expo
                )
            else:
                rate = np.clip(dd["rate"], 1e-9, None)
                EN = VN = rate * expo
                prob = wm.rate_to_prob(rate, expo)
                elpd += wm.elpd_poisson(n_test, np.log(rate)[None, :], expo)
        y_pool.append((n_test > 0).astype(int))
        p_pool.append(prob)
        z2.append((n_test - EN) ** 2 / VN)
        rc = wm.replacement_cost(fold["diam"], fold["length_km"])
        captured += wm.dollar_backtest({key: EN}, rc, n_test, budget=5.0)[key][2]
        tot_pred.append(EN.sum())
        tot_real.append(n_test.sum())
    pooled = wm.evalm(np.concatenate(y_pool), np.concatenate(p_pool))
    rows[key] = {
        "ELPD": round(elpd, 1),
        "ROC-AUC": round(pooled["ROC_AUC"], 3),
        "AUC-PR": round(pooled["AP"], 3),
        "captured@5M": round(captured, 1),
        "z2": round(float(np.concatenate(z2).mean()), 2),
        "bias/yr": round(float(np.mean(np.array(tot_pred) - np.array(tot_real))), 1),
    }

print("\n===== POOLED (1997 panel, M=1024, m4 @ 2500 steps) =====")
print(pd.DataFrame(rows).T.to_string())
for ref in ("m2", "hgb"):
    dr = np.concatenate(diff_rows[ref])
    print(f"pooled m4 - {ref}: {dr.sum():+.1f} +- {np.sqrt(len(dr) * dr.var(ddof=1)):.1f}")
print("DONE M4 CV")
