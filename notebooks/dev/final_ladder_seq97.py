"""Full ladder on the corrected 1997 panel: M1 (normalized cats), M2 (R2D2 Beta(1,99)),
retuned HGB, GLM. 7 folds, M=1024, caches watermains_cv_seq97_{t}.pkl."""
# ruff: noqa: E402

import pickle
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
import pandas as pd
import watermains as wm

import ptgp as pg

ALL_FOLDS = [2025, 2019, 2020, 2021, 2022, 2023, 2024]
# WM_FOLDS lets a worker fit a subset of folds (parallel-by-fold launcher); the pooled
# summary only runs when the full fold set is requested (the final summary pass).
FOLDS = (
    [int(x) for x in os.environ["WM_FOLDS"].split(",")] if os.environ.get("WM_FOLDS") else ALL_FOLDS
)
RUN_SUMMARY = set(FOLDS) == set(ALL_FOLDS)
M_IND, N_STEPS, BATCH, EVAL_SIZE = 1024, 3000, 1024, 32768

breaks, mains, mains_geoms = wm.load_kitchener_data(cache_path="watermains_cache.pkl")
folds = {t: wm.build_panel(mains, breaks, float(t)) for t in FOLDS}
if 2025 in FOLDS:
    tr25 = folds[2025][0]
    print(
        f"1997 panel, fold 2025: {len(tr25):,} train rows, {int(tr25['y'].sum()):,} breaks, "
        f"mean rate/row {tr25['y'].mean():.5f}",
        flush=True,
    )


def fit_m1(train, test, seed=0):
    Xtr, stats = wm.panel_design(train)
    Xte, _ = wm.panel_design(test, stats)
    ytr = train["y"].to_numpy(float)
    n_mat = int(Xtr[:, wm.MAT_COL].max()) + 1
    n_zone = int(Xtr[:, wm.ZONE_COL].max()) + 1
    Z0, vp, _ = wm.init_inducing(Xtr, M=M_IND, rng=seed)
    model, svgp, Z_var = wm.build_svgp_model(Xtr, ytr, n_mat, n_zone, Z0, vp, seed=seed)
    res = wm.train_svgp(
        model,
        svgp,
        Z_var,
        Z0,
        vp,
        Xtr,
        ytr,
        batch_size=BATCH,
        n_steps=N_STEPS,
        seed=seed,
        print_every=None,
        eval_size=EVAL_SIZE,
        eval_every=50,
    )
    rate_fn = wm.compile_rate_fn(svgp, model, res)
    _, fm, fv = rate_fn(Xte)
    tr = pg.optim.get_trained_params(model, res.shared)
    return {
        "fm": fm,
        "fv": fv,
        "eta": float(tr["eta"]),
        "ls": np.asarray(tr["ls"]).tolist(),
        "best_step": res.best_step,
    }


def load_or_fit(t, key, fit_fn):
    cache = Path(f"watermains_cv_seq97_{t}.pkl")
    fold = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    train, test = folds[t]
    if "n_test" not in fold:
        fold.update(
            n_test=test["y"].to_numpy(float),
            expo_test=test["length_km"].to_numpy(float),
            diam=test["diam"].to_numpy(float),
            length_km=test["length_km"].to_numpy(float),
        )
    if key not in fold:
        t0 = time.time()
        fold[key] = fit_fn(train, test)
        cache.write_bytes(pickle.dumps(fold))
        print(f"[{t}] {key} fit ({time.time()-t0:.0f}s)", flush=True)
    return fold


def row_ll(fold, key, draws=500):
    d = fold[key]
    lr = (
        wm.latent_draws(d["fm"], d["fv"], draws, seed=0)
        if "fm" in d
        else np.log(np.clip(d["rate"], 1e-9, None))[None, :]
    )
    return wm.elpd_row_poisson(fold["n_test"], lr, fold["expo_test"])


for t in FOLDS:
    fold = load_or_fit(t, "m1", fit_m1)
    fold = load_or_fit(
        t,
        "m2",
        lambda tr, te: wm.fit_panel_gp_r2d2(
            tr, te, M=M_IND, n_steps=N_STEPS, batch_size=BATCH, eval_size=EVAL_SIZE
        ),
    )
    fold = load_or_fit(
        t,
        "hgb",
        lambda tr, te, t=t: dict(zip(("rate", "params"), wm.fit_hgb_panel(tr, te, float(t)))),
    )
    fold = load_or_fit(t, "glm", lambda tr, te: {"rate": wm.fit_glm_panel(tr, te)})
    d2 = fold["m2"]
    d21, se21 = wm.elpd_diff_se(row_ll(fold, "m2"), row_ll(fold, "m1"))
    d2h, se2h = wm.elpd_diff_se(row_ll(fold, "m2"), row_ll(fold, "hgb"))
    print(
        f"[{t}] eta={fold['m1']['eta']:.3f} | m2 phi={np.round(d2['phi'], 3)} R2={d2['r2']:.4f} "
        f"betas={np.round(d2['betas'], 3)} | m2-m1: {d21:+.1f}+-{se21:.1f} m2-hgb: {d2h:+.1f}+-{se2h:.1f}",
        flush=True,
    )

if not RUN_SUMMARY:
    print(f"folds {FOLDS} done (subset run); pooled summary skipped", flush=True)
    raise SystemExit(0)

rows = {}
etas = []
for key in ("m1", "m2", "hgb", "glm"):
    elpd, z2, captured = 0.0, [], 0.0
    y_pool, p_pool, tot_pred, tot_real = [], [], [], []
    for t in sorted(FOLDS):
        fold = pickle.loads(Path(f"watermains_cv_seq97_{t}.pkl").read_bytes())
        n_test, expo = fold["n_test"], fold["expo_test"]
        d = fold[key]
        if "fm" in d:
            EN, VN = wm.poisson_moments(d["fm"], d["fv"], expo)
            prob = wm.rate_to_prob(wm.latent_to_rate(d["fm"], d["fv"]), expo)
            elpd += wm.elpd_poisson(n_test, wm.latent_draws(d["fm"], d["fv"], 500, seed=0), expo)
            if key == "m1":
                etas.append(d["eta"])
        else:
            rate = np.clip(d["rate"], 1e-9, None)
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
print("\n===== POOLED (1997 panel, M=1024) =====")
print(pd.DataFrame(rows).T.to_string())

etas = np.array(etas)
s2 = 1.0 / tr25["y"].to_numpy(float).mean()
fm_var = [
    float(np.var(pickle.loads(Path(f"watermains_cv_seq97_{t}.pkl").read_bytes())["m1"]["fm"]))
    for t in sorted(FOLDS)
]
r2a = etas**2 / (etas**2 + s2)
r2r = np.array(fm_var) / (np.array(fm_var) + s2)
print(f"\nsigma_tilde2 (1/mean y) = {s2:.1f}")
print(f"eta across folds: {np.round(etas, 3)}; amplitude R2 {r2a.mean():.4f} +- {r2a.std():.4f}")
print(f"realized R2 {r2r.mean():.4f} +- {r2r.std():.4f}")
print("DONE")
