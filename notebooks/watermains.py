"""Companion module for the water-mains decision-theory example notebooks.

Holds the mechanical code (data fetching, feature coding, model builders, decision
rules, evaluation, plotting) so the notebooks stay short. Import as::

    import watermains as wm

Unit tests live at the bottom of this file; run them with
``python -m pytest notebooks/watermains.py``. The module is intentionally not part
of the ptgp package and is not collected by the main test suite.
"""

from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from matplotlib.collections import LineCollection
from scipy.special import gammaln

import ptgp as pg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# City of Kitchener open data (ArcGIS feature services)
BREAKS_URL = "https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Water_Main_Breaks/FeatureServer/0"
MAINS_URL = (
    "https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Water_Mains/FeatureServer/0"
)

AS_OF = 2026.0  # data runs to mid-2026
# Break records nominally begin in 1985, but coverage is effectively complete only
# from 1997 (8 records total in 1985-94 and one each in 1995 and 1996, vs 99-125
# per year from 1997 on).
RECORD_START = 1997.0

# Consolidate near-duplicate / negligible categories (justified in the notebook tables).
MAT_MAP = {"HDPE IN CI": "HDPE", "COP": "OTHER", "ST": "OTHER", "XXX": "OTHER"}
PZ_MAP = {"RAW NO ZONE": "OTHER"}

# Pipes belonging to neighbouring municipalities (Waterloo, Cambridge, Breslau/Woolwich,
# Mannheim/Wilmot): a different utility's system, no breaks, some mislabeled.
DROP_ZONES = ["WAT 4", "CAM 1", "CAM 2W", "BRESLAU", "BRESLAU NORTH", "BRESLAU SOUTH", "MANNHEIM"]

# Panel design-matrix layout: one row per pipe per year. Columns 0-3 continuous
# (standardized age-at-year, log_size, lon, lat), 4 standardized calendar year,
# 5 material code, 6 zone code, 7 log exposure, 8 history feature, 9 pipe index.
# Kernels only ever touch 0-6; the mean function reads 7 (and 8 when the history
# term is on); the frailty reads 9.
CONT_DIMS = [0, 1, 2, 3]
YEAR_COL = 4
MAT_COL = 5
ZONE_COL = 6
EXPO_COL = 7
HIST_COL = 8
ID_COL = 9
PANEL_START = 1997.0  # first target year of the pipe-year panel (full effective record)

# Decision-theory prices
H_YEARS = 30.0  # planning horizon (years)
C_BREAK = 25000.0  # cost per break ($)
C_SURGE = 40000.0  # surge premium per break above the provisioned budget ($)
C_IDLE = 8000.0  # idle-capacity cost per provisioned-but-unused break ($)

COLORS = {"CI": "tab:red", "DI": "tab:orange", "PVC": "tab:green"}
MAP_ASPECT = 1.0 / np.cos(np.deg2rad(43.45))  # unit aspect at Kitchener's latitude


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def fetch_features(url, out_sr=4326, where="1=1", page=2000):
    """Page through an ArcGIS FeatureServer layer and return its GeoJSON features.

    Parameters
    ----------
    url : str
        Base layer URL (without the trailing ``/query``).
    out_sr : int
        Output spatial reference (default WGS84).
    where : str
        ArcGIS where clause (default: all rows).
    page : int
        Rows per request.
    """
    import requests

    feats, offset = [], 0
    while True:
        params = dict(
            where=where,
            outFields="*",
            f="geojson",
            outSR=out_sr,
            resultOffset=offset,
            resultRecordCount=page,
        )
        r = requests.get(url + "/query", params=params, timeout=120)
        r.raise_for_status()
        batch = r.json().get("features", [])
        feats.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return feats


def features_to_df(feats):
    """Split GeoJSON features into a properties DataFrame and a list of geometries."""
    df = pd.DataFrame([f["properties"] for f in feats])
    geoms = [f.get("geometry") for f in feats]
    return df, geoms


def centroid(geom):
    """Mean of a LineString or MultiLineString's coordinates as ``(lon, lat)``."""
    if not geom:
        return (np.nan, np.nan)
    if geom["type"] == "LineString":
        cs = geom["coordinates"]
    else:  # MultiLineString
        cs = [p for part in geom["coordinates"] for p in part]
    arr = np.asarray(cs, float)
    return arr[:, 0].mean(), arr[:, 1].mean()


def load_kitchener_data(cache_path=None):
    """Fetch the Kitchener breaks and mains layers, ready for modeling.

    Returns ``(breaks, mains, mains_geoms)``. Breaks carry point ``lon``/``lat`` and a
    parsed ``INCIDENT_DATE``; mains carry ``install_year`` and centroid ``lon``/``lat``.

    Parameters
    ----------
    cache_path : str or pathlib.Path, optional
        If given, load the result from this pickle when it exists and write it
        after fetching otherwise, so reruns skip the network.
    """
    import pickle

    from pathlib import Path

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    bf = fetch_features(BREAKS_URL)
    mf = fetch_features(MAINS_URL)

    breaks, breaks_geoms = features_to_df(bf)
    breaks["lon"] = [g["coordinates"][0] if g else np.nan for g in breaks_geoms]
    breaks["lat"] = [g["coordinates"][1] if g else np.nan for g in breaks_geoms]
    breaks["INCIDENT_DATE"] = pd.to_datetime(breaks["INCIDENT_DATE"], unit="ms", errors="coerce")

    mains, mains_geoms = features_to_df(mf)
    mains["install_year"] = pd.to_datetime(
        mains["INSTALLATION_DATE"], unit="ms", errors="coerce"
    ).dt.year
    cent = np.array([centroid(g) for g in mains_geoms])
    mains["lon"], mains["lat"] = cent[:, 0], cent[:, 1]

    out = (breaks, mains, mains_geoms)
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
    return out


def attach_break_year(breaks):
    """Return a copy of the breaks frame with an integer ``break_year`` column.

    Expects ``INCIDENT_DATE`` already parsed to datetimes (as from
    ``load_kitchener_data``).
    """
    out = breaks.copy()
    out["break_year"] = out["INCIDENT_DATE"].dt.year
    return out


def exposure(length_km, install_year, as_of=AS_OF, start=RECORD_START):
    """Pipe-km-years at risk: length times years observed within the record window.

    Years before ``start`` do not count because no breaks were recorded then, so
    ``install_year`` is clamped up to ``start``.
    """
    return length_km * (as_of - np.maximum(install_year, start))


def build_modeling_frame(mains, breaks, drop_zones=DROP_ZONES, as_of=AS_OF, start=RECORD_START):
    """Build the modeling frame of active mains with break counts and exposure.

    Filters to MAIN breaks (dropping the two extreme-longitude spatial outliers),
    links them to mains by asset id, keeps active mains with plausible install years
    and sizes, and drops fringe-municipality pressure zones.

    Returns
    -------
    m : pandas.DataFrame
        Modeling frame with ``age``, ``length_km``, ``exposure``, ``n_breaks``.
    mb : pandas.DataFrame
        The filtered MAIN-break records (for maps and prior-break features).
    pz_source : pandas.Series
        Pressure-zone labels before the fringe zones were dropped (for tables).
    """
    mb = breaks[breaks["BREAK_TYPE"].eq("MAIN")]
    mb = mb[mb["lon"].between(mb["lon"].min(), mb["lon"].max(), inclusive="neither")]
    bc = mb.loc[mb["ASSETID"].isin(set(mains["WATMAINID"])), "ASSETID"].value_counts()

    mains = mains.copy()
    mains["n_breaks"] = mains["WATMAINID"].map(bc).fillna(0).astype(int)

    m = mains[(mains["STATUS"] == "ACTIVE") & mains["install_year"].between(1850, as_of)].copy()
    m["age"] = as_of - m["install_year"]
    m["length_km"] = m["Shape__Length"] / 1000.0
    m["exposure"] = exposure(m["length_km"], m["install_year"], as_of=as_of, start=start)
    m = m[(m["exposure"] > 0) & m["lat"].notna() & m["PIPE_SIZE"].between(25, 1200)]

    pz_source = m["PRESSURE_ZONE"].copy()
    m = m[~m["PRESSURE_ZONE"].isin(drop_zones)]
    return m, mb, pz_source


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def consolidate_categories(series, mapping):
    """Merge near-duplicate or negligible category labels via ``mapping``."""
    return series.replace(mapping)


def frequency_codes(series):
    """Integer level codes, most common level first.

    Keeps LowRankCategorical ``W`` rows interpretable: row 0 is the most common level.

    Returns
    -------
    codes : numpy.ndarray
        Float array of integer codes.
    levels : list
        Level labels in code order.
    """
    levels = series.value_counts().index.tolist()
    codes = pd.Categorical(series, categories=levels).codes.astype(float)
    return codes, levels


def standardize(df, mean=None, std=None):
    """Standardize columns; pass a training ``mean``/``std`` for leak-free splits.

    Returns ``(Z, mean, std)`` where ``Z`` is the standardized frame.
    """
    if mean is None:
        mean, std = df.mean(), df.std()
    return (df - mean) / std, mean, std


def static_frame(mains, drop_zones=DROP_ZONES):
    """Static per-pipe frame with consolidated integer codes, shared by all folds.

    ``geom_idx`` keeps each pipe's row position in the original ``mains`` frame so
    geometries stay addressable after filtering.
    """
    base = mains[(mains["STATUS"] == "ACTIVE") & mains["install_year"].between(1850, AS_OF)]
    base = base[base["PIPE_SIZE"].between(25, 1200) & base["lat"].notna()]
    base = base[~base["PRESSURE_ZONE"].isin(drop_zones)].copy()
    base["geom_idx"] = base.index
    base = base.reset_index(drop=True)
    base["length_km"] = base["Shape__Length"] / 1000.0
    base["mat_code"], base.attrs["mat_levels"] = frequency_codes(
        consolidate_categories(base["MATERIAL"], MAT_MAP)
    )
    base["zone_code"], base.attrs["zone_levels"] = frequency_codes(
        consolidate_categories(base["PRESSURE_ZONE"], PZ_MAP)
    )
    base["pipe_idx"] = np.arange(len(base), dtype=float)
    return base


def break_table(base, breaks):
    """Per-pipe, per-year counts of linked MAIN breaks (pipes x integer years)."""
    mb = attach_break_year(breaks[breaks["BREAK_TYPE"].eq("MAIN")])
    bym = mb[mb["ASSETID"].isin(set(base["WATMAINID"]))]
    return pd.crosstab(bym["ASSETID"], bym["break_year"].astype(int))


def _year_counts(ct, ids, s):
    """Break counts in year ``s`` for the given asset ids."""
    if s in ct.columns:
        return ids.map(ct[s]).fillna(0).to_numpy(float)
    return np.zeros(len(ids))


def _hist_rate(ct, sub, s, hist_start=RECORD_START):
    """Empirical prior break rate (per km-year) over ``[hist_start, s)`` per pipe.

    A rate rather than a count so the feature keeps its scale as the history
    window grows with the row's year. Pipes with no exposure in the window get 0.
    """
    cols = [c for c in ct.columns if hist_start <= c < s]
    n = sub["WATMAINID"].map(ct[cols].sum(axis=1)).fillna(0).to_numpy(float) if cols else 0.0
    years = s - np.maximum(sub["install_year"].to_numpy(float), hist_start)
    expo = sub["length_km"].to_numpy(float) * np.maximum(years, 0.0)
    return np.where(expo > 0, n / np.where(expo > 0, expo, 1.0), 0.0)


def build_panel(mains, breaks, t, target_start=PANEL_START):
    """Pipe-year panel for one walk-forward fold at planning year ``t``.

    One row per active pipe per year ``s`` in ``[target_start, t]``: target = that
    pipe's breaks in year ``s``, exposure = one year of its length, covariates
    measured at ``s``, and the history feature computed strictly before ``s``.
    Returns ``(train, test)`` where train rows have ``year < t`` and the test slice
    is the planning year itself.
    """
    base = static_frame(mains)
    ct = break_table(base, breaks)
    frames = []
    for s in range(int(target_start), int(t) + 1):
        sub = base[base["install_year"] <= s - 1]
        frames.append(
            pd.DataFrame(
                {
                    "pipe_idx": sub["pipe_idx"].to_numpy(),
                    "year": float(s),
                    "age": s - sub["install_year"].to_numpy(float),
                    "log_size": np.log(sub["PIPE_SIZE"].to_numpy(float)),
                    "lon": sub["lon"].to_numpy(float),
                    "lat": sub["lat"].to_numpy(float),
                    "mat_code": sub["mat_code"].to_numpy(float),
                    "zone_code": sub["zone_code"].to_numpy(float),
                    "length_km": sub["length_km"].to_numpy(float),
                    "diam": sub["PIPE_SIZE"].to_numpy(float),
                    "y": _year_counts(ct, sub["WATMAINID"], s),
                    "hist": _hist_rate(ct, sub, s),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    return panel[panel["year"] < t], panel[panel["year"] == t]


def panel_design(df, stats=None):
    """Design matrix for panel rows, in the canonical column layout.

    Pass the training ``stats`` when building a test matrix so continuous columns
    and the history feature are standardized leak-free. Returns ``(X, stats)``.
    """
    cont = df[["age", "log_size", "lon", "lat", "year"]]
    hist = np.log1p(df["hist"].to_numpy(float))
    if stats is None:
        stats = (cont.mean(), cont.std(), float(hist.mean()), max(float(hist.std()), 1e-9))
    cont_z = (cont - stats[0]) / stats[1]
    X = np.column_stack(
        [
            cont_z.to_numpy(),
            df["mat_code"].to_numpy(float),
            df["zone_code"].to_numpy(float),
            np.log(df["length_km"].to_numpy(float)),  # one year of exposure
            (hist - stats[2]) / stats[3],
            df["pipe_idx"].to_numpy(float),
        ]
    ).astype(np.float64)
    return X, stats


# ---------------------------------------------------------------------------
# Mean functions and model builders
# ---------------------------------------------------------------------------


class ExposureOffset:
    """Mean function returning ``c + log_exposure``, the latent baseline log-rate.

    The constant ``c`` is a free intercept, so exp(c) is the break rate per
    pipe-km-year at the reference and the GP models deviations around it. The
    log-exposure is read from column ``col`` of X so it travels with each minibatch;
    zero that column to predict the offset-free log-rate. With the standard ptgp
    Poisson likelihood this gives y ~ Poisson(exp(c + f + log_exposure)).
    """

    def __init__(self, c, col):
        self.c, self.col = c, col

    def __call__(self, X):
        return self.c + X[:, self.col]


class ExposureOffsetLinear:
    """``ExposureOffset`` plus a linear term ``beta * X[:, feat_col]``.

    Used by Model 2 to add prior break count as a linear predictor in the mean
    rather than a kernel dimension, assuming no interactions with the other
    covariates. ``feat_col`` is zeroed together with ``col`` only if the feature
    should be excluded at prediction time; normally only the exposure column is
    zeroed and the linear term stays active.
    """

    def __init__(self, c, col, beta, feat_col):
        self.c, self.col = c, col
        self.beta, self.feat_col = beta, feat_col

    def __call__(self, X):
        return self.c + self.beta * X[:, self.feat_col] + X[:, self.col]


class LinearMainEffects:
    """Mean function ``c + coeffs . X[:, cols] + log-exposure offset``.

    The general form of ``ExposureOffsetLinear``: any number of design columns
    enter linearly, each with its own coefficient.
    """

    def __init__(self, c, expo_col, coeffs, cols):
        self.c, self.expo_col = c, expo_col
        self.coeffs, self.cols = coeffs, list(cols)

    def __call__(self, X):
        lin = pt.sum(self.coeffs * X[:, self.cols], axis=1)
        return self.c + lin + X[:, self.expo_col]


def init_inducing(X, M=512, cont_dims=None, code_cols=(MAT_COL, ZONE_COL), subsample=30000, rng=0):
    """Select inducing points by greedy variance reduction and round code columns.

    Uses a unit-lengthscale Matern proxy over the continuous dims, which carry the
    function's smooth variation; the categorical correlations are learned later.
    Selection runs on a random subsample when the panel is larger than
    ``subsample`` rows. Selected rows are real data, so rounding restores exact
    integer codes.

    Returns ``(Z0, vp, ip_diag)``: the inducing matrix, fresh variational
    parameters, and the greedy-selection diagnostics.
    """
    if cont_dims is None:
        cont_dims = CONT_DIMS
    if len(X) > subsample:
        X = X[np.random.default_rng(rng).choice(len(X), subsample, replace=False)]
    proxy = pg.kernels.Matern52(input_dim=X.shape[1], ls=1.0, active_dims=list(cont_dims))
    points, ip_diag = pg.inducing.greedy_variance_init(X, M, kernel=proxy, rng=rng)
    Z0 = points.Z.copy()
    for c in code_cols:
        Z0[:, c] = np.round(Z0[:, c])
    vp = pg.gp.init_variational_params(len(Z0))
    return Z0, vp, ip_diag


def inducing_diagnostics(X, Z0, cont_dims=None, subsample=30000, rng=0):
    """Per-point residual variance of the greedy inducing set (Nystrom diagonal).

    Evaluated with the same unit-lengthscale Matern proxy used for selection, on a
    random subsample when ``X`` is large. Returns ``(d_final, idx)`` where ``idx``
    are the sampled row positions.
    """
    if cont_dims is None:
        cont_dims = CONT_DIMS
    idx = np.arange(len(X))
    if len(X) > subsample:
        idx = np.random.default_rng(rng).choice(len(X), subsample, replace=False)
    proxy = pg.kernels.Matern52(input_dim=X.shape[1], ls=1.0, active_dims=list(cont_dims))
    health = pg.inducing.compute_inducing_diagnostics(proxy, X[idx], Z0)
    return health.d_final, idx


def plot_inducing_diagnostics(ip_diag, d_final, age):
    """Three views of inducing-set coverage: trace curve, fraction unexplained, residuals."""
    tc, tv = ip_diag.trace_curve, ip_diag.total_variance
    iters = np.arange(len(tc))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    axes[0].plot(iters, tc)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set(
        title="Residual variance vs M", xlabel="inducing points added", ylabel="tr(Kff - Q)"
    )
    axes[1].plot(iters, tc / tv)
    axes[1].axhline(0.01, color="C1", lw=1, ls="--", label="1% threshold")
    axes[1].set(
        title="Fraction unexplained vs M",
        xlabel="inducing points added",
        ylabel="tr(Kff - Q) / tr(Kff)",
    )
    axes[1].legend(fontsize=8)
    axes[2].scatter(age, d_final, s=6, alpha=0.5)
    axes[2].set(
        title="Per-point residual variance",
        xlabel="pipe age (years)",
        ylabel="conditional variance",
    )
    plt.tight_layout()
    plt.show()
    print(
        f"M = {len(tc)}: {100 * (1 - tc[-1] / tv):.1f}% of prior variance explained; "
        f"worst-covered point d = {d_final.max():.3f}"
    )


class NormalizedLowRankCategorical(pg.kernels.LowRankCategorical):
    """Low-rank categorical kernel with its overall scale pinned to 1.

    Dividing ``B = W W^T + diag(kappa)`` by its frequency-weighted mean diagonal
    removes the scale degeneracy between the categorical factors and the shared
    amplitude ``eta`` in a product kernel: relative per-level variances are
    preserved while ``eta**2`` becomes the identified mean per-point prior
    variance of the GP.
    """

    def __init__(self, input_dim, num_levels, W, kappa, freqs, active_dims=None):
        super().__init__(input_dim, num_levels, W, kappa, active_dims=active_dims)
        self.freqs = pt.as_tensor_variable(np.asarray(freqs, float))

    def _scale(self):
        Bdiag = pt.sum(pt.square(self.W), axis=-1) + self.kappa
        return pt.sum(self.freqs * Bdiag)

    def _eval(self, X, Y):
        return super()._eval(X, Y) / self._scale()

    def diag(self, X):
        return super().diag(X) / self._scale()


def build_svgp_model(
    X,
    y,
    n_mat,
    n_zone,
    Z0,
    vp,
    year=False,
    hist_mean=False,
    hist_kernel=False,
    n_mat_latents=2,
    n_zone_latents=2,
    seed=0,
):
    """Build the SVGP rate model on panel rows.

    The kernel is Matern52 ARD over the continuous dims (plus the calendar-year
    dim when ``year=True``) times normalized low-rank-plus-diagonal categorical
    kernels for material and pressure zone (scale pinned so ``eta**2`` is the
    identified per-point prior variance); the mean carries the exposure offset,
    plus a linear term on the history column when ``hist_mean=True``.
    ``hist_kernel`` also gives the history feature an ARD dimension, letting it
    interact with the other covariates. Model 1 is all-off; Model 2 turns year
    and history (kernel and mean) on; Model 3 adds the frailty on top (built
    inline in the notebook).

    Returns ``(model, svgp, Z_var)`` where ``Z_var`` is the trainable inducing-point
    matrix variable to pass to ``train_svgp``.
    """
    input_dim = X.shape[1]
    cont_dims = [*CONT_DIMS, YEAR_COL] if year else list(CONT_DIMS)
    if hist_kernel:
        cont_dims = [*cont_dims, HIST_COL]

    # W = 0 is a stationary point of the ELBO (the gradient through W W^T vanishes
    # there), so a prior-median start leaves W stuck at zero. Seed it with small
    # random values to break the symmetry.
    wrng = np.random.default_rng(seed)
    W_mat_init = 0.1 * wrng.standard_normal((n_mat, n_mat_latents))
    W_zone_init = 0.1 * wrng.standard_normal((n_zone, n_zone_latents))

    freq_mat = np.bincount(X[:, MAT_COL].astype(int), minlength=n_mat) / len(X)
    freq_zone = np.bincount(X[:, ZONE_COL].astype(int), minlength=n_zone) / len(X)

    # Centre the intercept prior on the overall empirical rate so the GP models
    # deviations around the right baseline.
    log_rate0 = float(np.log(y.sum() / np.exp(X[:, EXPO_COL]).sum()))

    with pm.Model() as model:
        c = pm.Normal("c", mu=log_rate0, sigma=1.0)
        ls = pm.InverseGamma("ls", alpha=3.0, beta=3.0, shape=len(cont_dims))
        eta = pm.HalfNormal("eta", sigma=1.5)
        k_cont = eta**2 * pg.kernels.Matern52(input_dim=input_dim, ls=ls, active_dims=cont_dims)

        W_mat = pm.Normal("W_mat", 0.0, 1.0, shape=(n_mat, n_mat_latents), initval=W_mat_init)
        kappa_mat = pm.HalfNormal("kappa_mat", sigma=1.0, shape=n_mat)
        k_mat = NormalizedLowRankCategorical(
            input_dim=input_dim,
            num_levels=n_mat,
            W=W_mat,
            kappa=kappa_mat,
            freqs=freq_mat,
            active_dims=[MAT_COL],
        )

        W_zone = pm.Normal("W_zone", 0.0, 1.0, shape=(n_zone, n_zone_latents), initval=W_zone_init)
        kappa_zone = pm.HalfNormal("kappa_zone", sigma=1.0, shape=n_zone)
        k_zone = NormalizedLowRankCategorical(
            input_dim=input_dim,
            num_levels=n_zone,
            W=W_zone,
            kappa=kappa_zone,
            freqs=freq_zone,
            active_dims=[ZONE_COL],
        )

        if hist_mean:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0)
            mean = ExposureOffsetLinear(c, EXPO_COL, beta, HIST_COL)
        else:
            mean = ExposureOffset(c, EXPO_COL)

        # Trainable inducing points: the continuous columns are optimized; the code
        # columns stay fixed because the int cast in the categorical kernel has zero
        # gradient (initialised from the greedy Z0).
        Z_var = pt.matrix("Z")
        svgp = pg.gp.SVGP(
            kernel=k_cont * k_mat * k_zone,
            likelihood=pg.likelihoods.Poisson(),
            mean=mean,
            inducing_variable=pg.inducing.Points(Z_var, Z_init=Z0),
            variational_params=vp,
        )
    return model, svgp, Z_var


class TrainResult(NamedTuple):
    """Loss trace plus the compiled shared state needed for prediction."""

    losses: list
    shared: dict
    extras: object
    extra_vars: list
    eval_curve: list | None = None
    best_step: int | None = None


def train_svgp(
    model,
    svgp,
    Z_var,
    Z0,
    vp,
    X,
    y,
    batch_size=1024,
    n_steps=300,
    learning_rate=1e-2,
    seed=0,
    print_every=20,
    eval_size=None,
    eval_every=50,
    keep_best=True,
    objective=None,
    extra_vars=None,
    extra_init=None,
):
    """Fit the SVGP by minibatch Adam on the ELBO; returns a ``TrainResult``.

    Pass ``batch_size=None`` for full-batch steps. The printed loss is
    per-minibatch, so expect sampling noise.

    When ``eval_size`` is given, a fixed evaluation batch of that many rows is
    drawn once and the (deterministic) objective is evaluated on it every
    ``eval_every`` steps: the minibatch loss is too noisy to compare checkpoints
    (its swings mostly reflect how many breaks land in a batch), while the fixed
    batch gives an honest curve. With ``keep_best`` the parameters from the best
    fixed-eval checkpoint are restored at the end.

    ``objective`` overrides the plain ELBO (e.g. to add the R2D2 budget prior);
    ``extra_vars``/``extra_init`` override the point-estimated variable list when
    the objective introduces its own (both default to the variational parameters
    plus the inducing points).
    """
    Xv, yv = pt.matrix("X"), pt.vector("y")
    if objective is None:

        def objective(gp, Xb, yb):
            return pg.objectives.elbo(gp, Xb, yb, n_data=len(y)).elbo

    if extra_vars is None:
        extra_vars = [*vp.extra_vars, Z_var]
        extra_init = [*vp.extra_init, Z0]
    step, shared, extras = pg.optim.compile_training_step(
        objective,
        svgp,
        Xv,
        yv,
        model=model,
        extra_vars=extra_vars,
        extra_init=extra_init,
        learning_rate=learning_rate,
    )
    eval_fn, Xe, ye = None, None, None
    if eval_size is not None:
        from ptgp.optim.training import _replace_graph

        e_idx = np.random.default_rng(seed + 1).choice(
            len(y), min(eval_size, len(y)), replace=False
        )
        Xe, ye = X[e_idx], y[e_idx]
        loss_sym = -objective(svgp, Xv, yv)
        [loss_rep] = _replace_graph([loss_sym], model, shared, extra_vars, extras)
        eval_fn = pytensor.function([Xv, yv], loss_rep)

    rng = np.random.default_rng(seed)
    losses, eval_curve = [], []
    params = [*shared.values(), *extras]
    best_val, best_step, best_state = np.inf, None, None
    for i in range(n_steps):
        if batch_size is None or batch_size >= len(y):
            xb, yb = X, y
        else:
            idx = rng.choice(len(y), batch_size, replace=False)
            xb, yb = X[idx], y[idx]
        losses.append(float(step(xb, yb)))
        if eval_fn is not None and (i % eval_every == 0 or i == n_steps - 1):
            ev = float(eval_fn(Xe, ye))
            eval_curve.append((i, ev))
            if keep_best and ev < best_val:
                best_val, best_step = ev, i
                best_state = [p.get_value() for p in params]
        if print_every and (i % print_every == 0 or i == n_steps - 1):
            print(f"step {i:4d}  loss {losses[-1]:,.1f}")
    if eval_fn is not None and keep_best and best_state is not None:
        for p, v in zip(params, best_state):
            p.set_value(v)
    return TrainResult(
        losses=losses,
        shared=shared,
        extras=extras,
        extra_vars=extra_vars,
        eval_curve=eval_curve or None,
        best_step=best_step,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def latent_to_rate(fmean, fvar):
    """Posterior-mean rate ``E[exp(f)] = exp(fm + fv/2)`` from latent moments."""
    return np.exp(fmean + 0.5 * fvar)


def compile_rate_fn(svgp, model, result, expo_col=EXPO_COL):
    """Compile the latent predictor and wrap it as ``rate_at(Xq) -> (rate, fm, fv)``.

    ``rate_at`` zeroes the exposure column, so it returns the offset-free break rate
    per pipe-km-year rather than an expected count.
    """
    X_new = pt.matrix("X_new")
    predict_latent = pg.optim.compile_predict(
        svgp,
        X_new,
        model,
        result.shared,
        extra_vars=result.extra_vars,
        shared_extras=result.extras,
        incl_lik=False,
    )

    def rate_at(Xq):
        Xq = np.asarray(Xq, float).copy()
        Xq[:, expo_col] = 0.0
        fm, fv = predict_latent(Xq)
        return latent_to_rate(fm, fv), fm, fv

    return rate_at


def compile_lik_predict(likelihood):
    """Compile ``likelihood.predict_mean_and_var`` into ``(mu, var) -> (mean, var)``.

    ptgp evaluates this by Gauss-Hermite quadrature over the latent, so it holds for
    any likelihood. Passing ``var = 0`` collapses it to the plug-in prediction that
    ignores latent uncertainty.
    """
    mu, va = pt.vector("mu"), pt.vector("va")
    emean, evar = likelihood.predict_mean_and_var(mu, va)
    return pytensor.function([mu, va], [emean, evar])


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def replacement_cost(diam_mm, length_km, base=150.0, per_mm=0.7):
    """Replacement cost in dollars: ``(base + per_mm * diameter) $/m`` times length."""
    return (base + per_mm * np.asarray(diam_mm, float)) * np.asarray(length_km, float) * 1000.0


def replace_decision(expected_breaks, replace_cost, c_break=C_BREAK):
    """Replace iff the expected cost of deferring exceeds the replacement cost.

    This is the argmin over the two actions of the expected loss (R&W eq. 2.33):
    ``loss(defer) = c_break * E[breaks over the horizon]`` vs ``loss(replace)``.
    """
    return c_break * np.asarray(expected_breaks) > np.asarray(replace_cost)


def network_total_predictive(lik_predict, fmean, fvar, length_km, n_draws=4000, seed=7):
    """Gaussian predictive draws of the annual total network break count.

    Each main's annual count has a predictive mean and variance from the likelihood
    (integrating latent uncertainty and observation noise); the network total is
    their sum, approximately Gaussian by the CLT across many near-independent mains.

    Returns ``(totals, mu_tot, sd_tot)``.
    """
    lat_1 = fmean + np.log(length_km)  # one year of exposure per main
    EN_i, VN_i = lik_predict(lat_1, fvar)
    mu_tot, sd_tot = float(EN_i.sum()), float(np.sqrt(VN_i.sum()))
    rng = np.random.default_rng(seed)
    return rng.normal(mu_tot, sd_tot, size=n_draws), mu_tot, sd_tot


class NewsvendorResult(NamedTuple):
    """Optimal budget, the search grid, expected costs, and the critical fractile."""

    b_star: int
    b_grid: np.ndarray
    expected_cost: np.ndarray
    q: float


def newsvendor_budget(totals, c_surge=C_SURGE, c_idle=C_IDLE):
    """Minimize the asymmetric provisioning loss over the budget by direct search.

    The analytic optimum is the critical fractile ``q = c_surge / (c_surge + c_idle)``
    of the predictive distribution; the returned ``q`` allows that check.
    """
    totals = np.asarray(totals, float)
    b_grid = np.arange(int(totals.min()), int(totals.max()) + 1)
    expected_cost = np.array(
        [
            (c_surge * np.maximum(totals - b, 0.0) + c_idle * np.maximum(b - totals, 0.0)).mean()
            for b in b_grid
        ]
    )
    b_star = int(b_grid[np.argmin(expected_cost)])
    q = c_surge / (c_surge + c_idle)
    return NewsvendorResult(b_star=b_star, b_grid=b_grid, expected_cost=expected_cost, q=q)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def roc_auc(y, score):
    """Rank-based ROC AUC (probability a positive outranks a negative)."""
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    p = y.sum()
    n = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def pr_curve(y, score):
    """Precision-recall curve; returns ``(precision, recall, ap, best_f1)``."""
    o = np.argsort(-score)
    yy = y[o]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    ap = float(np.sum(prec * np.diff(np.concatenate([[0], rec]))))
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, ap, float(f1.max())


def evalm(y, prob):
    """Ranking metrics for a binary target: ``{"F1", "AP", "ROC_AUC"}``."""
    _, _, ap, best_f1 = pr_curve(y, prob)
    return {"F1": best_f1, "AP": ap, "ROC_AUC": float(roc_auc(y, prob))}


def precision_at_k(y, score, k):
    """Fraction of positives among the top ``k`` ranked by ``score``."""
    return float(y[np.argsort(-score)[:k]].mean())


def rate_to_prob(rate, expo):
    """Probability of at least one break: ``1 - exp(-rate * exposure)``."""
    return 1.0 - np.exp(-rate * expo)


def latent_draws(fmean, fvar, n_draws=1000, seed=0):
    """Sample latent log-rate draws ``fm + sqrt(fv) * z`` with shape ``(S, N)``."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_draws, len(fmean)))
    return fmean[None, :] + np.sqrt(fvar)[None, :] * z


def elpd_poisson(counts, log_rate_draws, expo):
    """Expected log predictive density under a Poisson with exposure offset.

    ``log_rate_draws`` has shape ``(S, N)`` of offset-free log rates; the per-point
    predictive density is the draw average, so VI models (moment draws), MCMC models
    (posterior draws), and a single point estimate (S = 1, plug-in) all score on the
    same scale.
    """
    from scipy.special import gammaln, logsumexp

    counts = np.asarray(counts, float)
    log_lam = np.atleast_2d(log_rate_draws) + np.log(expo)[None, :]
    ll = counts[None, :] * log_lam - np.exp(log_lam) - gammaln(counts + 1)[None, :]
    return float(np.sum(logsumexp(ll, axis=0) - np.log(ll.shape[0])))


def elpd_row_poisson(counts, log_rate_draws, expo):
    """Per-row log predictive density under a Poisson with exposure offset.

    Same mixture-over-draws construction as ``elpd_poisson`` but returning the
    per-row vector, for paired model comparisons.
    """
    from scipy.special import gammaln, logsumexp

    counts = np.asarray(counts, float)
    log_lam = np.atleast_2d(log_rate_draws) + np.log(expo)[None, :]
    ll = counts[None, :] * log_lam - np.exp(log_lam) - gammaln(counts + 1)[None, :]
    return logsumexp(ll, axis=0) - np.log(ll.shape[0])


def elpd_diff_se(row_elpd_a, row_elpd_b):
    """Paired ELPD difference (A minus B) with its standard error.

    Both inputs are per-row log predictive densities on IDENTICAL rows (from
    ``elpd_row_poisson``). Pairing removes shared row difficulty, so the SE
    reflects only where the models disagree; ``sqrt(n) * sd(diffs)`` is the
    Vehtari-style uncertainty for the summed difference. Rough guide: |diff|
    above two SEs is a real difference, inside it is noise.
    """
    d = np.asarray(row_elpd_a) - np.asarray(row_elpd_b)
    return float(d.sum()), float(np.sqrt(len(d)) * d.std())


def fit_gp_prob(Xtr, counts_tr, expo_tr, Xte, expo_te, steps=300, seed=0, max_inducing=128):
    """Small SVGP rate model returning P(at least one break) on the test rows.

    Follows the primary notebook's exposure convention: log exposure is appended as
    the last X column and enters through ``ExposureOffset``; prediction zeroes it and
    converts the rate via ``rate_to_prob``. The kernel is a plain Matern52 ARD over
    the given features.
    """
    Xtr = np.asarray(Xtr, float)
    nf = Xtr.shape[1]
    ntr = len(Xtr)
    Xtr_full = np.column_stack([Xtr, np.log(np.asarray(expo_tr, float))])
    Xte_full = np.column_stack([np.asarray(Xte, float), np.zeros(len(Xte))])

    m_loc = int(min(max_inducing, ntr // 4))
    rng = np.random.default_rng(seed)
    Z = Xtr_full[rng.choice(ntr, m_loc, replace=False)].copy()
    vp = pg.gp.init_variational_params(m_loc)
    log_rate0 = float(np.log(np.sum(counts_tr) / np.sum(expo_tr)))

    with pm.Model() as mdl:
        c = pm.Normal("c", mu=log_rate0, sigma=1.0)
        ls = pm.InverseGamma("ls", alpha=3.0, beta=3.0, shape=nf)
        eta = pm.HalfNormal("eta", sigma=1.0)
        gp = pg.gp.SVGP(
            kernel=eta**2
            * pg.kernels.Matern52(input_dim=nf + 1, ls=ls, active_dims=list(range(nf))),
            likelihood=pg.likelihoods.Poisson(),
            mean=ExposureOffset(c, nf),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            variational_params=vp,
        )
    Xv, yv = pt.matrix("Xtr"), pt.vector("ytr")
    step, sp, se = pg.optim.compile_training_step(
        lambda g, X, y: pg.objectives.elbo(g, X, y, n_data=ntr).elbo,
        gp,
        Xv,
        yv,
        model=mdl,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    for _ in range(steps):
        step(Xtr_full, np.asarray(counts_tr, float))
    Xnv = pt.matrix("Xte")
    pred = pg.optim.compile_predict(
        gp, Xnv, mdl, sp, extra_vars=vp.extra_vars, shared_extras=se, incl_lik=False
    )
    fm, fv = pred(Xte_full)
    return rate_to_prob(latent_to_rate(fm, fv), np.asarray(expo_te, float))


def curve(score, replace_cost, n_post):
    """Spend-vs-captured curve: replace in score order, accumulate cost and breaks.

    Returns ``(cumulative_spend_in_$M, cumulative_holdout_breaks_captured)``.
    """
    order = np.argsort(-score)
    return np.cumsum(replace_cost[order]) / 1e6, np.cumsum(n_post[order])


def dollar_backtest(policies, replace_cost, n_post, budget=5.0):
    """Evaluate ranking policies by holdout breaks captured per capital dollar.

    Parameters
    ----------
    policies : dict
        ``{name: score}`` where higher score means replace sooner.
    replace_cost : numpy.ndarray
        Per-main replacement cost in dollars.
    n_post : numpy.ndarray
        Holdout break counts per main.
    budget : float
        Capital budget in $M at which to interpolate breaks captured.

    Returns
    -------
    dict
        ``{name: (spend_curve, captured_curve, captured_at_budget)}``.
    """
    out = {}
    for name, score in policies.items():
        cap, capt = curve(np.asarray(score, float), replace_cost, n_post)
        out[name] = (cap, capt, float(np.interp(budget, cap, capt)))
    return out


def fit_hgb_rate(Xtr, counts_tr, expo_tr, Xte, categorical_features=None, params=None):
    """Poisson gradient-boosting benchmark; returns predicted rates on ``Xte``.

    Fits sklearn's ``HistGradientBoostingRegressor(loss="poisson")`` on the observed
    rate ``counts / exposure`` with ``sample_weight=exposure``, which is equivalent
    to a log-exposure offset for the Poisson deviance. Multiply the returned rate by
    a holdout exposure to get expected counts. Categorical columns can be passed
    natively via ``categorical_features``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    kwargs = dict(loss="poisson", random_state=0)
    if params:
        kwargs.update(params)
    est = HistGradientBoostingRegressor(categorical_features=categorical_features, **kwargs)
    expo_tr = np.asarray(expo_tr, float)
    est.fit(Xtr, np.asarray(counts_tr, float) / expo_tr, sample_weight=expo_tr)
    return est.predict(Xte)


def fit_panel_gp(
    train,
    test,
    year=False,
    hist_mean=False,
    hist_kernel=False,
    M=512,
    n_steps=500,
    batch_size=1024,
    seed=0,
    eval_size=None,
    eval_every=50,
):
    """Fit an SVGP on one fold's panel rows; offset-free ``(fmean, fvar)`` on the test slice.

    ``year=False, hist_mean=False`` is Model 1; year, history mean, and history
    kernel all on is Model 2. Model 3 (the frailty) extends this inline in the
    notebook using the same builders.
    """
    Xtr, stats = panel_design(train)
    Xte, _ = panel_design(test, stats)
    ytr = train["y"].to_numpy(float)
    n_mat = int(Xtr[:, MAT_COL].max()) + 1
    n_zone = int(Xtr[:, ZONE_COL].max()) + 1
    cont_dims = [*CONT_DIMS, YEAR_COL] if year else list(CONT_DIMS)
    if hist_kernel:
        cont_dims = [*cont_dims, HIST_COL]
    Z0, vp, _ = init_inducing(Xtr, M=M, cont_dims=cont_dims, rng=seed)
    model, svgp, Z_var = build_svgp_model(
        Xtr,
        ytr,
        n_mat,
        n_zone,
        Z0,
        vp,
        year=year,
        hist_mean=hist_mean,
        hist_kernel=hist_kernel,
        seed=seed,
    )
    res = train_svgp(
        model,
        svgp,
        Z_var,
        Z0,
        vp,
        Xtr,
        ytr,
        batch_size=batch_size,
        n_steps=n_steps,
        seed=seed,
        print_every=None,
        eval_size=eval_size,
        eval_every=eval_every,
    )
    rate_fn = compile_rate_fn(svgp, model, res)
    _, fm, fv = rate_fn(Xte)
    return fm, fv


def r2d2_log_prior(log_v, n_comp, sigma2, a=1.0, b=99.0):
    """Log density of the R2D2 budget prior in log-variance coordinates.

    ``R2 = W / (W + sigma2) ~ Beta(a, b)`` on the total budget ``W = sum(exp(log_v))``,
    with a uniform Dirichlet split over the components and the Jacobian into
    per-component log variances, the coordinates that keep MAP well conditioned
    (the natural (R2, stick-breaking) coordinates leave the allocation stuck).
    """
    v = pt.exp(log_v)
    W = pt.sum(v)
    r2 = W / (W + sigma2)
    return (
        (a - 1.0) * pt.log(r2)
        + (b - 1.0) * pt.log1p(-r2)
        + np.log(sigma2)
        - 2.0 * pt.log(W + sigma2)
        + gammaln(float(n_comp))
        - (n_comp - 1) * pt.log(W)
        + pt.sum(log_v)
    )


def fit_panel_gp_r2d2(
    train,
    test,
    M=1024,
    n_steps=700,
    batch_size=1024,
    seed=0,
    eval_size=None,
    eval_every=50,
    r2_prior=(1.0, 99.0),
):
    """Model 2 on one fold: M1's kernel plus linear history and year mean terms
    under a shared R2D2 variance budget, MAP-estimated in log-variance coordinates.

    The budget splits over three components {GP amplitude, history slab, year
    slab}; ``R2 = W / (W + sigma2) ~ Beta(*r2_prior)`` with ``sigma2 = 1/mean(y)``
    the Poisson pseudo-variance. Initialized at the prior-mean budget, split
    equally. Returns a dict with the test-slice ``(fm, fv)``, fitted component
    variances and allocation, R2, and the two coefficients.
    """
    Xtr, stats = panel_design(train)
    Xte, _ = panel_design(test, stats)
    ytr = train["y"].to_numpy(float)
    N = len(ytr)
    n_mat = int(Xtr[:, MAT_COL].max()) + 1
    n_zone = int(Xtr[:, ZONE_COL].max()) + 1
    input_dim = Xtr.shape[1]
    sigma2 = 1.0 / max(ytr.mean(), 1e-6)
    log_rate0 = float(np.log(ytr.sum() / np.exp(Xtr[:, EXPO_COL]).sum()))
    freq_mat = np.bincount(Xtr[:, MAT_COL].astype(int), minlength=n_mat) / N
    freq_zone = np.bincount(Xtr[:, ZONE_COL].astype(int), minlength=n_zone) / N
    Z0, vp, _ = init_inducing(Xtr, M=M, rng=seed)
    wrng = np.random.default_rng(seed)

    log_v = pt.vector("log_v")  # {log eta^2, log v_hist, log v_year}
    c = pt.scalar("c")
    betas = pt.vector("betas")  # [beta_hist, beta_year]
    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=3.0, beta=3.0, shape=4)
        W_mat = pm.Normal(
            "W_mat", 0.0, 1.0, shape=(n_mat, 2), initval=0.1 * wrng.standard_normal((n_mat, 2))
        )
        kappa_mat = pm.HalfNormal("kappa_mat", sigma=1.0, shape=n_mat)
        W_zone = pm.Normal(
            "W_zone", 0.0, 1.0, shape=(n_zone, 2), initval=0.1 * wrng.standard_normal((n_zone, 2))
        )
        kappa_zone = pm.HalfNormal("kappa_zone", sigma=1.0, shape=n_zone)
        kernel = (
            pt.exp(log_v[0])
            * pg.kernels.Matern52(input_dim=input_dim, ls=ls, active_dims=list(CONT_DIMS))
            * NormalizedLowRankCategorical(
                input_dim=input_dim,
                num_levels=n_mat,
                W=W_mat,
                kappa=kappa_mat,
                freqs=freq_mat,
                active_dims=[MAT_COL],
            )
            * NormalizedLowRankCategorical(
                input_dim=input_dim,
                num_levels=n_zone,
                W=W_zone,
                kappa=kappa_zone,
                freqs=freq_zone,
                active_dims=[ZONE_COL],
            )
        )
        Z_var = pt.matrix("Z")
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Poisson(),
            mean=LinearMainEffects(c, EXPO_COL, betas, [HIST_COL, YEAR_COL]),
            inducing_variable=pg.inducing.Points(Z_var, Z_init=Z0),
            variational_params=vp,
        )

    a, b = r2_prior

    def objective(gp, Xb, yb):
        base = pg.objectives.elbo(gp, Xb, yb, n_data=N).elbo
        lp = r2d2_log_prior(log_v, 3, sigma2, a, b)
        lp = lp - 0.5 * (c - log_rate0) ** 2
        lp = lp - 0.5 * betas[0] ** 2 / pt.exp(log_v[1]) - 0.5 * log_v[1]
        lp = lp - 0.5 * betas[1] ** 2 / pt.exp(log_v[2]) - 0.5 * log_v[2]
        return base + lp

    r2_mean = a / (a + b)
    W0 = sigma2 * r2_mean / (1.0 - r2_mean)
    res = train_svgp(
        model,
        svgp,
        Z_var,
        Z0,
        vp,
        Xtr,
        ytr,
        batch_size=batch_size,
        n_steps=n_steps,
        seed=seed,
        print_every=None,
        eval_size=eval_size,
        eval_every=eval_every,
        objective=objective,
        extra_vars=[*vp.extra_vars, Z_var, log_v, c, betas],
        extra_init=[
            *vp.extra_init,
            Z0,
            np.log(np.full(3, W0 / 3)),
            np.array(log_rate0),
            np.zeros(2),
        ],
    )
    rate_fn = compile_rate_fn(svgp, model, res)
    _, fm, fv = rate_fn(Xte)
    v_hat = np.exp(res.extras[-3].get_value())
    return {
        "fm": fm,
        "fv": fv,
        "v": v_hat.tolist(),
        "phi": (v_hat / v_hat.sum()).tolist(),
        "r2": float(v_hat.sum() / (v_hat.sum() + sigma2)),
        "c": float(res.extras[-2].get_value()),
        "betas": res.extras[-1].get_value().tolist(),
        "best_step": res.best_step,
    }


def hgb_random_grid(n=80, seed=7):
    """Random-search configurations for the HGB benchmark hyperparameters."""
    rng = np.random.default_rng(seed)
    return [
        {
            "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.3)))),
            "max_iter": int(rng.choice([100, 200, 400])),
            "max_leaf_nodes": int(rng.choice([15, 31, 63, 127])),
            "max_depth": [None, 4, 8][rng.integers(3)],
            "min_samples_leaf": int(rng.choice([20, 50, 100, 200, 500, 1000])),
            "l2_regularization": float(np.exp(rng.uniform(np.log(1e-2), np.log(10.0)))),
            "max_bins": int(rng.choice([63, 128, 255])),
            "max_features": float(rng.choice([0.6, 0.8, 1.0])),
        }
        for _ in range(n)
    ]


HGB_COLS = ["age", "log_size", "lon", "lat", "year", "mat_code", "zone_code", "hist"]


def fit_hgb_panel(train, test, t, grid=None, cols=None):
    """Tuned gradient-boosting benchmark on panel rows; test-slice rates.

    Untuned HGB explodes on the panel's spiky per-row rates (one break on a short
    pipe is an enormous rate), so hyperparameters are selected by random search
    scored on the last training year, then the winner is refit on all training
    rows. Returns ``(rate, best_params)``.
    """
    grid = grid or hgb_random_grid()
    cols = cols or HGB_COLS
    tr = train[train["year"] < t - 1]
    val = train[train["year"] == t - 1]
    cat = [cols.index("mat_code"), cols.index("zone_code")]
    best, best_ll = None, -np.inf
    for params in grid:
        rate = fit_hgb_rate(
            tr[cols].to_numpy(float),
            tr["y"].to_numpy(float),
            tr["length_km"].to_numpy(float),
            val[cols].to_numpy(float),
            categorical_features=cat,
            params=params,
        )
        ll = elpd_poisson(
            val["y"].to_numpy(float),
            np.log(np.clip(rate, 1e-9, None))[None, :],
            val["length_km"].to_numpy(float),
        )
        if ll > best_ll:
            best, best_ll = params, ll
    rate = fit_hgb_rate(
        train[cols].to_numpy(float),
        train["y"].to_numpy(float),
        train["length_km"].to_numpy(float),
        test[cols].to_numpy(float),
        categorical_features=cat,
        params=best,
    )
    return rate, best


def fit_glm_panel(train, test):
    """Poisson GLM reference (the classical two-line model); test-slice rates.

    Standardized continuous features plus one-hot material/zone, exposure via
    ``sample_weight`` on the rate, light L2.
    """
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cont_cols = ["age", "log_size", "lon", "lat", "year", "hist"]
    sc = StandardScaler().fit(train[cont_cols])
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train[["mat_code", "zone_code"]]
    )

    def feats(df):
        return np.column_stack(
            [sc.transform(df[cont_cols]), oh.transform(df[["mat_code", "zone_code"]])]
        )

    glm = PoissonRegressor(alpha=1e-4, max_iter=500)
    glm.fit(
        feats(train),
        train["y"].to_numpy(float) / train["length_km"].to_numpy(float),
        sample_weight=train["length_km"].to_numpy(float),
    )
    return glm.predict(feats(test))


def poisson_moments(fmean, fvar, expo):
    """Exact predictive mean/variance of a Poisson-lognormal count."""
    EN = expo * np.exp(fmean + 0.5 * fvar)
    VN = EN + expo**2 * np.exp(2 * fmean + fvar) * (np.exp(fvar) - 1.0)
    return EN, VN


# ---------------------------------------------------------------------------
# Overdispersion
# ---------------------------------------------------------------------------


def pearson_dispersion(y, mu):
    """Pearson dispersion ``phi = mean((y - mu)^2 / mu)``; 1 under a true Poisson."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(np.mean((y - mu) ** 2 / mu))


def ppc_count_stats(y_rep, y_obs):
    """Posterior-predictive check table for count data.

    Compares the observed variance/mean ratio, zero fraction, and maximum against
    their distribution over replicated datasets ``y_rep`` (one replicate per row).
    """

    def _row(v):
        v = np.asarray(v, float)
        return {"var/mean": v.var() / v.mean(), "frac zero": (v == 0).mean(), "max": v.max()}

    rep = pd.DataFrame([_row(r) for r in np.atleast_2d(y_rep)])
    return pd.DataFrame(
        {
            "observed": pd.Series(_row(y_obs)),
            "replicated mean": rep.mean(),
            "replicated 2.5%": rep.quantile(0.025),
            "replicated 97.5%": rep.quantile(0.975),
        }
    ).T


# ---------------------------------------------------------------------------
# Tables and plots
# ---------------------------------------------------------------------------

_TBL_STYLES = [
    {
        "selector": "caption",
        "props": [
            ("font-weight", "bold"),
            ("font-size", "1.05em"),
            ("text-align", "left"),
            ("padding", "0 0 6px 0"),
        ],
    },
    {"selector": "th", "props": [("text-align", "left")]},
    {"selector": "td", "props": [("text-align", "right"), ("padding", "2px 14px")]},
]


def _counts_table(series, axis_name, caption):
    """Styled value-counts HTML table with counts, shares, and bars."""
    t = series.value_counts().rename_axis(axis_name).reset_index(name="mains")
    t["share"] = t["mains"] / t["mains"].sum()
    return (
        t.style.hide(axis="index")
        .format({"mains": "{:,}", "share": "{:.1%}"})
        .bar(subset=["mains"], color="#cfe3f3")
        .set_caption(caption)
        .set_table_styles(_TBL_STYLES)
        .to_html()
    )


def _display_side_by_side(*htmls):
    """Render HTML tables in one flex row (notebook display)."""
    from IPython.display import display_html

    wrap = '<div style="padding-right:72px">{}</div>'
    display_html(
        '<div style="display:flex; align-items:flex-start">'
        + "".join(wrap.format(h) for h in htmls)
        + "</div>",
        raw=True,
    )


def mat_value_counts_table(m):
    """Display material mix before and after consolidation, side by side."""
    raw = m["MATERIAL"]
    con = consolidate_categories(raw, MAT_MAP)
    _display_side_by_side(
        _counts_table(raw, "material", f"Material (raw, {raw.nunique()} categories)"),
        _counts_table(con, "material", f"Material (consolidated, {con.nunique()} categories)"),
    )


def pz_value_counts_table(m, pz_source):
    """Display pressure-zone mix before and after cleaning, side by side."""
    modeled = consolidate_categories(m["PRESSURE_ZONE"], PZ_MAP)
    _display_side_by_side(
        _counts_table(
            pz_source, "pressure_zone", f"Pressure zone (raw, {pz_source.nunique()} zones)"
        ),
        _counts_table(
            modeled, "pressure_zone", f"Pressure zone (modeled, {modeled.nunique()} zones)"
        ),
    )


def cov_matrix(W, kappa):
    """Learned level covariance ``B = W W^T + diag(kappa)``."""
    return W @ W.T + np.diag(kappa)


def plot_cov(ax, B, labels, title, vmax=4.0):
    """Annotated heatmap of a learned category covariance matrix."""
    im = ax.imshow(B, vmin=0, vmax=vmax, cmap="cool")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{B[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(B[i, j]) > 0.6 * vmax else "black",
            )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def line_segments(geoms):
    """Flatten LineString/MultiLineString geometries into LineCollection segments."""
    segs = []
    for g in geoms:
        if g:
            segs += [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
    return segs


def plot_network(segs, mb):
    """Map of the mains network with recorded break locations."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(segs, colors="0.5", linewidths=0.5, zorder=1))
    ax.scatter(
        mb["lon"],
        mb["lat"],
        s=9,
        c="tab:blue",
        alpha=0.55,
        zorder=2,
        label=f"main break (n={len(mb)})",
    )
    ax.set_aspect(MAP_ASPECT)
    ax.autoscale()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Kitchener water mains and recorded breaks")
    ax.legend(loc="upper left", markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_feature_hist(m):
    """Histograms of the continuous features and the (log-scaled) break counts."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].hist(m["age"], bins=40, color="tab:blue", alpha=0.85)
    ax[0].set_xlabel("pipe age (years)")
    ax[0].set_ylabel("mains")
    ax[0].set_title("Pipe age")
    ax[1].hist(np.log(m["PIPE_SIZE"].to_numpy(float)), bins=40, color="tab:blue", alpha=0.85)
    ax[1].set_xlabel("log diameter (log mm)")
    ax[1].set_title("Diameter (log scale)")
    ax[2].hist(m["exposure"], bins=40, color="tab:blue", alpha=0.85)
    ax[2].set_xlabel("exposure (pipe-km-years)")
    ax[2].set_title("Exposure")
    nb_max = int(m["n_breaks"].max())
    ax[3].hist(m["n_breaks"], bins=range(0, nb_max + 2), color="tab:blue", alpha=0.85)
    ax[3].set_xlabel("recorded breaks")
    ax[3].set_title("Break count")
    ax[3].set_yscale("log")  # heavily zero-inflated: most mains never break
    plt.tight_layout()
    plt.show()


def plot_risk_map(m, mains_geoms, rate):
    """Network map coloured by predicted break rate (log colour scale)."""
    geoms_m = [mains_geoms[i] for i in m.index]
    sg, sr = [], []
    for g, r in zip(geoms_m, rate):
        if not g:
            continue
        for p in [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]:
            sg.append(p)
            sr.append(r)
    sr = np.asarray(sr)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    norm = mpl.colors.LogNorm(vmin=max(np.quantile(sr, 0.02), 1e-3), vmax=np.quantile(sr, 0.99))
    lc = LineCollection(sg, array=sr, cmap="inferno", norm=norm, linewidths=1.1)
    ax.add_collection(lc)
    ax.set_aspect(MAP_ASPECT)
    ax.autoscale()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Predicted water main break risk")
    plt.colorbar(lc, ax=ax, label="break rate (per km-year)", shrink=0.7)
    plt.tight_layout()
    plt.show()


def plot_decision_map(m, segs, repl_dt, repl_naive, rate_plug_km, sd_lat):
    """Decision scatter (rate vs uncertainty) and map of flagged mains."""
    cat = np.where(repl_naive, 1, np.where(repl_dt, 2, 0))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    groups = {
        0: ("0.8", "defer"),
        1: ("tab:red", "replace (both)"),
        2: ("tab:blue", "replace (uncertainty-driven)"),
    }
    for k, (col, lab) in groups.items():
        sel = cat == k
        ax[0].scatter(
            rate_plug_km[sel],
            sd_lat[sel],
            s=9,
            c=col,
            alpha=0.5,
            label=f"{lab} (n={int(sel.sum())})",
        )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("plug-in break rate (per km-year)")
    ax[0].set_ylabel("GP latent uncertainty (sd)")
    ax[0].set_title("Integrating the predictive changes the action")
    ax[0].legend(loc="upper left", fontsize=8)

    ax[1].add_collection(LineCollection(segs, colors="0.88", linewidths=0.4, zorder=1))
    fl = m.assign(cat=cat)
    for k, (col, lab) in {1: ("tab:red", "both"), 2: ("tab:blue", "uncertainty-driven")}.items():
        s = fl[fl["cat"] == k]
        ax[1].scatter(s["lon"], s["lat"], s=22, c=col, label=f"{lab} (n={len(s)})", zorder=3)
    ax[1].set_aspect(MAP_ASPECT)
    ax[1].autoscale()
    ax[1].set_xlabel("longitude")
    ax[1].set_ylabel("latitude")
    ax[1].set_title("Mains flagged for replacement")
    ax[1].legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_newsvendor(totals, nv):
    """Predictive total-breaks histogram and expected cost vs provisioned budget.

    Also prints the annual saving of planning at the optimum instead of the mean.
    """
    mean_n = float(np.mean(totals))
    at_mean = nv.expected_cost[np.argmin(np.abs(nv.b_grid - round(mean_n)))]
    saving = at_mean - nv.expected_cost.min()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(totals, bins=40, color="tab:gray", alpha=0.8)
    ax[0].axvline(mean_n, color="tab:blue", lw=2, label=f"mean ({mean_n:.0f})")
    ax[0].axvline(nv.b_star, color="tab:red", lw=2, label=f"optimal budget ({nv.b_star})")
    ax[0].set_xlabel("annual network breaks")
    ax[0].set_ylabel("predictive draws")
    ax[0].set_title("Predictive distribution of annual breaks")
    ax[0].legend()
    ax[1].plot(nv.b_grid, nv.expected_cost / 1e3)
    ax[1].axvline(nv.b_star, color="tab:red", ls=":", label=f"B* = {nv.b_star}")
    ax[1].axvline(mean_n, color="tab:blue", ls=":", label=f"mean = {mean_n:.0f}")
    ax[1].set_xlabel("provisioned budget (breaks)")
    ax[1].set_ylabel("expected cost ($k/yr)")
    ax[1].set_title("Optimum is above the mean")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    print(f"planning to the optimum instead of the mean saves ${saving:,.0f}/yr")


def plot_flagged_map(
    base, segs, first_flagged, title="Mains flagged for replacement, by plan year"
):
    """Map of flagged mains, coloured by the first planning year that flagged them.

    Parameters
    ----------
    base : pandas.DataFrame
        Static frame from ``static_frame`` (positional pipe index).
    segs : list
        Network line segments for the grey background.
    first_flagged : dict
        ``{pipe_idx: year}`` of each flagged pipe's first plan year.
    """
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.add_collection(LineCollection(segs, colors="0.88", linewidths=0.4, zorder=1))
    idx = np.array(sorted(first_flagged), dtype=int)
    years = np.array([first_flagged[i] for i in idx], dtype=float)
    sc = ax.scatter(
        base["lon"].to_numpy()[idx],
        base["lat"].to_numpy()[idx],
        s=16,
        c=years,
        cmap="viridis",
        zorder=3,
    )
    ax.set_aspect(MAP_ASPECT)
    ax.autoscale()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{title} (n={len(idx)})")
    plt.colorbar(sc, ax=ax, label="first plan year", shrink=0.7)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(prec, rec, ap, base_rate, title="Precision-recall, temporal holdout"):
    """Precision-recall curve against the base-rate floor."""
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(rec, prec, label=f"GP rate model (AP={ap:.2f})")
    plt.axhline(base_rate, color="0.6", ls="--", label=f"base rate ({base_rate:.2f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_backtest(results, budget=5.0, styles=None, title="Dollar backtest"):
    """Spend-vs-captured curves for each policy from ``dollar_backtest``."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, (cap, capt, _) in results.items():
        col, ls = (styles or {}).get(name, (None, "-"))
        ax.plot(cap, capt, label=name, color=col, ls=ls)
    ax.axvline(budget, color="0.7", lw=0.8)
    ax.set_xlim(0, 15)
    ax.set_xlabel("replacement capital spent ($M)")
    ax.set_ylabel("actual holdout breaks captured (avoided)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest notebooks/watermains.py)
# ---------------------------------------------------------------------------


def test_exposure():
    """Years before the record start must not count toward exposure."""
    np.testing.assert_allclose(exposure(2.0, 2000.0), 2.0 * (AS_OF - 2000.0))
    np.testing.assert_allclose(exposure(2.0, 1950.0), 2.0 * (AS_OF - RECORD_START))


def test_frequency_codes():
    """Most common level gets code 0; order follows frequency."""
    codes, levels = frequency_codes(pd.Series(["a", "b", "a", "c", "a", "b"]))
    assert levels == ["a", "b", "c"]
    np.testing.assert_array_equal(codes, [0.0, 1.0, 0.0, 2.0, 0.0, 1.0])


def test_cov_matrix():
    """B = W W^T + diag(kappa) is symmetric with diagonal at least kappa."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 2))
    kappa = rng.uniform(0.1, 1.0, 5)
    B = cov_matrix(W, kappa)
    np.testing.assert_allclose(B, B.T)
    assert np.all(np.diag(B) >= kappa)


def test_replace_decision():
    """Replace exactly when expected break cost exceeds replacement cost."""
    decision = replace_decision(np.array([1.0, 0.1]), np.array([20000.0, 20000.0]), c_break=25000.0)
    np.testing.assert_array_equal(decision, [True, False])


def test_newsvendor_budget():
    """Grid-search optimum matches the analytic critical fractile of a Gaussian."""
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    totals = rng.normal(100.0, 10.0, size=200_000)
    nv = newsvendor_budget(totals, c_surge=40000.0, c_idle=8000.0)
    assert abs(nv.q - 40000.0 / 48000.0) < 1e-12
    b_analytic = 100.0 + 10.0 * norm.ppf(nv.q)
    assert abs(nv.b_star - b_analytic) <= 1.0


def test_roc_auc():
    """Matches the hand-computed AUC and hits 1.0 under perfect separation."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    np.testing.assert_allclose(roc_auc(y, s), 0.75)
    np.testing.assert_allclose(roc_auc(y, y.astype(float)), 1.0)


def test_evalm():
    """Perfect ranking scores 1.0 on all three metrics."""
    y = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    res = evalm(y, prob)
    np.testing.assert_allclose([res["F1"], res["AP"], res["ROC_AUC"]], 1.0)


def test_pearson_dispersion():
    """Dispersion is about 1 for Poisson data and clearly above 1 when inflated."""
    rng = np.random.default_rng(0)
    mu = np.full(20_000, 5.0)
    assert abs(pearson_dispersion(rng.poisson(5.0, 20_000), mu) - 1.0) < 0.05
    lam = 5.0 * rng.gamma(2.0, 0.5, 20_000)  # mean 5, extra-Poisson variation
    assert pearson_dispersion(rng.poisson(lam), mu) > 1.2


def test_build_panel():
    """Panel rows respect the time windows: targets per year, history strictly before."""
    mains = pd.DataFrame(
        {
            "WATMAINID": ["A", "B"],
            "STATUS": ["ACTIVE"] * 2,
            "install_year": [1950.0, 2016.0],
            "PIPE_SIZE": [150.0, 200.0],
            "Shape__Length": [1000.0, 2000.0],
            "PRESSURE_ZONE": ["KIT 4"] * 2,
            "MATERIAL": ["CI", "DI"],
            "lat": [43.4] * 2,
            "lon": [-80.5] * 2,
        }
    )
    breaks = pd.DataFrame(
        {
            "ASSETID": ["A", "A", "A", "B"],
            "BREAK_TYPE": ["MAIN"] * 4,
            "INCIDENT_DATE": pd.to_datetime(
                ["1990-06-01", "2014-03-01", "2018-07-01", "2019-01-15"]
            ),
        }
    )
    train, test = build_panel(mains, breaks, t=2019.0, target_start=2010.0)
    # A contributes every year 2010-2018 to train; B (installed 2016) only 2017-2018
    a_tr = train[train["pipe_idx"] == 0.0]
    b_tr = train[train["pipe_idx"] == 1.0]
    assert len(a_tr) == 9 and len(b_tr) == 2
    # targets land in their own year and nowhere else
    assert a_tr.set_index("year")["y"].loc[2014.0] == 1
    assert a_tr.set_index("year")["y"].loc[2018.0] == 1
    assert a_tr["y"].sum() == 2
    # test slice is the planning year, with B's 2019 break
    assert set(test["year"]) == {2019.0} and test.set_index("pipe_idx")["y"].loc[1.0] == 1
    # history at row year s covers [RECORD_START, s): A's 2014 row sees nothing
    # (the 1990 break predates effective record coverage); the 2018 row sees 2014
    row = a_tr[a_tr["year"] == 2014.0].iloc[0]
    np.testing.assert_allclose(row["hist"], 0.0)
    row18 = a_tr[a_tr["year"] == 2018.0].iloc[0]
    np.testing.assert_allclose(row18["hist"], 1 / (1.0 * (2018 - RECORD_START)))
    # age is measured at the row's year
    assert row["age"] == 2014 - 1950


def test_elpd_diff_se():
    """Identical models differ by exactly zero; a uniformly better one is positive."""
    rng = np.random.default_rng(0)
    expo = rng.uniform(1, 5, 400)
    y = rng.poisson(0.05 * expo)
    draws_a = np.log(np.full((50, 400), 0.05)) + 0.1 * rng.standard_normal((50, 400))
    ra = elpd_row_poisson(y, draws_a, expo)
    d, se = elpd_diff_se(ra, ra)
    assert d == 0.0 and se == 0.0
    rb = ra - 0.01  # B uniformly worse
    d, se = elpd_diff_se(ra, rb)
    np.testing.assert_allclose(d, 4.0)
    assert se < 1e-8


def test_curve_monotone():
    """Backtest curves are nondecreasing and capture every break at full spend."""
    rng = np.random.default_rng(0)
    score = rng.standard_normal(50)
    cost = rng.uniform(1e4, 1e6, 50)
    n_post = rng.poisson(0.3, 50).astype(float)
    spend, captured = curve(score, cost, n_post)
    assert np.all(np.diff(spend) >= 0)
    assert np.all(np.diff(captured) >= 0)
    np.testing.assert_allclose(captured[-1], n_post.sum())
