"""Estimation functions used in the archived experimental report."""
import numpy as np
from scipy import stats
import statsmodels.api as sm


def interval(x):
    x = np.asarray(x, dtype=float)
    n, estimate = len(x), float(np.mean(x))
    se = float(stats.sem(x)) if n > 1 else np.nan
    if n < 2:
        return dict(n=n, estimate=estimate, se=se, ci_low=np.nan, ci_high=np.nan, p_raw=np.nan)
    margin = stats.t.ppf(.975, n - 1) * se
    p = 2 * stats.t.sf(abs(estimate / se), n - 1) if se else (1. if estimate == 0 else 0.)
    return dict(n=n, estimate=estimate, se=se, ci_low=estimate-margin, ci_high=estimate+margin, p_raw=float(p))


def welch(risk, benefit):
    risk, benefit = np.asarray(risk, float), np.asarray(benefit, float)
    nr, nb = len(risk), len(benefit)
    estimate = float(risk.mean() - benefit.mean())
    if min(nr, nb) < 2:
        return dict(n=nr+nb, estimate=estimate, se=np.nan, ci_low=np.nan, ci_high=np.nan, p_raw=np.nan)
    vr, vb = risk.var(ddof=1)/nr, benefit.var(ddof=1)/nb
    se = float(np.sqrt(vr + vb))
    df = (vr+vb)**2/(vr**2/(nr-1)+vb**2/(nb-1)) if se else np.inf
    margin = stats.t.ppf(.975, df)*se
    p = 2*stats.t.sf(abs(estimate/se), df) if se else (1. if estimate == 0 else 0.)
    return dict(n=nr+nb, estimate=estimate, se=se, ci_low=estimate-margin, ci_high=estimate+margin, p_raw=float(p))


def slope(x, y):
    """Single-feature OLS, HC3 t interval and training-mean baseline for held-out tasks."""
    fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(cov_type='HC3', use_t=True)
    ci = fit.conf_int()[1]
    predictions = []; baselines = []
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        training = sm.OLS(y.to_numpy()[keep], sm.add_constant(x.to_numpy()[keep])).fit()
        predictions.append(float(training.params[0] + training.params[1]*x.iloc[held_out]))
        baselines.append(y.to_numpy()[keep].mean())
    return dict(effect=fit.params[1], se=fit.bse[1], ci_low=ci[0], ci_high=ci[1],
                p_raw=fit.pvalues[1],
                loto_r2=1-np.sum((y-np.array(predictions))**2)/np.sum((y-np.array(baselines))**2))

