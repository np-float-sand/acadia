"""
Linear + non-linear regression specs (pre-registration Decision 5).

Linear:
  A) GARCH-style variance regression:
       equity_var_t = a + b*hub_var_lag1 + c*control_var_t + d*control_ret_t + e
     `hub_var_lag1` is the hub's realized variance as of one equity-trading-day
     before t (see data.align_lagged_hub_to_equity); control_var/control_ret are
     CONTEMPORANEOUS (same day t) BTC (miner arm) or SPY (generator arm) terms,
     included specifically to soak up the bloc-move confound Probe B caught
     (docs/electricity-earnings-exposure-probe-results.md) before crediting the
     lagged hub variance with anything.
  B) Plain OLS of forward return on lagged hub return (secondary/informational;
     no pre-registered sign, per the pre-registration -- not gate-bearing on its
     own, reported for completeness):
       equity_ret_t = a + b*hub_ret_lag1 + c*control_ret_t + d*control_var_t + e

Non-linear:
  C) GBM regressor on the same feature set as (A) trained on the TRAIN window
     and scored out-of-sample on the TEST window (the split IS the OOS check,
     per Decision 5 -- no further split needed).
  D) Tail-only cut: top-decile |hub_ret_lag1| days vs the rest, Welch t-test on
     the difference in mean subsequent equity variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


def _clean_ols(y: pd.Series, X: pd.DataFrame):
    frame = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(frame) < 20:
        return None
    Xc = sm.add_constant(frame.drop(columns="y"))
    model = sm.OLS(frame["y"], Xc).fit()
    return model, len(frame)


def variance_regression(equity_var: pd.Series, hub_var_lag: pd.Series,
                         control_var: pd.Series, control_ret: pd.Series) -> dict:
    X = pd.DataFrame({
        "hub_var_lag": hub_var_lag,
        "control_var": control_var,
        "control_ret": control_ret,
    })
    fit = _clean_ols(equity_var, X)
    if fit is None:
        return dict(coef=np.nan, tstat=np.nan, n=0, pvalue=np.nan)
    model, n = fit
    return dict(
        coef=model.params.get("hub_var_lag", np.nan),
        tstat=model.tvalues.get("hub_var_lag", np.nan),
        pvalue=model.pvalues.get("hub_var_lag", np.nan),
        n=n,
    )


def return_regression(equity_ret: pd.Series, hub_ret_lag: pd.Series,
                       control_ret: pd.Series, control_var: pd.Series) -> dict:
    X = pd.DataFrame({
        "hub_ret_lag": hub_ret_lag,
        "control_ret": control_ret,
        "control_var": control_var,
    })
    fit = _clean_ols(equity_ret, X)
    if fit is None:
        return dict(coef=np.nan, tstat=np.nan, n=0, pvalue=np.nan)
    model, n = fit
    return dict(
        coef=model.params.get("hub_ret_lag", np.nan),
        tstat=model.tvalues.get("hub_ret_lag", np.nan),
        pvalue=model.pvalues.get("hub_ret_lag", np.nan),
        n=n,
    )


def gbm_oos(train_X: pd.DataFrame, train_y: pd.Series,
            test_X: pd.DataFrame, test_y: pd.Series, seed: int = 42) -> dict:
    tr = pd.concat([train_y.rename("y"), train_X], axis=1).dropna()
    te = pd.concat([test_y.rename("y"), test_X], axis=1).dropna()
    if len(tr) < 30 or len(te) < 10:
        return dict(oos_r2=np.nan, oos_spearman=np.nan, oos_spearman_p=np.nan,
                     n_train=len(tr), n_test=len(te))
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    model.fit(tr.drop(columns="y"), tr["y"])
    pred = model.predict(te.drop(columns="y"))
    oos_r2 = r2_score(te["y"], pred)
    rho, p = stats.spearmanr(pred, te["y"])
    return dict(oos_r2=oos_r2, oos_spearman=rho, oos_spearman_p=p,
                n_train=len(tr), n_test=len(te))


def tail_cut_test(hub_ret_lag: pd.Series, equity_var: pd.Series, quantile: float = 0.90) -> dict:
    frame = pd.concat([hub_ret_lag.rename("x"), equity_var.rename("y")], axis=1).dropna()
    if len(frame) < 30:
        return dict(mean_tail=np.nan, mean_rest=np.nan, diff=np.nan, tstat=np.nan,
                     pvalue=np.nan, n_tail=0, n_rest=0)
    thresh = frame["x"].abs().quantile(quantile)
    tail = frame.loc[frame["x"].abs() >= thresh, "y"]
    rest = frame.loc[frame["x"].abs() < thresh, "y"]
    if len(tail) < 5 or len(rest) < 5:
        return dict(mean_tail=np.nan, mean_rest=np.nan, diff=np.nan, tstat=np.nan,
                     pvalue=np.nan, n_tail=len(tail), n_rest=len(rest))
    t, p = stats.ttest_ind(tail, rest, equal_var=False)
    return dict(
        mean_tail=tail.mean(), mean_rest=rest.mean(), diff=tail.mean() - rest.mean(),
        tstat=t, pvalue=p, n_tail=len(tail), n_rest=len(rest),
    )
