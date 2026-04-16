"""
Forecasting models for monthly spending series.

Models available:
  - Rolling average          (baseline, always available)
  - ETS / Holt's smoothing   (statsmodels, always available)
  - ARIMA(1,1,1)             (statsmodels, requires >=6 data points)
  - Ridge regression         (scikit-learn, requires >=4 data points)
  - Prophet                  (optional, requires prophet package + >=6 points)

Each model returns (forecast, lower_ci, upper_ci) as pd.Series indexed by
future month-start timestamps.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except ImportError:
    pass


def _future_index(series: pd.Series, n: int) -> pd.DatetimeIndex:
    """Return n monthly timestamps after the last date in series."""
    return pd.date_range(
        series.index[-1] + pd.DateOffset(months=1),
        periods=n,
        freq="MS",
    )


def prepare_monthly_series(
    df: pd.DataFrame,
    exclude_categories: list[str] | None = None,
    clip: bool = False,
    clip_method: str = "iqr",
) -> pd.Series:
    """Aggregate spending to monthly totals (Month-Start index)."""
    if df.empty:
        return pd.Series(dtype=float)
    filtered = df.copy()
    if exclude_categories:
        filtered = filtered[~filtered["Category"].isin(exclude_categories)]
    monthly = filtered.groupby("YearMonth")["Amount"].sum().sort_index()
    monthly.index = pd.DatetimeIndex(monthly.index)
    if clip:
        monthly = clip_outliers(monthly, method=clip_method)
    return monthly

def clip_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Cap extreme monthly spending spikes while preserving the datetime index.

    Parameters
    ----------
    series : pd.Series
        Monthly spending totals with a DatetimeIndex.
    method : "iqr" | "percentile"
        "iqr"        — upper cap = Q3 + 1.5 × IQR  (more aggressive, better for
                        short series with one dominant outlier month)
        "percentile" — upper cap = 90th percentile  (softer, needs ≥10 points
                        to be meaningful)

    Returns
    -------
    pd.Series
        Clipped series with the original DatetimeIndex intact.
    """
    if series.empty or len(series) < 3:
        return series

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
    else:  
        upper = series.quantile(0.90)

    upper = max(upper, series.median())

    clipped = series.clip(upper=upper)
    clipped.index = series.index          
    return clipped

def rolling_forecast(
    series: pd.Series,
    n_months: int = 3,
    window: int = 3,
) -> tuple:
    """3-month rolling average — naive baseline."""
    if len(series) < 1:
        return None, None, None
    w = min(window, len(series))
    level = series.rolling(w).mean().iloc[-1]
    std = series.rolling(w).std().iloc[-1]
    if pd.isna(std) or std == 0:
        std = series.std() if len(series) > 1 else level * 0.1

    idx = _future_index(series, n_months)
    fc = pd.Series([level] * n_months, index=idx)
    lo = (fc - 1.96 * std).clip(lower=0)
    hi = fc + 1.96 * std
    return fc, lo, hi


def ets_forecast(series: pd.Series, n_months: int = 3) -> tuple:
    """Holt's double exponential smoothing (trend-aware)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    if len(series) < 2:
        return rolling_forecast(series, n_months)
    try:
        if len(series) >= 6:
            model = ExponentialSmoothing(
                series, trend="add", initialization_method="estimated"
            )
        else:
            model = SimpleExpSmoothing(series, initialization_method="estimated")

        fit = model.fit(optimized=True)
        fc_raw = fit.forecast(n_months)
        idx = _future_index(series, n_months)
        fc = pd.Series(fc_raw.values, index=idx).clip(lower=0)

        std = fit.resid.std()
        lo = (fc - 1.96 * std).clip(lower=0)
        hi = fc + 1.96 * std
        return fc, lo, hi
    except Exception:
        return rolling_forecast(series, n_months)


def arima_forecast(series: pd.Series, n_months: int = 3) -> tuple:
    """ARIMA(1,1,1) — auto-regressive with differencing."""
    if len(series) < 6:
        return ets_forecast(series, n_months)
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fit = ARIMA(series, order=(1, 1, 1)).fit()
        res = fit.get_forecast(steps=n_months)
        fc_raw = res.predicted_mean
        ci = res.conf_int(alpha=0.05)

        idx = _future_index(series, n_months)
        fc = pd.Series(fc_raw.values, index=idx).clip(lower=0)
        lo = pd.Series(ci.iloc[:, 0].values, index=idx).clip(lower=0)
        hi = pd.Series(ci.iloc[:, 1].values, index=idx)
        return fc, lo, hi
    except Exception:
        return ets_forecast(series, n_months)


def ridge_forecast(series: pd.Series, n_months: int = 3) -> tuple:
    """
    Ridge regression with time index + month-of-year + lag features.
    Interpretable and robust to small datasets.
    """
    if len(series) < 4:
        return ets_forecast(series, n_months)
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        df = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
        df["t"] = np.arange(len(df))
        df["month"] = pd.DatetimeIndex(df["ds"]).month
        df["lag1"] = df["y"].shift(1)
        df["lag2"] = df["y"].shift(2)
        df["lag3"] = df["y"].shift(3)
        df = df.dropna()

        feat_cols = ["t", "month", "lag1", "lag2", "lag3"]
        X = df[feat_cols].values
        y = df["y"].values

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = Ridge(alpha=10.0)
        model.fit(X_sc, y)

        last_t = int(df["t"].iloc[-1])
        tail = list(series.values[-3:])
        preds, idx = [], _future_index(series, n_months)

        for i, fd in enumerate(idx):
            t = last_t + i + 1
            lag1 = tail[-1] if len(tail) >= 1 else series.mean()
            lag2 = tail[-2] if len(tail) >= 2 else series.mean()
            lag3 = tail[-3] if len(tail) >= 3 else series.mean()
            row = scaler.transform([[t, fd.month, lag1, lag2, lag3]])
            p = max(0.0, model.predict(row)[0])
            preds.append(p)
            tail.append(p)

        fc = pd.Series(preds, index=idx)
        resid = y - model.predict(X_sc)
        std = resid.std()
        lo = (fc - 1.96 * std).clip(lower=0)
        hi = fc + 1.96 * std
        return fc, lo, hi
    except Exception:
        return ets_forecast(series, n_months)


def prophet_forecast(series: pd.Series, n_months: int = 3) -> tuple:
    """Facebook Prophet — handles seasonality and change-points."""
    if not PROPHET_AVAILABLE:
        return ets_forecast(series, n_months)
    if len(series) < 6:
        return ets_forecast(series, n_months)
    try:
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        m = Prophet(
            yearly_seasonality=len(series) >= 12,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            changepoint_prior_scale=0.05,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)

        idx = _future_index(series, n_months)
        future = pd.DataFrame({"ds": idx})
        fc_df = m.predict(future)

        fc = pd.Series(fc_df["yhat"].values, index=idx).clip(lower=0)
        lo = pd.Series(fc_df["yhat_lower"].values, index=idx).clip(lower=0)
        hi = pd.Series(fc_df["yhat_upper"].values, index=idx)
        return fc, lo, hi
    except Exception:
        return ets_forecast(series, n_months)


def run_all_forecasts(series: pd.Series, n_months: int = 3) -> dict:
    """Run every available model and return dict of results."""
    results = {
        "Rolling avg": rolling_forecast(series, n_months),
        "ETS (Holt)": ets_forecast(series, n_months),
        "ARIMA(1,1,1)": arima_forecast(series, n_months),
        "Ridge": ridge_forecast(series, n_months),
    }
    if PROPHET_AVAILABLE and len(series) >= 6:
        results["Prophet"] = prophet_forecast(series, n_months)
    return {k: v for k, v in results.items() if v[0] is not None}


def leave_n_out_cv(series: pd.Series, n_test: int = 3) -> pd.DataFrame:
    """
    Leave-N-out cross-validation: train on all but last n_test months,
    evaluate against actual last n_test months.
    Returns a DataFrame with MAE, RMSE, MAPE per model.
    """
    if len(series) < n_test + 3:
        return pd.DataFrame()

    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]

    models: dict = {
        "Rolling avg": rolling_forecast,
        "ETS (Holt)": ets_forecast,
        "ARIMA(1,1,1)": arima_forecast,
        "Ridge": ridge_forecast,
    }
    if PROPHET_AVAILABLE:
        models["Prophet"] = prophet_forecast

    rows = []
    for name, fn in models.items():
        try:
            fc, _, _ = fn(train, n_months=n_test)
            if fc is None:
                continue
            fc_aligned = fc.reindex(test.index).dropna()
            actual = test.loc[fc_aligned.index]
            if len(actual) == 0:
                continue
            mae = float(np.mean(np.abs(actual.values - fc_aligned.values)))
            rmse = float(np.sqrt(np.mean((actual.values - fc_aligned.values) ** 2)))
            mape = float(
                np.mean(np.abs((actual.values - fc_aligned.values) / actual.values.clip(lower=1)))
                * 100
            )
            rows.append({"Model": name, "MAE (฿)": round(mae), "RMSE (฿)": round(rmse), "MAPE (%)": round(mape, 1)})
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("MAE (฿)").reset_index(drop=True)
