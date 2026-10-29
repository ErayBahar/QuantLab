"""Pure portfolio return and risk calculations for QuantLab."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from quantlab.backtest import drawdown_from_equity


TRADING_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RATE = 0.0
SHARPE_VOLATILITY_ABSOLUTE_TOLERANCE = 1e-12
SPARSE_RETURN_WARNING_THRESHOLD = 60
WEIGHT_SUM_TOLERANCE = 1e-6


class PortfolioAnalyticsError(ValueError):
    """Raised when portfolio weights or aligned price data are unusable."""


@dataclass(frozen=True)
class PortfolioAnalyticsResult:
    """Portfolio and equal-weight results on one common observation index."""

    common_prices: pd.DataFrame
    asset_returns: pd.DataFrame
    weights: pd.Series
    portfolio_returns: pd.Series
    equal_weight_returns: pd.Series
    portfolio_equity: pd.Series
    equal_weight_equity: pd.Series
    portfolio_drawdown: pd.Series
    equal_weight_drawdown: pd.Series
    correlations: pd.DataFrame
    metrics: pd.DataFrame
    warnings: tuple[str, ...]

    @property
    def evaluation_start(self) -> pd.Timestamp:
        return self.asset_returns.index[0]

    @property
    def evaluation_end(self) -> pd.Timestamp:
        return self.asset_returns.index[-1]


def validate_portfolio_weights(weights: Mapping[str, float]) -> pd.Series:
    """Validate decimal long-only weights and normalize floating-point dust.

    Weights must sum to 1 within WEIGHT_SUM_TOLERANCE. Only the tiny accepted
    rounding error is normalized; materially incomplete or leveraged weights
    are rejected.
    """
    if not isinstance(weights, Mapping):
        raise TypeError("weights must map ticker symbols to decimal weights.")
    if len(weights) < 2:
        raise PortfolioAnalyticsError("Select at least two assets for a portfolio.")

    normalized: dict[str, float] = {}
    for ticker, raw_weight in weights.items():
        symbol = str(ticker).strip().upper()
        if not symbol:
            raise PortfolioAnalyticsError("Ticker symbols cannot be blank.")
        if symbol in normalized:
            raise PortfolioAnalyticsError(f"Duplicate ticker: {symbol}.")
        if isinstance(raw_weight, bool):
            raise PortfolioAnalyticsError(f"Weight for {symbol} must be numeric.")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise PortfolioAnalyticsError(
                f"Weight for {symbol} must be numeric."
            ) from exc
        if not math.isfinite(weight):
            raise PortfolioAnalyticsError(f"Weight for {symbol} must be finite.")
        if weight < 0:
            raise PortfolioAnalyticsError(
                f"Weight for {symbol} cannot be negative; short positions are not supported."
            )
        normalized[symbol] = weight

    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        raise PortfolioAnalyticsError(
            f"Portfolio weights must sum to 100%; current total is {total:.6%}."
        )

    return pd.Series(
        {ticker: weight / total for ticker, weight in normalized.items()},
        name="weight",
        dtype=float,
    )


def _prepare_common_prices(
    prices: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame with one adjusted-close column per ticker.")
    if prices.empty or prices.shape[1] < 2:
        raise PortfolioAnalyticsError("Provide adjusted prices for at least two assets.")

    frame = prices.copy()
    symbols = [str(column).strip().upper() for column in frame.columns]
    if any(not symbol for symbol in symbols):
        raise PortfolioAnalyticsError("Price columns must have ticker symbols.")
    if len(set(symbols)) != len(symbols):
        raise PortfolioAnalyticsError("Price data contains duplicate ticker columns.")
    frame.columns = symbols

    missing_tickers = weights.index.difference(frame.columns)
    extra_tickers = frame.columns.difference(weights.index)
    if len(missing_tickers) or len(extra_tickers):
        details = []
        if len(missing_tickers):
            details.append(f"missing prices for {', '.join(missing_tickers)}")
        if len(extra_tickers):
            details.append(f"no weights supplied for {', '.join(extra_tickers)}")
        raise PortfolioAnalyticsError("Price and weight tickers do not match: " + "; ".join(details) + ".")

    frame = frame.loc[:, weights.index]
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    if index.isna().any():
        raise PortfolioAnalyticsError("Price data contains an invalid date.")
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise PortfolioAnalyticsError("Price data contains duplicate dates.")
    frame.index = index
    frame = frame.sort_index()

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric))
    if (numeric <= 0).any().any():
        raise PortfolioAnalyticsError("Adjusted prices must be positive.")

    # Complete-case shared dates avoid manufacturing zero returns with forward-fill.
    common_prices = numeric.dropna(axis=0, how="any")
    if len(common_prices) < 3:
        raise PortfolioAnalyticsError(
            "At least three shared price dates (two return intervals) are required; "
            f"found {len(common_prices)}."
        )
    return common_prices


def _observation_warnings(common_dates: pd.DatetimeIndex, return_count: int) -> tuple[str, ...]:
    warnings: list[str] = []
    if return_count < SPARSE_RETURN_WARNING_THRESHOLD:
        warnings.append(
            f"Sparse overlap: only {return_count} shared return intervals are available; "
            f"volatility estimates with fewer than {SPARSE_RETURN_WARNING_THRESHOLD} "
            "intervals are unstable."
        )

    days = np.asarray(common_dates.date, dtype="datetime64[D]")
    weekday_gaps = np.busday_count(days[:-1], days[1:])
    irregular = weekday_gaps != 1
    if irregular.any():
        warnings.append(
            f"Irregular overlap: {int(irregular.sum())} of {len(weekday_gaps)} shared "
            "price intervals span other than one weekday. The 252-interval annualization "
            "is approximate; exchange-specific holiday calendars are not modeled."
        )
    return tuple(warnings)


def calculate_portfolio_analytics(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
) -> PortfolioAnalyticsResult:
    """Calculate daily-rebalanced portfolio metrics from adjusted close prices.

    Weights are reset at each common valuation date for the following interval.
    Returns are gross of transaction costs, slippage, taxes, and cash interest.
    All assets are assumed to use the same quote currency; no FX conversion is
    applied. Returns are computed only between consecutive shared price dates.
    Annualized volatility uses sample standard deviation times sqrt(252).
    Annualized Sharpe ratio assumes a 0% annual risk-free rate and uses sample
    standard deviation of daily excess returns. Daily volatilities at or below
    1e-12 are treated as zero to avoid amplifying floating-point noise.
    """
    validated_weights = validate_portfolio_weights(weights)
    common_prices = _prepare_common_prices(prices, validated_weights)
    asset_returns = common_prices.pct_change(fill_method=None).iloc[1:]
    asset_returns.columns.name = "ticker"
    asset_returns.index.name = "date"

    ordered_weights = validated_weights.reindex(asset_returns.columns)
    portfolio_returns = asset_returns.dot(ordered_weights).rename("Portfolio")
    equal_weights = pd.Series(
        1.0 / len(asset_returns.columns),
        index=asset_returns.columns,
        name="equal_weight",
    )
    equal_weight_returns = asset_returns.dot(equal_weights).rename("Equal weight")

    portfolio_equity = (1.0 + portfolio_returns).cumprod().rename("Portfolio")
    equal_weight_equity = (1.0 + equal_weight_returns).cumprod().rename("Equal weight")
    portfolio_drawdown = drawdown_from_equity(portfolio_equity)
    equal_weight_drawdown = drawdown_from_equity(equal_weight_equity)

    def metrics(
        returns: pd.Series,
        equity: pd.Series,
        drawdown: pd.Series,
    ) -> tuple[float, float, float, float]:
        cumulative_return = float(equity.iloc[-1] - 1.0)
        annualized_volatility = float(
            returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        daily_risk_free_rate = (
            (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        )
        daily_excess_returns = returns - daily_risk_free_rate
        daily_excess_volatility = float(daily_excess_returns.std(ddof=1))
        annualized_sharpe_ratio = (
            float(
                daily_excess_returns.mean()
                / daily_excess_volatility
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )
            if daily_excess_volatility > SHARPE_VOLATILITY_ABSOLUTE_TOLERANCE
            else math.nan
        )
        maximum_drawdown = float(drawdown.min())
        return (
            cumulative_return,
            annualized_volatility,
            annualized_sharpe_ratio,
            maximum_drawdown,
        )

    portfolio_metrics = metrics(portfolio_returns, portfolio_equity, portfolio_drawdown)
    equal_metrics = metrics(equal_weight_returns, equal_weight_equity, equal_weight_drawdown)
    summary = pd.DataFrame(
        [portfolio_metrics, equal_metrics],
        columns=[
            "Cumulative return",
            "Annualized volatility",
            "Annualized Sharpe ratio",
            "Maximum drawdown",
        ],
        index=["Configured portfolio", "Equal weight"],
    )

    correlations = asset_returns.corr(method="pearson", min_periods=2)
    return PortfolioAnalyticsResult(
        common_prices=common_prices,
        asset_returns=asset_returns,
        weights=validated_weights,
        portfolio_returns=portfolio_returns,
        equal_weight_returns=equal_weight_returns,
        portfolio_equity=portfolio_equity,
        equal_weight_equity=equal_weight_equity,
        portfolio_drawdown=portfolio_drawdown,
        equal_weight_drawdown=equal_weight_drawdown,
        correlations=correlations,
        metrics=summary,
        warnings=_observation_warnings(common_prices.index, len(asset_returns)),
    )
