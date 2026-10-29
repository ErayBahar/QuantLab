"""Deterministic calculations for a long-or-flat moving-average crossover."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import pandas as pd


class InsufficientDataError(ValueError):
    """Raised when full moving averages and a following return cannot be formed."""


@dataclass(frozen=True)
class BacktestResult:
    """Daily results and summary metrics over a shared evaluation window."""

    daily: pd.DataFrame
    summary: pd.DataFrame
    short_window: int
    long_window: int
    transaction_cost_pct: float

    @property
    def evaluation_start(self) -> pd.Timestamp:
        return self.daily.index[0]

    @property
    def evaluation_end(self) -> pd.Timestamp:
        return self.daily.index[-1]


def _validate_parameters(
    short_window: int,
    long_window: int,
    transaction_cost_pct: float | None = None,
) -> None:
    for name, value in (
        ("short_window", short_window),
        ("long_window", long_window),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive whole number.")
    if short_window >= long_window:
        raise ValueError("The short moving-average window must be less than the long window.")

    if transaction_cost_pct is not None:
        if (
            not math.isfinite(transaction_cost_pct)
            or transaction_cost_pct < 0
            or transaction_cost_pct >= 100
        ):
            raise ValueError("Transaction cost must be between 0% and 100%.")


def _prepare_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series indexed by date.")

    numeric = pd.to_numeric(prices, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()

    values = numeric.to_numpy(dtype=float)
    valid = (~index.isna()) & np.isfinite(values)
    cleaned = pd.Series(
        values[valid],
        index=index[valid],
        name=prices.name or "Adjusted close",
        dtype=float,
    )
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")].sort_index()
    if cleaned.empty:
        raise InsufficientDataError("No valid prices were supplied.")
    if (cleaned <= 0).any():
        raise ValueError("All prices must be positive.")
    return cleaned


def moving_average_signals(
    prices: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Compute full-window SMAs and close-based signals.

    Signal values remain missing until both complete moving-average windows
    exist. A value of 1 means long; 0 means flat.
    """
    _validate_parameters(short_window, long_window)
    clean_prices = _prepare_prices(prices)
    short_sma = clean_prices.rolling(
        window=short_window, min_periods=short_window
    ).mean()
    long_sma = clean_prices.rolling(
        window=long_window, min_periods=long_window
    ).mean()
    full_windows = short_sma.notna() & long_sma.notna()
    signal = (short_sma > long_sma).astype(float).where(full_windows)
    return pd.DataFrame(
        {
            "adjusted_close": clean_prices,
            "short_sma": short_sma,
            "long_sma": long_sma,
            "signal": signal,
        }
    )


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Return drawdown from a running peak, including initial capital of 1.0."""
    if equity.empty:
        return equity.astype(float).rename("drawdown")
    running_peak = equity.cummax().clip(lower=1.0)
    return (equity / running_peak - 1.0).rename("drawdown")


def _equity_from_returns(returns: pd.Series, name: str) -> pd.Series:
    return (1.0 + returns).cumprod().rename(name)


def _apply_pretrade_costs(
    gross_returns: pd.Series,
    cost_fraction: pd.Series,
) -> pd.Series:
    """Debit fees from pre-trade equity, then apply the interval's market return."""
    return ((1.0 + gross_returns) * (1.0 - cost_fraction) - 1.0)


def run_sma_backtest(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
    transaction_cost_pct: float = 0.10,
) -> BacktestResult:
    """Backtest a long-or-flat SMA crossover against buy-and-hold.

    A signal formed from close t-1 controls the return ending at close t.
    This one-row lag prevents the close at t from influencing that same
    close-to-close return. It still implies an idealized fill at close t-1;
    actual next-open or other execution prices are not modeled.

    Transaction costs are assessed at the prior close, when the position for
    the next return interval is established. The fee is a one-way fraction of
    pre-trade equity times turnover; it reduces equity before the interval's
    market return is applied. The strategy pays on each position change.
    Buy-and-hold pays for its initial purchase, with no forced sale charged at
    the end of the sample.
    """
    _validate_parameters(short_window, long_window, transaction_cost_pct)
    clean_prices = _prepare_prices(prices)
    if len(clean_prices) <= long_window:
        raise InsufficientDataError(
            f"Need at least {long_window + 1} valid daily prices to form a full "
            "long moving average and one following close-to-close return."
        )

    indicators = moving_average_signals(
        clean_prices,
        short_window=short_window,
        long_window=long_window,
    )
    daily_return = clean_prices.pct_change()
    # A position on date t uses only the signal available at the prior close.
    position = indicators["signal"].shift(1)
    eligible = position.notna() & daily_return.notna()
    if not eligible.any():
        raise InsufficientDataError(
            "There are no returns after both moving-average windows are available."
        )

    evaluation_dates = clean_prices.index[eligible]
    position = position.loc[evaluation_dates].astype(float)
    daily_return = daily_return.loc[evaluation_dates].astype(float)
    indicators = indicators.loc[evaluation_dates]

    prior_position = position.shift(1).fillna(0.0)
    turnover = (position - prior_position).abs()
    cost_rate = transaction_cost_pct / 100.0
    strategy_cost = turnover * cost_rate
    strategy_gross_return = position * daily_return
    strategy_return = _apply_pretrade_costs(strategy_gross_return, strategy_cost)

    # Buy-and-hold owns the first evaluated return interval, anchored at its prior close.
    buy_hold_position = pd.Series(1.0, index=evaluation_dates)
    buy_hold_turnover = buy_hold_position.diff().fillna(1.0).abs()
    buy_hold_cost = buy_hold_turnover * cost_rate
    buy_hold_return = _apply_pretrade_costs(daily_return, buy_hold_cost)

    strategy_equity = _equity_from_returns(strategy_return, "strategy_equity")
    buy_hold_equity = _equity_from_returns(buy_hold_return, "buy_hold_equity")
    strategy_drawdown = drawdown_from_equity(strategy_equity)
    buy_hold_drawdown = drawdown_from_equity(buy_hold_equity)

    daily = pd.DataFrame(
        {
            "adjusted_close": indicators["adjusted_close"],
            "short_sma": indicators["short_sma"],
            "long_sma": indicators["long_sma"],
            "signal": indicators["signal"],
            "position": position,
            "daily_return": daily_return,
            "turnover": turnover,
            "strategy_cost": strategy_cost,
            "strategy_return": strategy_return,
            "buy_hold_cost": buy_hold_cost,
            "buy_hold_return": buy_hold_return,
            "strategy_equity": strategy_equity,
            "buy_hold_equity": buy_hold_equity,
            "strategy_cumulative_return": strategy_equity - 1.0,
            "buy_hold_cumulative_return": buy_hold_equity - 1.0,
            "strategy_drawdown": strategy_drawdown,
            "buy_hold_drawdown": buy_hold_drawdown,
        },
        index=evaluation_dates,
    )

    summary = pd.DataFrame(
        {
            "Cumulative return": [
                strategy_equity.iloc[-1] - 1.0,
                buy_hold_equity.iloc[-1] - 1.0,
            ],
            "Maximum drawdown": [
                strategy_drawdown.min(),
                buy_hold_drawdown.min(),
            ],
        },
        index=["SMA crossover", "Buy & hold"],
    )

    return BacktestResult(
        daily=daily,
        summary=summary,
        short_window=int(short_window),
        long_window=int(long_window),
        transaction_cost_pct=float(transaction_cost_pct),
    )
