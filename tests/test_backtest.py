import pandas as pd
import pytest

from quantlab.backtest import (
    InsufficientDataError,
    _apply_pretrade_costs,
    drawdown_from_equity,
    moving_average_signals,
    run_sma_backtest,
)


def prices(values: list[float]) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, name="Adjusted close")


def test_requires_a_full_long_window_and_a_following_return() -> None:
    with pytest.raises(InsufficientDataError, match="at least 4"):
        run_sma_backtest(prices([10, 11, 12]), short_window=2, long_window=3)


def test_signals_wait_for_full_windows_and_position_is_lagged_one_bar() -> None:
    series = prices([3, 2, 1, 2, 3, 4, 5])
    signals = moving_average_signals(series, short_window=2, long_window=3)

    assert signals["signal"].iloc[:2].isna().all()
    assert signals["signal"].iloc[2] == 0
    assert signals["signal"].iloc[4] == 1

    result = run_sma_backtest(series, short_window=2, long_window=3, transaction_cost_pct=0)
    assert result.daily.index.equals(
        pd.date_range("2025-01-04", periods=4, freq="D")
    )
    assert result.daily["position"].tolist() == [0, 0, 1, 1]
    # The close-based signal turns long on Jan 5; the modeled position changes
    # on Jan 6, so the Jan 5 close does not affect the return ending that day.
    assert result.daily.loc["2025-01-05", "signal"] == 1
    assert result.daily.loc["2025-01-05", "position"] == 0
    assert result.daily["long_sma"].notna().all()


def test_near_total_loss_with_entry_fee_does_not_create_negative_equity() -> None:
    result = run_sma_backtest(
        prices([100.0, 110.0, 0.01]),
        short_window=1,
        long_window=2,
        transaction_cost_pct=10.0,
    )

    assert result.daily["position"].tolist() == [1]
    assert result.daily["strategy_cost"].iloc[0] == pytest.approx(0.10)
    expected_equity = (1.0 - 0.10) * (1.0 + result.daily["daily_return"].iloc[0])
    assert result.daily["strategy_equity"].iloc[0] == pytest.approx(expected_equity)
    assert result.daily["strategy_equity"].iloc[0] >= 0.0


def test_zero_fees_leave_gross_returns_unchanged() -> None:
    gross_returns = pd.Series([0.12, -0.25, 0.0])
    zero_costs = pd.Series([0.0, 0.0, 0.0])

    net_returns = _apply_pretrade_costs(gross_returns, zero_costs)

    assert net_returns.tolist() == pytest.approx(gross_returns.tolist())


def test_ordinary_entry_and_exit_fees_reduce_equity_before_returns() -> None:
    # First row enters before a +10% market return; second row exits to cash.
    gross_returns = pd.Series([0.10, 0.0])
    costs = pd.Series([0.01, 0.02])

    net_returns = _apply_pretrade_costs(gross_returns, costs)

    assert net_returns.tolist() == pytest.approx([0.089, -0.02])


def test_full_asset_loss_with_entry_fee_leaves_zero_not_negative_equity() -> None:
    net_returns = _apply_pretrade_costs(
        pd.Series([-1.0]),
        pd.Series([0.10]),
    )
    equity = (1.0 + net_returns).cumprod()

    assert net_returns.iloc[0] == pytest.approx(-1.0)
    assert equity.iloc[0] == pytest.approx(0.0)


def test_costs_are_charged_on_strategy_turnover_and_buy_hold_entry() -> None:
    result = run_sma_backtest(
        prices([10, 10, 12, 14, 8, 6, 9]),
        short_window=1,
        long_window=2,
        transaction_cost_pct=1.0,
    )

    assert result.daily["position"].tolist() == [0, 1, 1, 0, 0]
    assert result.daily["strategy_cost"].tolist() == pytest.approx(
        [0, 0.01, 0, 0.01, 0]
    )
    assert result.daily["buy_hold_cost"].tolist() == pytest.approx(
        [0.01, 0, 0, 0, 0]
    )
    assert result.daily["strategy_return"].iloc[1] == pytest.approx(
        (1.0 + result.daily["daily_return"].iloc[1]) * (1.0 - 0.01) - 1.0
    )
    assert result.daily["strategy_return"].iloc[3] == pytest.approx(-0.01)
    assert result.daily["buy_hold_return"].iloc[0] == pytest.approx(
        (1.0 + result.daily["daily_return"].iloc[0]) * (1.0 - 0.01) - 1.0
    )
    assert result.daily.index.equals(result.daily["buy_hold_return"].index)
    assert result.daily.index.equals(result.daily["strategy_return"].index)


def test_cumulative_returns_compound_net_returns_on_the_shared_dates() -> None:
    result = run_sma_backtest(
        prices([10, 10, 12, 14, 8, 6, 9]),
        short_window=1,
        long_window=2,
        transaction_cost_pct=1.0,
    )
    expected_strategy_total = (1 + result.daily["strategy_return"]).prod() - 1
    expected_buy_hold_total = (1 + result.daily["buy_hold_return"]).prod() - 1

    assert result.daily.index.equals(result.daily["strategy_return"].index)
    assert result.daily.index.equals(result.daily["buy_hold_return"].index)
    assert result.daily["strategy_cumulative_return"].iloc[-1] == pytest.approx(
        expected_strategy_total
    )
    assert result.daily["buy_hold_cumulative_return"].iloc[-1] == pytest.approx(
        expected_buy_hold_total
    )
    assert result.summary.loc["SMA crossover", "Cumulative return"] == pytest.approx(
        expected_strategy_total
    )
    assert result.summary.loc["Buy & hold", "Cumulative return"] == pytest.approx(
        expected_buy_hold_total
    )


def test_drawdown_includes_initial_capital_as_a_peak() -> None:
    equity = pd.Series([0.9, 1.08, 0.86, 1.0])

    drawdown = drawdown_from_equity(equity)

    assert drawdown.tolist() == pytest.approx(
        [-0.10, 0.0, (0.86 / 1.08) - 1.0, (1.0 / 1.08) - 1.0]
    )
    assert drawdown.min() == pytest.approx((0.86 / 1.08) - 1.0)
