import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio import (
    PortfolioAnalyticsError,
    calculate_portfolio_analytics,
    validate_portfolio_weights,
)


def price_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-06", periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=dates)


def test_weights_are_normalized_for_rounding_and_reject_invalid_portfolios() -> None:
    weights = validate_portfolio_weights({"aaa": 0.6, "BBB": 0.4000005})
    assert weights.index.tolist() == ["AAA", "BBB"]
    assert weights.sum() == pytest.approx(1.0)

    with pytest.raises(PortfolioAnalyticsError, match="sum to 100%"):
        validate_portfolio_weights({"AAA": 0.6, "BBB": 0.3})
    with pytest.raises(PortfolioAnalyticsError, match="cannot be negative"):
        validate_portfolio_weights({"AAA": 1.1, "BBB": -0.1})
    with pytest.raises(PortfolioAnalyticsError, match="Duplicate ticker"):
        validate_portfolio_weights({"aaa": 0.5, "AAA": 0.5})


def test_weighted_returns_and_cumulative_portfolio_return() -> None:
    prices = price_frame(
        {
            "AAA": [100.0, 110.0, 121.0, 110.0],
            "BBB": [100.0, 100.0, 110.0, 121.0],
        }
    )

    result = calculate_portfolio_analytics(prices, {"AAA": 0.75, "BBB": 0.25})
    expected_returns = np.array([0.075, 0.10, -0.04318181818181818])

    assert result.portfolio_returns.to_numpy() == pytest.approx(expected_returns)
    assert result.portfolio_equity.iloc[-1] == pytest.approx(
        np.prod(1.0 + expected_returns)
    )
    assert result.metrics.loc[
        "Configured portfolio", "Cumulative return"
    ] == pytest.approx(np.prod(1.0 + expected_returns) - 1.0)


def test_equal_weight_benchmark_is_rebalanced_on_portfolio_dates() -> None:
    prices = price_frame(
        {
            "AAA": [100.0, 110.0, 121.0, 110.0],
            "BBB": [100.0, 100.0, 110.0, 121.0],
        }
    )

    result = calculate_portfolio_analytics(prices, {"AAA": 0.9, "BBB": 0.1})
    expected_equal_returns = prices.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    expected_equal_returns.index.name = "date"

    pd.testing.assert_series_equal(
        result.equal_weight_returns,
        expected_equal_returns.rename("Equal weight"),
        check_freq=False,
    )
    assert result.equal_weight_returns.index.equals(result.portfolio_returns.index)
    assert result.metrics.loc[
        "Equal weight", "Cumulative return"
    ] == pytest.approx((1.0 + expected_equal_returns).prod() - 1.0)


def test_only_shared_price_dates_are_used_and_irregularity_is_reported() -> None:
    dates = pd.bdate_range("2025-01-06", periods=4)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 120.0, 132.0],
            "BBB": [100.0, np.nan, 110.0, 121.0],
        },
        index=dates,
    )

    result = calculate_portfolio_analytics(prices, {"AAA": 0.5, "BBB": 0.5})

    assert result.common_prices.index.equals(pd.DatetimeIndex([dates[0], dates[2], dates[3]]))
    assert result.asset_returns.index.equals(pd.DatetimeIndex([dates[2], dates[3]]))
    assert result.asset_returns.loc[dates[2], "AAA"] == pytest.approx(0.20)
    assert result.asset_returns.loc[dates[2], "BBB"] == pytest.approx(0.10)
    assert any("Sparse overlap" in warning for warning in result.warnings)
    assert any("Irregular overlap" in warning for warning in result.warnings)


def test_requires_enough_overlapping_prices_for_volatility() -> None:
    dates = pd.bdate_range("2025-01-06", periods=3)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, np.nan],
            "BBB": [np.nan, 100.0, 101.0],
        },
        index=dates,
    )

    with pytest.raises(PortfolioAnalyticsError, match="At least three shared price dates"):
        calculate_portfolio_analytics(prices, {"AAA": 0.5, "BBB": 0.5})


def test_non_overlapping_price_dates_raise_an_actionable_error() -> None:
    dates = pd.bdate_range("2025-01-06", periods=4)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, np.nan, np.nan],
            "BBB": [np.nan, np.nan, 100.0, 101.0],
        },
        index=dates,
    )

    with pytest.raises(PortfolioAnalyticsError, match="found 0"):
        calculate_portfolio_analytics(prices, {"AAA": 0.5, "BBB": 0.5})


def test_annualized_volatility_uses_sample_std_and_252_intervals() -> None:
    prices = price_frame(
        {
            "AAA": [100.0, 101.0, 104.03],
            "BBB": [100.0, 101.0, 104.03],
        }
    )

    result = calculate_portfolio_analytics(prices, {"AAA": 0.5, "BBB": 0.5})
    expected_volatility = 0.01 * np.sqrt(504.0)

    assert result.portfolio_returns.tolist() == pytest.approx([0.01, 0.03])
    assert result.metrics.loc[
        "Configured portfolio", "Annualized volatility"
    ] == pytest.approx(expected_volatility)


def test_maximum_drawdown_uses_compounded_portfolio_equity() -> None:
    prices = price_frame(
        {
            "AAA": [100.0, 110.0, 99.0],
            "BBB": [100.0, 110.0, 99.0],
        }
    )

    result = calculate_portfolio_analytics(prices, {"AAA": 0.5, "BBB": 0.5})

    assert result.portfolio_equity.tolist() == pytest.approx([1.1, 0.99])
    assert result.metrics.loc[
        "Configured portfolio", "Maximum drawdown"
    ] == pytest.approx(-0.10)


def test_correlations_are_computed_and_undefined_values_remain_nan() -> None:
    prices = price_frame(
        {
            "AAA": [100.0, 110.0, 99.0, 108.9],
            "BBB": [100.0, 90.0, 99.0, 89.1],
            "CCC": [100.0, 100.0, 100.0, 100.0],
        }
    )

    result = calculate_portfolio_analytics(
        prices,
        {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2},
    )

    assert result.correlations.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert result.correlations.loc["AAA", "BBB"] == pytest.approx(-1.0)
    assert result.correlations["CCC"].isna().all()
