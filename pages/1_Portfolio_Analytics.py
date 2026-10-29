"""Portfolio analytics page for QuantLab."""

from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.data import PriceDataError, PriceHistory, load_price_history, normalize_ticker
from quantlab.portfolio import (
    PortfolioAnalyticsError,
    PortfolioAnalyticsResult,
    calculate_portfolio_analytics,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_price_history(ticker: str, start: date, end: date) -> PriceHistory:
    return load_price_history(ticker, start, end)


def _parse_tickers(raw_tickers: str) -> list[str]:
    tokens = [token.strip() for token in raw_tickers.split(",") if token.strip()]
    tickers = [normalize_ticker(token) for token in tokens]
    if len(tickers) < 2:
        raise PortfolioAnalyticsError("Enter at least two comma-separated tickers.")
    if len(set(tickers)) != len(tickers):
        raise PortfolioAnalyticsError("Each ticker may appear only once.")
    return tickers


def _default_weight_percentages(count: int) -> list[float]:
    base = round(100.0 / count, 2)
    values = [base] * (count - 1)
    values.append(round(100.0 - math.fsum(values), 2))
    return values


def _build_equity_figure(result: PortfolioAnalyticsResult) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.portfolio_equity.index,
            y=(result.portfolio_equity - 1.0) * 100,
            mode="lines",
            name="Configured portfolio",
            line={"color": "#52c8b8", "width": 2.5},
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.equal_weight_equity.index,
            y=(result.equal_weight_equity - 1.0) * 100,
            mode="lines",
            name="Equal weight",
            line={"color": "#9a8df2", "width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        margin={"l": 12, "r": 18, "t": 20, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce5f3"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(showgrid=False, title="Shared valuation date")
    figure.update_yaxes(
        title="Cumulative return",
        ticksuffix="%",
        gridcolor="rgba(145,166,199,.12)",
        zerolinecolor="rgba(145,166,199,.2)",
    )
    return figure


def _build_correlation_figure(correlations: pd.DataFrame) -> go.Figure:
    tickers = correlations.columns.tolist()
    values = correlations.to_numpy(dtype=float)
    labels = [
        [
            f"{value:.2f}" if np.isfinite(value) else ""
            for value in row
        ]
        for row in values
    ]
    undefined_x: list[str] = []
    undefined_y: list[str] = []
    for row_index, ticker_row in enumerate(tickers):
        for column_index, ticker_column in enumerate(tickers):
            if not np.isfinite(values[row_index, column_index]):
                undefined_x.append(ticker_column)
                undefined_y.append(ticker_row)

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=tickers,
            y=tickers,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale="RdBu",
            text=labels,
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar={"title": "Pearson r"},
            hovertemplate="%{y} / %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    if undefined_x:
        figure.add_trace(
            go.Scatter(
                x=undefined_x,
                y=undefined_y,
                mode="text",
                text=["n/a"] * len(undefined_x),
                textfont={"color": "#e8edf6", "size": 12},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        template="plotly_dark",
        height=430,
        margin={"l": 12, "r": 18, "t": 20, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce5f3"},
    )
    figure.update_xaxes(side="bottom", showgrid=False)
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


st.title("Portfolio analytics")
st.caption("Compare a weighted, daily-rebalanced portfolio with an equal-weight benchmark.")
st.info(
    "Weights are reset at each shared valuation close for the next interval. Results are "
    "gross of transaction costs, slippage, and taxes. Assets are assumed to share a quote "
    "currency; no FX conversion is applied."
)

st.sidebar.markdown("### Portfolio inputs")
raw_tickers = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, MSFT")
try:
    tickers = _parse_tickers(raw_tickers)
except (PortfolioAnalyticsError, PriceDataError) as exc:
    st.error(str(exc))
    st.stop()

today = date.today()
date_range = st.sidebar.date_input(
    "Price history",
    value=(today - timedelta(days=365 * 3), today),
    help="Returns begin at the second date shared by every selected asset.",
)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select a start and end date.")
    st.stop()
start_date, end_date = date_range
if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()

st.sidebar.markdown("#### Target weights")
default_weights = _default_weight_percentages(len(tickers))
weight_percentages: dict[str, float] = {}
weight_key_prefix = "_".join(tickers)
for ticker, default_weight in zip(tickers, default_weights):
    weight_percentages[ticker] = st.sidebar.number_input(
        f"{ticker} weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_weight,
        step=0.01,
        format="%.2f",
        key=f"portfolio_weight_{weight_key_prefix}_{ticker}",
    )

weight_total = math.fsum(weight_percentages.values())
if not math.isclose(weight_total, 100.0, rel_tol=0.0, abs_tol=0.0001):
    st.error(f"Weights must sum to 100%; current total is {weight_total:.2f}%.")
    st.stop()

weights = {ticker: value / 100.0 for ticker, value in weight_percentages.items()}
price_histories: dict[str, PriceHistory] = {}
load_errors: list[str] = []
with st.spinner("Loading adjusted price histories..."):
    for ticker in tickers:
        try:
            price_histories[ticker] = cached_price_history(ticker, start_date, end_date)
        except PriceDataError as exc:
            load_errors.append(f"{ticker}: {exc}")

if load_errors:
    for error in load_errors:
        st.error(error)
    st.stop()

price_frame = pd.concat(
    {ticker: history.prices for ticker, history in price_histories.items()},
    axis=1,
)
try:
    result = calculate_portfolio_analytics(price_frame, weights)
except PortfolioAnalyticsError as exc:
    st.error(str(exc))
    st.stop()

for warning in result.warnings:
    st.warning(warning)

source_text = " · ".join(
    f"{ticker}: {history.source}" for ticker, history in price_histories.items()
)
st.caption(
    f"Sources: {source_text} · Evaluation: {result.evaluation_start:%b %d, %Y} to "
    f"{result.evaluation_end:%b %d, %Y} · {len(result.asset_returns):,} shared return intervals"
)
st.caption(
    "Annualized volatility is sample standard deviation × √252. This assumes roughly "
    "252 shared trading intervals per year; sparse or irregular overlap is flagged above."
)

st.caption(
    "Annualized Sharpe ratio assumes a 0% risk-free rate and annualizes daily excess "
    "return using sqrt(252)."
)

portfolio_metrics = result.metrics.loc["Configured portfolio"]
metric_columns = st.columns(4)
metric_columns[0].metric(
    "Portfolio cumulative return",
    f"{portfolio_metrics['Cumulative return']:.2%}",
)
metric_columns[1].metric(
    "Portfolio annualized volatility",
    f"{portfolio_metrics['Annualized volatility']:.2%}",
)
metric_columns[2].metric(
    "Portfolio maximum drawdown",
    f"{portfolio_metrics['Maximum drawdown']:.2%}",
)
sharpe_ratio = portfolio_metrics["Annualized Sharpe ratio"]
metric_columns[3].metric(
    "Annualized Sharpe ratio",
    f"{sharpe_ratio:.2f}" if pd.notna(sharpe_ratio) else "n/a",
)

chart_column, correlation_column = st.columns([1.35, 1.0])
with chart_column:
    st.subheader("Cumulative returns")
    st.plotly_chart(
        _build_equity_figure(result),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
with correlation_column:
    st.subheader("Asset return correlation")
    st.plotly_chart(
        _build_correlation_figure(result.correlations),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption("n/a marks an undefined correlation, such as a zero-variance return series.")

st.subheader("Portfolio comparison")
st.dataframe(
    result.metrics.style.format(
        {
            "Cumulative return": "{:.2%}",
            "Annualized volatility": "{:.2%}",
            "Annualized Sharpe ratio": "{:.2f}",
            "Maximum drawdown": "{:.2%}",
        },
        na_rep="n/a",
    ),
    use_container_width=True,
)
