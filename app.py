"""Streamlit interface for the QuantLab research dashboard."""

from __future__ import annotations

from datetime import date, timedelta
from html import escape

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from quantlab.backtest import (
    BacktestResult,
    InsufficientDataError,
    moving_average_signals,
    run_sma_backtest,
)
from quantlab.data import PriceDataError, PriceHistory, load_price_history


st.set_page_config(
    page_title="QuantLab · Strategy research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 75% -15%, rgba(64, 101, 173, .18), transparent 42%),
          #0a0e17;
      }
      [data-testid="stHeader"] { background: rgba(10, 14, 23, .76); }
      [data-testid="stSidebar"] {
        background: #0d1320;
        border-right: 1px solid rgba(145, 166, 199, .13);
      }
      .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -.035em; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23, 32, 49, .94), rgba(15, 21, 33, .94));
        border: 1px solid rgba(145, 166, 199, .14);
        border-radius: 14px;
        padding: 17px 19px;
      }
      [data-testid="stMetricLabel"] { color: #9eacc2; }
      [data-testid="stMetricValue"] { font-size: 1.7rem; }
      .eyebrow {
        color: #7dd3c7;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
      }
      .source-chip {
        display: inline-block;
        border: 1px solid rgba(125, 211, 199, .3);
        background: rgba(45, 125, 117, .12);
        color: #9de2d7;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .78rem;
      }
      div[data-testid="stTabs"] button { font-weight: 600; }
      div[data-testid="stExpander"] {
        border: 1px solid rgba(145, 166, 199, .14);
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_price_history(ticker: str, start: date, end: date) -> PriceHistory:
    return load_price_history(ticker, start, end)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def build_price_figure(
    history: PriceHistory,
    indicators: pd.DataFrame,
    result: BacktestResult,
) -> go.Figure:
    daily = result.daily
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.prices.index,
            y=history.prices,
            mode="lines",
            name="Adjusted close",
            line={"color": "#e6edf7", "width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>Close: %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["short_sma"],
            mode="lines",
            name=f"{result.short_window} day SMA",
            line={"color": "#52c8b8", "width": 1.5},
            hovertemplate="%{x|%b %d, %Y}<br>Short SMA: %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["long_sma"],
            mode="lines",
            name=f"{result.long_window} day SMA",
            line={"color": "#9a8df2", "width": 1.5},
            hovertemplate="%{x|%b %d, %Y}<br>Long SMA: %{y:,.2f}<extra></extra>",
        )
    )

    prior_position = daily["position"].shift(1).fillna(0.0)
    entries = daily.index[(daily["position"] == 1) & (prior_position == 0)]
    exits = daily.index[(daily["position"] == 0) & (prior_position == 1)]
    if len(entries):
        figure.add_trace(
            go.Scatter(
                x=entries,
                y=daily.loc[entries, "adjusted_close"],
                mode="markers",
                name="Modeled entry",
                marker={"color": "#6de0a7", "size": 10, "symbol": "triangle-up"},
                hovertemplate="%{x|%b %d, %Y}<br>Modeled long entry<extra></extra>",
            )
        )
    if len(exits):
        figure.add_trace(
            go.Scatter(
                x=exits,
                y=daily.loc[exits, "adjusted_close"],
                mode="markers",
                name="Modeled exit",
                marker={"color": "#f0a267", "size": 9, "symbol": "x"},
                hovertemplate="%{x|%b %d, %Y}<br>Modeled exit to cash<extra></extra>",
            )
        )

    figure.update_layout(
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        margin={"l": 12, "r": 18, "t": 18, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce5f3"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(showgrid=False, rangeslider_visible=False)
    figure.update_yaxes(title="Adjusted close", gridcolor="rgba(145,166,199,.12)")
    return figure


def build_performance_figure(daily) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=("Cumulative return", "Drawdown"),
    )
    for name, color, cumulative_column, drawdown_column in (
        (
            "SMA crossover",
            "#52c8b8",
            "strategy_cumulative_return",
            "strategy_drawdown",
        ),
        (
            "Buy & hold",
            "#9a8df2",
            "buy_hold_cumulative_return",
            "buy_hold_drawdown",
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily[cumulative_column] * 100,
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily[drawdown_column] * 100,
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": color, "width": 1.8},
                fill="tozeroy",
                fillcolor=(
                    "rgba(82, 200, 184, .09)"
                    if name == "SMA crossover"
                    else "rgba(154, 141, 242, .09)"
                ),
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    figure.update_layout(
        template="plotly_dark",
        height=570,
        hovermode="x unified",
        margin={"l": 12, "r": 18, "t": 50, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce5f3"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(showgrid=False, row=1, col=1)
    figure.update_xaxes(showgrid=False, row=2, col=1, title="Date")
    figure.update_yaxes(
        title="Return",
        ticksuffix="%",
        gridcolor="rgba(145,166,199,.12)",
        zerolinecolor="rgba(145,166,199,.2)",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="Drawdown",
        ticksuffix="%",
        gridcolor="rgba(145,166,199,.12)",
        zerolinecolor="rgba(145,166,199,.2)",
        row=2,
        col=1,
    )
    return figure


st.sidebar.markdown('<div class="eyebrow">Research controls</div>', unsafe_allow_html=True)
st.sidebar.title("QuantLab")
st.sidebar.caption("Historical strategy analysis")

today = date.today()
ticker = st.sidebar.text_input("Ticker", value="AAPL", max_chars=20).strip().upper()
date_range = st.sidebar.date_input(
    "Price history",
    value=(today - timedelta(days=365 * 3), today),
    help="The first part of this range warms up the moving averages.",
)
short_window = int(
    st.sidebar.number_input("Short moving average (days)", min_value=1, value=20, step=1)
)
long_window = int(
    st.sidebar.number_input("Long moving average (days)", min_value=2, value=50, step=1)
)
transaction_cost_pct = st.sidebar.number_input(
    "One-way transaction cost (%)",
    min_value=0.0,
    max_value=10.0,
    value=0.10,
    step=0.01,
    format="%.2f",
    help="Charged on each crossover entry or exit. Buy-and-hold pays once to enter.",
)
st.sidebar.caption("Long-or-flat strategy · no orders are sent.")
if st.sidebar.button("Refresh market data", use_container_width=True):
    cached_price_history.clear()

st.markdown('<div class="eyebrow">Quantitative research workspace</div>', unsafe_allow_html=True)
st.title("QuantLab")
st.caption(
    "Explore a moving-average strategy against buy-and-hold using adjusted daily prices."
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select a start and end date to load a price history.")
    st.stop()
start_date, end_date = date_range
if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()
if short_window >= long_window:
    st.error("The short moving-average window must be smaller than the long window.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker or 'ticker'} price history..."):
        history = cached_price_history(ticker, start_date, end_date)
except PriceDataError as exc:
    st.error(str(exc))
    st.stop()

indicators = moving_average_signals(history.prices, short_window, long_window)
try:
    result = run_sma_backtest(
        history.prices,
        short_window=short_window,
        long_window=long_window,
        transaction_cost_pct=transaction_cost_pct,
    )
except InsufficientDataError as exc:
    st.error(str(exc))
    st.info(
        "Choose a wider price-history range or a smaller long moving-average window. "
        "A full long window plus one following daily return is required."
    )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

source_label = escape(history.source)
st.markdown(
    f'<span class="source-chip">Price source · {source_label}</span>',
    unsafe_allow_html=True,
)
st.markdown(f"### {ticker} · Moving-average crossover")
st.caption(
    f"Evaluation window: {result.evaluation_start:%b %d, %Y} – "
    f"{result.evaluation_end:%b %d, %Y} · "
    f"{len(result.daily):,} daily returns · "
    f"{short_window}/{long_window} day averages · "
    f"{transaction_cost_pct:.2f}% one-way cost"
)

strategy_return = result.summary.loc["SMA crossover", "Cumulative return"]
buy_hold_return = result.summary.loc["Buy & hold", "Cumulative return"]
strategy_drawdown = result.summary.loc["SMA crossover", "Maximum drawdown"]
buy_hold_drawdown = result.summary.loc["Buy & hold", "Maximum drawdown"]
metric_columns = st.columns(4)
metric_columns[0].metric(
    "Crossover · total return",
    format_percent(strategy_return),
    delta=f"{strategy_return - buy_hold_return:+.2%} vs buy & hold",
)
metric_columns[1].metric(
    "Crossover · max drawdown",
    format_percent(strategy_drawdown),
)
metric_columns[2].metric(
    "Buy & hold · total return",
    format_percent(buy_hold_return),
)
metric_columns[3].metric(
    "Buy & hold · max drawdown",
    format_percent(buy_hold_drawdown),
)

price_tab, performance_tab = st.tabs(["Price & signals", "Performance"])
with price_tab:
    st.plotly_chart(
        build_price_figure(history, indicators, result),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "Entry and exit markers show the lagged model position changes; they are not "
        "actual broker fills."
    )
    with st.expander("Daily calculations"):
        st.dataframe(
            result.daily[
                [
                    "adjusted_close",
                    "short_sma",
                    "long_sma",
                    "signal",
                    "position",
                    "strategy_return",
                    "buy_hold_return",
                    "strategy_drawdown",
                    "buy_hold_drawdown",
                ]
            ].style.format(
                {
                    "adjusted_close": "{:,.2f}",
                    "short_sma": "{:,.2f}",
                    "long_sma": "{:,.2f}",
                    "signal": "{:.0f}",
                    "position": "{:.0f}",
                    "strategy_return": "{:.2%}",
                    "buy_hold_return": "{:.2%}",
                    "strategy_drawdown": "{:.2%}",
                    "buy_hold_drawdown": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=350,
        )

with performance_tab:
    st.plotly_chart(
        build_performance_figure(result.daily),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.dataframe(
        result.summary.style.format(
            {
                "Cumulative return": "{:.2%}",
                "Maximum drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

with st.expander("Methodology & execution assumptions"):
    st.markdown(
        """
        - A long signal is generated only when the full short and long moving-average
          windows are available. The strategy is long when the short average is above
          the long average and otherwise holds cash.
        - For the return ending at close *t*, the model applies the signal calculated
          from close *t−1*. The signal is shifted by one daily row, so close *t* cannot
          affect the position used for the return ending at close *t*.
        - This lagged close-to-close convention implies an idealized fill at the prior
          close, after the signal is calculated. Daily close data cannot guarantee that
          fill. It does not model next-open execution, bid/ask spreads, slippage, taxes,
          partial fills, market impact, or borrow costs.
        - The strategy pays the configured one-way cost on every entry and exit.
          Buy-and-hold enters at the prior close anchoring the first evaluation return
          and pays its entry cost on that first return row. Neither series pays a forced
          liquidation cost at the end of the selected period.
        - Both curves use the same evaluation dates, beginning at the first date with
          a valid lagged strategy position and daily return. Earlier selected dates
          supply moving-average warm-up history.
        - Prices use adjusted closes. The strategy is long-or-flat; cash earns zero.
          Results are historical research output and do not place orders.
        """
    )
