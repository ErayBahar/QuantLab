# QuantLab

QuantLab is a small Streamlit dashboard for exploring a simple moving-average
crossover against buy-and-hold. It loads adjusted daily prices from Yahoo Finance,
falls back to a local CSV, and calculates returns without sending orders.

## Requirements and launch

Use Python 3.10 or newer. From the project directory, create an environment and
install dependencies:

    python -m venv .venv
    python -m pip install -r requirements.txt

Launch the dashboard with:

    streamlit run app.py

Run the automated calculation tests with:

    pytest -q

## Local CSV fallback

When Yahoo Finance is unavailable or returns no usable prices, QuantLab checks for
<code>data/&lt;TICKER&gt;.csv</code>, using the uppercase ticker in the filename.
A fallback file must contain:

- <code>date</code>: daily market date, parseable as a date.
- <code>adj_close</code>: adjusted closing price as a positive number.

The chosen start and end dates are inclusive. Yahoo Finance is requested with
adjusted prices and its exclusive end date is extended by one calendar day to
include the selected end date.

## Strategy and execution assumptions

The short and long simple moving averages require their complete configured
windows. No signal is emitted until both averages have a value. The signal is
long when the short average is greater than the long average and flat otherwise.
The strategy does not short.

For close C[t], the daily close-to-close return is
C[t] / C[t-1] - 1. The position used for that return is the signal formed at the
previous close: position[t] = signal[t-1]. This one-row lag prevents the close
at t from generating a signal that is then used for the return ending at that
same close. The signal series begins only after both full moving-average
windows are available.

This convention still implies an idealized fill at the previous close, after
calculating a signal from that close. A daily close alone cannot guarantee such
a fill. QuantLab does not model an actual order or fill price, next-open
execution, intraday price movement, bid/ask spreads, slippage, taxes, partial
fills, market impact, borrow costs, or interest on cash. A more execution-aware
study should use suitable open or intraday data and explicitly model when orders
could be submitted and filled.

Transaction cost is a configurable one-way percentage of traded notional. The
strategy pays it on each change between flat and long. Buy-and-hold is entered
at the prior close anchoring the first evaluation return and pays one entry cost
on that first return row. No forced end-of-period sale is modeled. Net daily
returns are compounded to form each equity curve, and drawdown is measured from
the running peak, including initial capital.

Each fee is debited from pre-trade equity at the prior close before the next
close-to-close market return is applied. For a return interval, the net growth
factor is the market growth factor multiplied by one minus the fee fraction.

Both performance series use identical evaluation dates. The evaluation starts
with the first date that has both a prior-close signal and a daily return.
Selected dates before that point are used to warm up the moving averages and are
visible on the price chart. Buy-and-hold and the strategy use the same return
dates, with buy-and-hold's initial position anchored at the prior close for the
first evaluation return.

## Portfolio analytics assumptions

The Portfolio Analytics page loads adjusted closes for every selected ticker
through Yahoo Finance, using the existing per-ticker local CSV fallback when
needed. If any selected ticker cannot be loaded, the portfolio calculation
stops and identifies that ticker; it does not silently drop an asset or alter
the requested weights.

The calculation joins adjusted prices on dates shared by all assets and does
not forward-fill missing quotes. A date missing any asset is omitted for every
asset, then returns are calculated between consecutive shared price dates. At
least three shared price dates are required. Sparse overlap and intervals that
span more than one weekday are flagged because an interval may cover multiple
market sessions; exchange-specific holiday calendars are not modeled.

Configured weights are long-only and must sum to 100% within a small
floating-point tolerance. The portfolio is rebalanced to those weights at each
shared valuation close for the next interval. The equal-weight benchmark is
rebalanced on the same dates. Both results are gross of transaction costs,
slippage, taxes, and cash interest.

Annualized volatility is sample standard deviation of the shared-interval
returns multiplied by the square root of 252. The app warns when fewer than 60
shared return intervals are available or when common dates are irregular. The
252-interval assumption may be approximate in those cases. Correlations use
Pearson correlation on the same complete-case return matrix; undefined values,
such as correlations for a constant-return asset, are shown as n/a.

All selected assets are assumed to have the same quote currency. QuantLab does
not convert prices or returns using foreign-exchange rates, so mixed-currency
portfolios do not represent base-currency performance.

## Project layout

- <code>app.py</code> — Streamlit controls, charts, and methodology notes.
- <code>pages/1_Portfolio_Analytics.py</code> — Portfolio analytics inputs and
  charts, while preserving the existing single-stock page.
- <code>quantlab/data.py</code> — Yahoo Finance loading, adjusted-price normalization, and
  CSV fallback.
- <code>quantlab/backtest.py</code> — Moving-average signals, backtest returns, transaction
  costs, cumulative equity, and drawdown metrics.
- <code>quantlab/portfolio.py</code> — Portfolio weights, shared-date returns, benchmarks,
  volatility, drawdown, and correlations.
- <code>tests/</code> — Deterministic tests for price-source fallback, data sufficiency,
  signal timing, portfolio weights and dates, returns, risk metrics, and correlation.
- <code>data/</code> — Optional local fallback CSV files.

