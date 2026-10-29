"""Price history loading with a Yahoo Finance source and local CSV fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class PriceDataError(ValueError):
    """Raised when no usable price history can be loaded."""


@dataclass(frozen=True)
class PriceHistory:
    """A normalized adjusted-close series and its source label."""

    prices: pd.Series
    source: str


def normalize_ticker(ticker: str) -> str:
    """Return a safe uppercase ticker suitable for Yahoo Finance and filenames."""
    normalized = ticker.strip().upper()
    if not re.fullmatch("[A-Z0-9.^=-]{1,20}", normalized):
        raise PriceDataError(
            "Enter a ticker containing only letters, numbers, '.', '^', '=', or '-'."
        )
    return normalized


def _pick_yahoo_close(frame: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(frame, pd.Series):
        return frame
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PriceDataError("Yahoo Finance returned no price rows.")

    choices: list[tuple[int, object]] = []
    for column in frame.columns:
        parts = column if isinstance(column, tuple) else (column,)
        normalized_parts = [
            str(part).strip().lower().replace("_", " ") for part in parts
        ]
        if any(part in {"adj close", "adjusted close"} for part in normalized_parts):
            choices.append((0, column))
        elif any(part == "close" for part in normalized_parts):
            choices.append((1, column))

    if not choices:
        raise PriceDataError("Yahoo Finance data did not include a close price column.")

    selected = frame[min(choices, key=lambda choice: choice[0])[1]]
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[:, 0]
    return selected


def _clean_price_series(
    prices: pd.Series,
    *,
    start: date,
    end: date,
) -> pd.Series:
    if start > end:
        raise PriceDataError("The start date must be on or before the end date.")

    values = pd.to_numeric(prices, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()

    finite = np.isfinite(values.to_numpy(dtype=float))
    valid = (~index.isna()) & finite
    result = pd.Series(
        values.to_numpy(dtype=float)[valid],
        index=index[valid],
        name="Adjusted close",
        dtype=float,
    )
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    result = result.loc[
        (result.index >= pd.Timestamp(start))
        & (result.index <= pd.Timestamp(end))
    ]
    if result.empty:
        raise PriceDataError("No usable prices were found in the selected date range.")
    if (result <= 0).any():
        raise PriceDataError("Price history contains a zero or negative close.")
    return result


def _read_local_csv(path: Path, *, start: date, end: date) -> pd.Series:
    frame = pd.read_csv(path)
    columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in frame.columns
    }
    date_column = columns.get("date")
    price_column = columns.get("adj_close") or columns.get("adjusted_close")
    if date_column is None or price_column is None:
        raise PriceDataError(
            "CSV must contain a 'date' column and an 'adj_close' column."
        )

    index = pd.to_datetime(frame[date_column], errors="coerce")
    prices = pd.Series(frame[price_column].to_numpy(), index=index)
    return _clean_price_series(prices, start=start, end=end)


def load_price_history(
    ticker: str,
    start: date,
    end: date,
    *,
    csv_dir: Path = DEFAULT_DATA_DIR,
    downloader: Callable[..., pd.DataFrame | pd.Series] | None = None,
) -> PriceHistory:
    """Load adjusted daily closes from Yahoo Finance, falling back to data/TICKER.csv.

    The UI's end date is inclusive. Yahoo Finance treats its end parameter as
    exclusive, so one calendar day is added to the download request.
    """
    normalized_ticker = normalize_ticker(ticker)
    if start > end:
        raise PriceDataError("The start date must be on or before the end date.")

    fetch = downloader or yf.download
    yahoo_error = "Yahoo Finance returned no usable rows."
    try:
        downloaded = fetch(
            normalized_ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        yahoo_prices = _pick_yahoo_close(downloaded)
        cleaned = _clean_price_series(yahoo_prices, start=start, end=end)
        return PriceHistory(prices=cleaned, source="Yahoo Finance")
    except Exception as exc:
        yahoo_error = str(exc) or yahoo_error

    fallback_path = Path(csv_dir) / f"{normalized_ticker}.csv"
    if not fallback_path.is_file():
        raise PriceDataError(
            f"Yahoo Finance could not provide {normalized_ticker} "
            f"({yahoo_error}). Local fallback file not found: {fallback_path}"
        )

    try:
        local_prices = _read_local_csv(fallback_path, start=start, end=end)
    except Exception as exc:
        raise PriceDataError(
            f"Yahoo Finance could not provide {normalized_ticker} "
            f"({yahoo_error}). Local CSV {fallback_path.name} is unusable: {exc}"
        ) from exc
    return PriceHistory(
        prices=local_prices,
        source=f"Local CSV · {fallback_path.name}",
    )
