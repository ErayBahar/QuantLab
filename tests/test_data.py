from datetime import date

import pandas as pd

from quantlab.data import load_price_history


def test_yahoo_loader_uses_adjusted_close_and_inclusive_end_date() -> None:
    request: dict[str, object] = {}

    def download(ticker: str, **kwargs) -> pd.DataFrame:
        request.update({"ticker": ticker, **kwargs})
        return pd.DataFrame(
            {("Close", "AAPL"): [100.0, 101.0, 102.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )

    history = load_price_history(
        "aapl",
        date(2025, 1, 2),
        date(2025, 1, 3),
        downloader=download,
    )

    assert history.source == "Yahoo Finance"
    assert history.prices.tolist() == [101.0, 102.0]
    assert request["ticker"] == "AAPL"
    assert request["auto_adjust"] is True
    assert request["end"] == "2025-01-04"


def test_local_csv_is_used_when_yahoo_download_fails(tmp_path) -> None:
    (tmp_path / "AAPL.csv").write_text(
        "date,adj_close\n2025-01-01,100\n2025-01-02,101\n",
        encoding="utf-8",
    )

    def offline(*args, **kwargs):
        raise RuntimeError("offline")

    history = load_price_history(
        "AAPL",
        date(2025, 1, 1),
        date(2025, 1, 2),
        csv_dir=tmp_path,
        downloader=offline,
    )

    assert history.source == "Local CSV · AAPL.csv"
    assert history.prices.tolist() == [100.0, 101.0]
