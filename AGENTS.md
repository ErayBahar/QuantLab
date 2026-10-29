# Repository Guidelines

## Project Structure & Module Organization

The single-stock Streamlit app lives in `app.py`; additional pages belong in `pages/`. Keep reusable calculations in `quantlab/backtest.py` and `quantlab/portfolio.py`, and price loading, normalization, and CSV fallback logic in `quantlab/data.py`. Add automated tests under `tests/`. Optional local price files belong in `data/` and use `<TICKER>.csv` with `date,adj_close` columns. Dependencies are listed in `requirements.txt`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt` installs runtime and test dependencies.
- `streamlit run app.py` starts the dashboard from the repository root.
- `python -m pytest -q` runs the automated test suite.

## Coding Style & Naming Conventions

Use Python 3.10 or newer, four spaces for indentation, `snake_case` for functions and variables, and descriptive type hints. Keep the main UI in `app.py` and page-specific UI in `pages/`; keep calculation and data functions independently importable. Preserve the long-window warm-up and one-period signal lag in the backtest. Keep buy-and-hold and strategy comparisons on the same evaluation dates.

## Testing Guidelines

Use pytest with files named `test_*.py`. Prefer small deterministic price series and temporary CSV files; tests should not depend on live Yahoo Finance data. Cover signal timing, insufficient history, portfolio weights, common-date alignment, returns, volatility, drawdown, correlation, transaction costs, and the local CSV fallback. After code changes, run `python -m pytest -q` and relevant targeted checks. Report the commands run and their actual results; disclose failures or checks not run.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects, optionally prefixed by a type, such as `fix: align evaluation dates` or `docs: explain execution assumptions`. Pull requests should summarize the change, list test results, link a related issue when applicable, and include a screenshot for visible dashboard changes. Get explicit human authorization before committing, pushing, merging, or deploying.

## Security & Configuration

Do not commit virtual environments, secrets, or credentials. This is a research dashboard; changes should not introduce brokerage credentials or order execution.
