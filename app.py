from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MIN_OBSERVATIONS = 100

TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)",
    "GOOG": "Alphabet (Google)", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "META": "Meta Platforms", "NFLX": "Netflix",
    "AMD": "AMD", "INTC": "Intel", "IBM": "IBM", "ORCL": "Oracle",
    "ADBE": "Adobe", "CRM": "Salesforce", "JPM": "JPMorgan Chase",
    "WMT": "Walmart", "KO": "Coca-Cola", "PEP": "PepsiCo",
}
ALIASES = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL",
           "ALPHABET": "GOOGL", "AMAZON": "AMZN", "NVIDIA": "NVDA",
           "TESLA": "TSLA", "FACEBOOK": "META"}
FEATURES = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "Return1"]

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("Stock Price Prediction Using Machine Learning")
st.caption("Choose a company to explore its historical prices and evaluate next-trading-day estimates.")


def ticker_from_path(path: Path) -> str:
    """Derive a ticker from a filename such as MSFT.csv or MSFT_2010_2024.csv."""
    first = re.split(r"[_\-\s.]", path.stem.strip())[0].upper()
    return ALIASES.get(first, first)


def discover_datasets() -> dict[str, Path]:
    """Find local CSVs; keep the legacy root stock_data.csv as the AAPL dataset."""
    found: dict[str, Path] = {}
    legacy = ROOT / "stock_data.csv"
    if legacy.is_file():
        found["AAPL"] = legacy

    # Additional ticker-named CSVs can be uploaded to the repository root.
    for path in sorted(ROOT.glob("*.csv")):
        if path.name.lower() == "stock_data.csv":
            continue
        ticker = ticker_from_path(path)
        if ticker:
            found[ticker] = path

    # Preferred organized layout: files in data/ override root-level files.
    if DATA_DIR.is_dir():
        for path in sorted(DATA_DIR.glob("*.csv")):
            ticker = ticker_from_path(path)
            if ticker:
                found[ticker] = path
    return found

@st.cache_data(show_spinner=False)
def read_stock_csv(path_string: str, modified_time: float) -> pd.DataFrame:
    """Load and normalize a local CSV; modified_time invalidates cached data after updates."""
    del modified_time  # Used as a cache key only.
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path.name}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("The selected CSV is empty.")

    # Normalize names such as ' adj close ', 'DATE', and 'Trading Date'.
    normalized = {}
    for original in raw.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(original).strip().lower())
        normalized[key] = original

    def source_column(*candidates: str):
        for candidate in candidates:
            key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
            if key in normalized:
                return normalized[key]
        return None

    date_col = source_column("Date", "Trading Date")
    close_col = source_column("Close", "Adj Close", "Adjusted Close")
    if date_col is None or close_col is None:
        raise ValueError("CSV must contain at least a Date column and a Close (or Adj Close) column.")

    output = pd.DataFrame()
    output["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        source = source_column(column)
        if column == "Close":
            source = close_col
        if source is None:
            if column in {"Open", "High", "Low"}:
                output[column] = np.nan
            elif column == "Volume":
                output[column] = 0.0
            else:
                output[column] = np.nan
        else:
            values = raw[source].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
            output[column] = pd.to_numeric(values, errors="coerce")

    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna(subset=["Date", "Close"])
    output = output[output["Close"] > 0]
    output = output.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    for column in ["Open", "High", "Low"]:
        output[column] = output[column].fillna(output["Close"])
    output["Volume"] = output["Volume"].fillna(0).clip(lower=0)

    if len(output) < MIN_OBSERVATIONS:
        raise ValueError(
            f"This dataset has only {len(output)} usable daily rows. "
            f"Please use at least {MIN_OBSERVATIONS} observations."
        )
    if output["Date"].nunique() < MIN_OBSERVATIONS:
        raise ValueError("The dataset does not contain enough unique trading dates.")
    return output


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["MA5"] = frame["Close"].rolling(window=5, min_periods=5).mean()
    frame["MA10"] = frame["Close"].rolling(window=10, min_periods=10).mean()
    frame["Return1"] = frame["Close"].pct_change()
    frame["Target"] = frame["Close"].shift(-1)
    frame["TargetDate"] = frame["Date"].shift(-1)
    frame["NextReturn"] = frame["Target"] / frame["Close"] - 1.0
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=FEATURES + ["Target", "TargetDate", "NextReturn"]).copy()


def candidate_models() -> dict:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


datasets = discover_datasets()
if not datasets:
    st.error("No datasets found. Add one or more company CSV files to the repository's data/ folder.")
    st.stop()

options = sorted(datasets.keys())
company = st.selectbox(
    "Select a company",
    options=options,
    format_func=lambda ticker: f"{TICKER_NAMES.get(ticker, ticker)} ({ticker})",
)
selected_path = datasets[company]

try:
    prices = read_stock_csv(str(selected_path), selected_path.stat().st_mtime)
    features_data = build_features(prices)
    if len(features_data) < MIN_OBSERVATIONS:
        st.error("After feature preparation, fewer than 100 usable rows remain. Choose a longer dataset.")
        st.stop()

    # Chronological train/validation/test split. Test data is not used to choose the model.
    n = len(features_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end < 50 or val_end <= train_end or n - val_end < 20:
        st.error("Not enough data for a reliable chronological train/validation/test split.")
        st.stop()

    X = features_data[FEATURES]
    y_return = features_data["NextReturn"]
    prices_today = features_data["Close"].to_numpy()
    prices_next = features_data["Target"].to_numpy()

    models = candidate_models()
    validation_scores = {}
    for name, candidate in models.items():
        candidate.fit(X.iloc[:train_end], y_return.iloc[:train_end])
        val_returns = candidate.predict(X.iloc[train_end:val_end])
        val_estimates = prices_today[train_end:val_end] * (1.0 + val_returns)
        validation_scores[name] = mean_absolute_error(
            prices_next[train_end:val_end], val_estimates
        )

    # Choose on validation only; final scores are reported on the untouched chronological test set.
    best_name = min(validation_scores, key=validation_scores.get)
    best_model = candidate_models()[best_name]
    best_model.fit(X.iloc[:val_end], y_return.iloc[:val_end])

    test_returns = best_model.predict(X.iloc[val_end:])
    test_actual = prices_next[val_end:]
    test_today = prices_today[val_end:]
    test_predicted = test_today * (1.0 + test_returns)
    test_dates = features_data["TargetDate"].iloc[val_end:]

    model_mae = mean_absolute_error(test_actual, test_predicted)
    model_rmse = float(np.sqrt(mean_squared_error(test_actual, test_predicted)))
    baseline_mae = mean_absolute_error(test_actual, test_today)
    baseline_rmse = float(np.sqrt(mean_squared_error(test_actual, test_today)))
    actual_direction = np.sign(test_actual - test_today)
    predicted_direction = np.sign(test_predicted - test_today)
    direction_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    st.subheader(f"{TICKER_NAMES.get(company, company)} ({company}) — data overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest recorded close", f"${prices['Close'].iloc[-1]:,.2f}")
    col2.metric("Usable daily rows", f"{len(prices):,}")
    col3.metric("Selected model", best_name)
    col4.metric("Test direction accuracy", f"{direction_accuracy:.1f}%")
    st.caption(
        f"Dataset: {selected_path.relative_to(ROOT)} · "
        f"History: {prices['Date'].min().date()} to {prices['Date'].max().date()} · "
        f"Held-out test period: {test_dates.min().date()} to {test_dates.max().date()}"
    )

    if prices["Date"].max().date() < (pd.Timestamp.today().date() - pd.Timedelta(days=45)):
        st.warning(
            "This dataset is older than 45 days. The results are based on historical data, "
            "not current market prices. Update the CSV before presenting a current forecast."
        )

    st.subheader("Historical closing prices")
    st.line_chart(prices.set_index("Date")["Close"], y_label="Closing price (USD)")

    st.subheader("Held-out test results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE", f"${model_mae:.3f}")
    m2.metric("Model RMSE", f"${model_rmse:.3f}")
    m3.metric("Baseline MAE", f"${baseline_mae:.3f}")

    comparison = pd.DataFrame({
        "Actual close": test_actual,
        f"{best_name} estimate": test_predicted,
        "Previous-close baseline": test_today,
    }, index=pd.to_datetime(test_dates))
    comparison.index.name = "Target trading date"
    st.line_chart(comparison, y_label="Closing price (USD)")

    st.subheader("Baseline comparison — what the result means")
    st.write(
        "The baseline predicts that the next trading day's closing price will equal "
        "the current day's closing price. A lower MAE means a smaller average absolute price error."
    )
    if model_mae < baseline_mae:
        improvement = (baseline_mae - model_mae) / baseline_mae * 100 if baseline_mae else 0.0
        st.success(
            f"On this held-out test period, {best_name} had {improvement:.1f}% lower MAE than the baseline. "
            "This is evidence for this dataset and period, not a guarantee of future performance."
        )
    elif np.isclose(model_mae, baseline_mae, rtol=0.01, atol=0.01):
        st.info("On this held-out test period, the selected model and baseline performed similarly on MAE.")
    else:
        gap = (model_mae - baseline_mae) / baseline_mae * 100 if baseline_mae else 0.0
        st.warning(
            f"On this held-out test period, {best_name}'s MAE was {gap:.1f}% higher than the baseline. "
            "The simple baseline was more accurate; this is a valid experimental result, not a website error."
        )

    st.subheader("Validation comparison")
    validation_table = pd.DataFrame({
        "Model": list(validation_scores.keys()),
        "Validation MAE (USD)": list(validation_scores.values()),
    }).sort_values("Validation MAE (USD)").reset_index(drop=True)
    st.dataframe(validation_table, hide_index=True, use_container_width=True)

   
    # Train on all labelled historical data
    final_model = candidate_models()[best_name]
    final_model.fit(X, y_return)

    # Build features using the latest available trading row
    latest_frame = prices.copy()
    latest_frame["MA5"] = latest_frame["Close"].rolling(5).mean()
    latest_frame["MA10"] = latest_frame["Close"].rolling(10).mean()
    latest_frame["Return1"] = latest_frame["Close"].pct_change()

    latest_features = latest_frame.dropna(
        subset=FEATURES
    ).iloc[[-1]]

    latest_return = float(
        final_model.predict(latest_features[FEATURES])[0]
    )

    estimate = float(
        latest_features["Close"].iloc[0] * (1.0 + latest_return)
    )

    last_date = latest_features["Date"].iloc[0].date()

    st.subheader("Next-observation estimate")
    st.metric("Estimated next trading-day close", f"${estimate:,.2f}")
    st.caption(
        f"Based on observations through {last_date}. "
        "This is a model estimate, not a live quote or guaranteed forecast."
    )
except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
    st.error(f"Cannot use the selected dataset: {exc}")
except Exception as exc:
    st.error("The app could not finish this analysis. Check that the selected CSV has valid daily stock-price data.")
    st.exception(exc)

with st.expander("How to add another company"):
    st.markdown(
        """1. Download a CSV for the additional company with daily `Date` and `Close` columns. `Open`, `High`, `Low`, and `Volume` are recommended.

2. Name the file using its ticker, for example `MSFT.csv`, `GOOGL.csv`, or `NVDA.csv`.

3. Upload it to a folder named `data/` in the GitHub repository.

4. Streamlit will discover the file and add the company to this selector after the app redeploys.

Use consistent price conventions within each dataset. Keep the file locally in the repository so the app does not depend on a live third-party API at runtime."""
    )
