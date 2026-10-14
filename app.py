import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("Apple Stock Price Prediction")
st.write("Machine learning using historical Apple (AAPL) stock data.")

@st.cache_data
def load_data():
    df = pd.read_csv("stock_data.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df = df.reset_index(drop=True)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Return1"] = df["Close"].pct_change()
    df["Target"] = df["Close"].shift(-1)
    df["TargetDate"] = df["Date"].shift(-1)

    return df

try:
    df = load_data()

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "MA5", "MA10", "Return1"
    ]

    data = df.dropna(subset=features + ["Target"]).copy()
    data["NextReturn"] = data["Target"] / data["Close"] - 1

    if len(data) < 100:
        st.error("The dataset needs more historical observations.")
        st.stop()

    split = int(len(data) * 0.8)

    X = data[features]
    y = data["NextReturn"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X.iloc[:split], y.iloc[:split])

    test_predictions = model.predict(X.iloc[split:])
    previous_closes = data["Close"].iloc[split:].to_numpy()
    actual_prices = data["Target"].iloc[split:].to_numpy()
    predicted_prices = previous_closes * (1 + test_predictions)

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    baseline_mae = mean_absolute_error(actual_prices, previous_closes)

    st.subheader("Historical Stock Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Recorded Close", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("Model MAE", f"${mae:.2f}")
    col3.metric("Model RMSE", f"${rmse:.2f}")

    st.subheader("Historical Closing Prices")
    st.line_chart(df.set_index("Date")["Close"])

    st.subheader("Actual vs Predicted Prices")

    results = pd.DataFrame({
        "Actual Price": actual_prices,
        "Predicted Price": predicted_prices
    }, index=data["TargetDate"].iloc[split:])

    st.line_chart(results)

    st.subheader("Baseline Comparison")
    st.write(f"Model MAE: ${mae:.4f}")
    st.write(f"Baseline MAE: ${baseline_mae:.4f}")

    if mae < baseline_mae:
        st.success("The model outperformed the baseline on MAE.")
    else:
        st.warning("The model did not outperform the baseline on MAE.")

    # Train on all labelled historical observations for the final estimate
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)

    latest = df.dropna(subset=features).iloc[[-1]]
    estimated_return = final_model.predict(latest[features])[0]
    estimated_price = latest["Close"].iloc[0] * (1 + estimated_return)

    st.subheader("Next-Observation Price Estimate")
    st.metric("Estimated Closing Price", f"${estimated_price:.2f}")
    st.caption(
        f"Based on data through {df['Date'].iloc[-1].date()}. "
        "This is a historical-data demonstration, not a live market forecast "
        "or financial advice."
    )

except FileNotFoundError:
    st.error("stock_data.csv was not found. Upload it beside app.py.")
except Exception as e:
    st.error(f"Something went wrong: {e}")
