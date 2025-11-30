import sys, os
sys.path.append(os.path.abspath("."))

import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.drift_metrics import kl_divergence, wasserstein


# ------------------------------
# Defaults
# ------------------------------
DEFAULT_FEATURES = ["temperature", "turbidity", "oxygen", "salinity"]
DEFAULT_RESULTS_PATH = "data/results.csv"


# ------------------------------
# Utilities
# ------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def hist_counts(x, bins=30):
    return np.histogram(x, bins=bins)[0].astype(np.float32)

def make_windows(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_lstm(input_dim: int, window: int, low_power: bool):
    import tensorflow as tf  # noqa
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    hidden = 8 if low_power else 16
    proj = 4 if low_power else 8

    model = Sequential([
        LSTM(hidden, return_sequences=False, input_shape=(window, input_dim)),
        Dense(proj, activation="relu"),
        Dense(input_dim),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def append_rows_to_csv(rows, path):
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    out = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    out.to_csv(path, mode="a", header=write_header, index=False)

def reset_csv(path):
    ensure_dir(os.path.dirname(path) or ".")
    if os.path.exists(path):
        os.remove(path)

def generate_synthetic(n=5000):
    """Simple synthetic multi-sensor environmental stream with drift + anomalies."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=n, freq="min")

    temp = np.random.normal(18, 1, n)
    turb = np.random.normal(2.5, 0.3, n)
    oxygen = np.random.normal(7, 0.4, n)
    salinity = np.random.normal(31, 0.6, n)

    # gradual drift
    temp += np.linspace(0, 3, n)

    # sudden drift event
    turb[2000:2600] += 1.2
    oxygen[2000:2600] -= 0.9

    # anomalies (spikes)
    idx = np.arange(0, n, 420)
    temp[idx] += np.random.normal(8, 1.5, len(idx))

    return pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "temperature": temp,
        "turbidity": turb,
        "oxygen": oxygen,
        "salinity": salinity,
    })

def normalise_features(df, features, ref_size):
    ref = df[features].iloc[:ref_size].values.astype(np.float32)
    mu = ref.mean(axis=0)
    sigma = ref.std(axis=0) + 1e-6

    def z(x):
        return (x - mu) / sigma

    return ref, mu, sigma, z


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Drift-Aware Online Learning Demo", layout="wide")
st.title("🌊 Drift-Aware Online Learning")
st.caption("Upload your own sensor CSV or use the built-in synthetic dataset. Live drift metrics (KL/W), LSTM error, anomaly flags, adaptation ON/OFF, and low-power mode.")

st.markdown(
    """
    <style>
      /* Remove underline from links in the sidebar */
      [data-testid="stSidebar"] a {
        text-decoration: none !important;
      }
      /* Optional: also remove underline on hover/focus */
      [data-testid="stSidebar"] a:hover,
      [data-testid="stSidebar"] a:focus {
        text-decoration: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:

    st.sidebar.markdown(
    "👤  \n"
    "[GitHub](https://github.com/freelansire/Drift-Aware_Online_learning)  "
    "[Website](https://freelansire.com)"
)
    
    st.header("Data")
    use_synthetic = st.toggle("Use synthetic demo dataset", value=True)
    uploaded = None
    if not use_synthetic:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.header("Stream settings")
    speed = st.slider("Stream speed (sec/step)", 0.01, 0.30, 0.05, 0.01)
    visible_points = st.slider("Visible points", 150, 1500, 500, 50)

    st.header("Model settings")
    low_power = st.checkbox("🔋 Low-power mode", value=False)
    adapt_on_drift = st.checkbox("🧠 Drift adaptation ON", value=True)
    window = st.slider("Window size", 10, 60, 20, 1)
    ref_size = st.slider("Reference size (calibration)", 200, 1500, 400, 50)
    update_every = st.slider("Online update frequency (steps)", 1, 40, 10 if low_power else 5, 1)

    st.header("Drift thresholds")
    drift_kl_thr = st.slider("KL threshold", 0.2, 3.0, 1.2, 0.1)
    drift_w_thr = st.slider("Wasserstein threshold", 0.5, 5.0, 2.0, 0.1)

    st.header("Logging / Proof")
    write_csv = st.checkbox(f"💾 Write to {DEFAULT_RESULTS_PATH}", value=True)
    flush_every = st.slider("CSV flush every N steps", 1, 200, 25, 1)
    if st.button("🧹 Reset results CSV"):
        reset_csv(DEFAULT_RESULTS_PATH)
        st.success("results.csv cleared.")

    st.divider()
    start = st.button("▶ Start", type="primary")
    stop = st.button("⏹ Stop")


# ------------------------------
# Load data
# ------------------------------
if use_synthetic:
    df = generate_synthetic()
    st.info("Using synthetic demo dataset (drift + anomalies injected). Toggle off to upload your own CSV.")
else:
    if uploaded is None:
        st.warning("Upload a CSV to continue, or toggle on the synthetic demo dataset.")
        st.stop()
    df = pd.read_csv(uploaded)

# Basic cleaning + validation
df.columns = [c.strip() for c in df.columns]
if "timestamp" not in df.columns:
    st.error("Your CSV must include a 'timestamp' column.")
    st.stop()

# Allow user to choose features from available numeric columns
numeric_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
if len(numeric_cols) < 2:
    st.error("Your CSV must include at least 2 numeric sensor columns besides 'timestamp'.")
    st.stop()

features = st.multiselect(
    "Select sensor feature columns",
    options=numeric_cols,
    default=[c for c in DEFAULT_FEATURES if c in numeric_cols] or numeric_cols[:4]
)

if len(features) < 2:
    st.warning("Select at least 2 sensor columns.")
    st.stop()

# Safe bounds
ref_size = min(ref_size, max(250, len(df) // 4))
if len(df) < (ref_size + window + 5):
    st.error(f"Dataset too small. Need at least ~{ref_size + window + 5} rows; got {len(df)}.")
    st.stop()

# Ensure timestamp string for plotting
df["timestamp"] = df["timestamp"].astype(str)


# ------------------------------
# Model setup
# ------------------------------
ref, mu, sigma, z = normalise_features(df, features, ref_size)
ref_z = z(ref)

X_ref, y_ref = make_windows(ref_z, window)
model = build_lstm(len(features), window, low_power)

epochs = 2 if low_power else 3
batch = 64 if low_power else 32
model.fit(X_ref, y_ref, epochs=epochs, batch_size=batch, verbose=0)

pred_ref = model.predict(X_ref, verbose=0)
errs_ref = np.linalg.norm(pred_ref - y_ref, axis=1)
anom_thr = float(errs_ref.mean() + 3.0 * errs_ref.std())

reference_for_drift = ref[:, 0]  # drift computed on first selected feature (interpretable)


# ------------------------------
# Plot placeholders
# ------------------------------
plot_placeholder = st.empty()
metrics = st.columns(6)

st.caption(
    f"Calibration: ref_size={ref_size}, window={window}, features={features}, "
    f"adaptation_on={adapt_on_drift}, low_power={low_power}"
)

# Session log buffer (so you can download even if you don't write CSV)
if "results_rows" not in st.session_state:
    st.session_state.results_rows = []


# ------------------------------
# Streaming loop
# ------------------------------
if start:
    # Buffers
    ts_buf = deque(maxlen=visible_points)
    vals_buf = {f: deque(maxlen=visible_points) for f in features}
    anom_x, anom_y = deque(maxlen=visible_points), deque(maxlen=visible_points)
    drift_x, drift_y = deque(maxlen=visible_points), deque(maxlen=visible_points)

    pending_rows = []
    stream_z = []

    for i in range(ref_size, len(df)):
        if stop:
            st.warning("Stopped.")
            break

        raw_point = df[features].iloc[i].values.astype(np.float32)
        stream_z.append(z(raw_point))

        if len(stream_z) < window:
            continue

        W = np.array(stream_z[-window:], dtype=np.float32).reshape(1, window, len(features))
        pred = model.predict(W, verbose=0)
        actual = W[:, -1, :]
        err = float(np.linalg.norm(pred - actual))

        # Drift metrics on first selected feature
        recent = df[features[0]].iloc[i-ref_size:i].values
        kl = float(kl_divergence(hist_counts(reference_for_drift), hist_counts(recent)))
        w = float(wasserstein(reference_for_drift, recent))
        drift_flag = (kl > drift_kl_thr) or (w > drift_w_thr)

        anomaly_flag = err > anom_thr

        # Optional adaptation
        if adapt_on_drift and ((i % update_every) == 0 or drift_flag):
            model.fit(W, actual, epochs=1, verbose=0)

        # Push to buffers for plotting (raw for readability)
        ts = df["timestamp"].iloc[i]
        ts_buf.append(ts)
        for j, f in enumerate(features):
            vals_buf[f].append(float(raw_point[j]))

        if anomaly_flag:
            anom_x.append(ts)
            anom_y.append(float(raw_point[0]))  # mark on first feature trace
        if drift_flag:
            drift_x.append(ts)
            drift_y.append(float(raw_point[1] if len(features) > 1 else raw_point[0]))  # mark on second if exists

        row = {
            "timestamp": ts,
            **{f: float(raw_point[j]) for j, f in enumerate(features)},
            "err": err,
            "anom_threshold": anom_thr,
            "kl": kl,
            "wasserstein": w,
            "drift_flag": bool(drift_flag),
            "anomaly_flag": bool(anomaly_flag),
            "adaptation_on": bool(adapt_on_drift),
            "low_power_mode": bool(low_power),
            "window": int(window),
            "ref_size": int(ref_size),
            "update_every": int(update_every),
        }

        pending_rows.append(row)
        st.session_state.results_rows.append(row)

        # Flush CSV periodically
        if write_csv and len(pending_rows) >= flush_every:
            append_rows_to_csv(pending_rows, DEFAULT_RESULTS_PATH)
            pending_rows = []

        # Plot
        fig = go.Figure()
        for f in features:
            fig.add_trace(go.Scatter(x=list(ts_buf), y=list(vals_buf[f]), mode="lines", name=f))

        if len(anom_x) > 0:
            fig.add_trace(go.Scatter(x=list(anom_x), y=list(anom_y), mode="markers",
                                     name="anomaly", marker_symbol="x"))

        if len(drift_x) > 0:
            fig.add_trace(go.Scatter(x=list(drift_x), y=list(drift_y), mode="markers",
                                     name="drift", marker_symbol="diamond"))

        fig.update_layout(
            height=560,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h"),
        )
        plot_placeholder.plotly_chart(fig, use_container_width=True)

        # Metrics row
        metrics[0].metric("Err", f"{err:.2f}")
        metrics[1].metric("Anom thr", f"{anom_thr:.2f}")
        metrics[2].metric("KL", f"{kl:.2f}")
        metrics[3].metric("Wasserstein", f"{w:.2f}")
        metrics[4].metric("Drift", str(drift_flag))
        metrics[5].metric("Anomaly", str(anomaly_flag))

        time.sleep(speed)

    # final flush
    if write_csv and pending_rows:
        append_rows_to_csv(pending_rows, DEFAULT_RESULTS_PATH)

    st.success("Stream finished.")
else:
    st.info("Click ▶ Start to run the live stream. Toggle adaptation ON/OFF to compare behavior.")


# ------------------------------
# Downloads / Proof
# ------------------------------
st.subheader("📦 Proof / Outputs")

col1, col2 = st.columns(2)

with col1:
    if write_csv:
        st.write(f"Local log file: `{DEFAULT_RESULTS_PATH}`")
        if os.path.exists(DEFAULT_RESULTS_PATH):
            st.download_button(
                "⬇ Download data/results.csv",
                data=open(DEFAULT_RESULTS_PATH, "rb").read(),
                file_name="results.csv",
                mime="text/csv"
            )

with col2:
    if st.session_state.results_rows:
        df_mem = pd.DataFrame(st.session_state.results_rows)
        st.download_button(
            "⬇ Download session log (CSV)",
            data=df_mem.to_csv(index=False).encode("utf-8"),
            file_name="session_results.csv",
            mime="text/csv"
        )
        st.caption(f"Session rows: {len(df_mem):,}")
