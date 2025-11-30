import pandas as pd
import numpy as np

from drift_metrics import kl_divergence, wasserstein
from online_lstm import build_online_lstm, make_window

df = pd.read_csv("./data/synthetic_environment.csv")

FEATURES = ["temperature", "turbidity", "oxygen", "salinity"]
WINDOW = 20
REF_SIZE = 400          # bigger reference improves stability
UPDATE_EVERY = 5        # train every N steps (lightweight)

# --- reference data ---
ref = df[FEATURES].iloc[:REF_SIZE].values.astype(np.float32)

# --- standardise using reference statistics (critical) ---
mu = ref.mean(axis=0)
sigma = ref.std(axis=0) + 1e-6

def z(x):
    return (x - mu) / sigma

ref_z = z(ref)

# --- build model ---
model = build_online_lstm(len(FEATURES))

# --- pretrain quickly on reference windows ---
X = []
y = []
for i in range(WINDOW, len(ref_z)):
    X.append(ref_z[i-WINDOW:i])
    y.append(ref_z[i])
X = np.array(X)
y = np.array(y)

# Small pretrain: makes errors meaningful
model.fit(X, y, epochs=3, batch_size=32, verbose=0)

# --- calibrate anomaly threshold from reference prediction errors ---
pred_ref = model.predict(X, verbose=0)
errs_ref = np.linalg.norm(pred_ref - y, axis=1)
ANOM_THRESH = float(errs_ref.mean() + 3.0 * errs_ref.std())  # dynamic threshold

reference_temp = ref[:, 0]  # unscaled temperature for drift tests
stream_buffer = []

for i in range(REF_SIZE, len(df)):
    point = df[FEATURES].iloc[i].values.astype(np.float32)
    stream_buffer.append(z(point))

    window = make_window(stream_buffer, WINDOW)
    if window is None:
        continue

    # Predict expected next reading (standardised)
    pred = model.predict(window, verbose=0)          # shape: (1, input_dim)
    actual = window[:, -1, :]                        # shape: (1, input_dim)
    err = float(np.linalg.norm(pred - actual))       # scalar

    # Drift metrics (use recent window vs reference temperature distribution)
    q = df["temperature"].iloc[i-REF_SIZE:i].values
    kl = kl_divergence(np.histogram(reference_temp, 30)[0], np.histogram(q, 30)[0])
    w  = wasserstein(reference_temp, q)
    drift_flag = (kl > 1.2) or (w > 2.0)

    anomaly_flag = err > ANOM_THRESH

    print(
        f"[{df['timestamp'][i]}] "
        f"Drift={drift_flag} | KL={kl:.2f} | W={w:.2f} | "
        f"Err={err:.2f} (thr={ANOM_THRESH:.2f}) | Anomaly={anomaly_flag}"
    )

    # Lightweight online update (keeps it stable even without drift)
    if (i % UPDATE_EVERY) == 0 or drift_flag:
        model.fit(window, actual, epochs=1, verbose=0)
