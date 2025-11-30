# stream_demo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from drift_metrics import kl_divergence, wasserstein
from online_lstm import build_online_lstm, make_window

# --- Config ---
FEATURES = ["temperature", "turbidity", "oxygen", "salinity"]
WINDOW = 20
REF_SIZE = 200
STEP = 5  # how many samples to skip per update (for speed)

def main():
    df = pd.read_csv("./data/synthetic_environment.csv")
    model = build_online_lstm(len(FEATURES))

    reference = df[FEATURES].iloc[:REF_SIZE].values
    buffer = []

    # plotting setup
    plt.ion()
    fig, (ax_temp, ax_drift) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    times = []
    temps = []
    drift_flags = []
    anomaly_flags = []
    kl_values = []
    w_values = []

    for i in range(REF_SIZE, len(df), STEP):
        row = df.iloc[i]
        point = row[FEATURES].values
        buffer.append(point)

        window = make_window(buffer, WINDOW)
        if window is None:
            continue

        # predict next sensor state
        pred = model.predict(window, verbose=0)
        err = np.linalg.norm(pred - window[:, -1, :])

        # compute drift (temperature-based for simplicity)
        p = reference[:, 0]
        q = df["temperature"].iloc[i-REF_SIZE:i].values

        hist_p, _ = np.histogram(p, bins=20, density=True)
        hist_q, _ = np.histogram(q, bins=20, density=True)

        kl = kl_divergence(hist_p, hist_q)
        w = wasserstein(p, q)

        drift = (kl > 0.6) or (w > 1.0)
        anomaly = err > 3.5

        # online adaptation
        if drift:
            model.fit(window, window[:, -1, :], epochs=1, verbose=0)

        # store for plotting
        times.append(row["timestamp"])
        temps.append(row["temperature"])
        kl_values.append(kl)
        w_values.append(w)
        drift_flags.append(drift)
        anomaly_flags.append(anomaly)

        # --- update plots ---
        ax_temp.clear()
        ax_drift.clear()

        # top: temperature + anomaly markers
        ax_temp.plot(times, temps, label="Temperature")
        # anomalies in red
        anomaly_times = [t for t, a in zip(times, anomaly_flags) if a]
        anomaly_temps = [v for v, a in zip(temps, anomaly_flags) if a]
        ax_temp.scatter(anomaly_times, anomaly_temps, marker="x", label="Anomaly")

        ax_temp.set_ylabel("Temperature")
        ax_temp.set_title("Drift-Aware Online Learning – Live Demo")
        ax_temp.legend()

        # bottom: drift metrics over time
        ax_drift.plot(times, kl_values, label="KL divergence")
        ax_drift.plot(times, w_values, label="Wasserstein")
        # highlight drift periods
        drift_times = [t for t, d in zip(times, drift_flags) if d]
        drift_kl = [v for v, d in zip(kl_values, drift_flags) if d]
        ax_drift.scatter(drift_times, drift_kl, marker="o", label="Drift flagged")

        ax_drift.set_ylabel("Drift metrics")
        ax_drift.set_xlabel("Time")
        ax_drift.legend()

        plt.tight_layout()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
