# 🌊 Drift-Aware Online Learning (CLI)
**Real-time drift monitoring + lightweight online learning for multivariate environmental sensor streams.**

## 🔗 Live Demo (Dashboard)
- Streamlit demo: `[PASTE_YOUR_LIVE_DEMO_URL_HERE](https://drift-aware.streamlit.app/)`

  ## 🔗 Live Demo (Dashboard)
- Streamlit demo: `PASTE_YOUR_LIVE_DEMO_URL_HERE`  
  *(Example: https://your-app-name.streamlit.app)*

> This repository is **CLI-first** (reproducible runs + saved logs). The live demo link is provided for quick viewing in a browser.

---

## Why this project
Environmental / IoT sensor networks drift over time (seasonality, sensor aging, pollution events). Static models degrade.  
This project demonstrates a **drift-aware streaming pipeline** that:
- monitors drift using **KL divergence** + **Wasserstein distance**
- runs a **lightweight LSTM** as an online predictor/encoder
- flags anomalies using a **calibrated dynamic threshold**
- optionally **adapts online** when drift is detected
- supports a **low-power mode** for edge-like setups
- writes **proof logs** to `data/results.csv`

---

## ✨ Key Features
- **Streaming pipeline**: processes sensor readings step-by-step (simulated real-time)
- **Drift detection**: KL + Wasserstein on rolling windows
- **Online learning**: optional adaptation ON/OFF (compare performance)
- **Low-power mode**: smaller model + less frequent updates
- **Proof outputs**: a reproducible `data/results.csv` log for plots, screenshots, and evidence

---

## 📁 Repository Structure
```txt
DriftAware-OnlineLearning/
├── data/
│   ├── synthetic_environment.csv      # generated (optional)
│   └── results.csv                   # produced by runs (proof)
├── src/
│   ├── generate_data.py              # synthetic stream (drift + anomalies)
│   ├── drift_metrics.py              # KL + Wasserstein
│   ├── online_lstm.py                # lightweight LSTM model
│   ├── online_detection.py           # CLI streaming detector (prints + logs)
│   └── plot_stream.py                # quick plot for drift/anomaly regions
├── requirements.txt
└── .gitignore

## Install
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt


## Quickstart (CLI)
1) Generate a dataset (synthetic drift + anomaly injection)
  python src/generate_data.py
2) Run streaming drift-aware anomaly detection (CLI)
  python src/online_detection.py
3) Plot the drift region (quick sanity check)
  python src/plot_stream.py

## Running on your own CSV (recommended)
Your CSV should include:
** a timestamp column (string or datetime) **
** 2+ numeric sensor columns (e.g., temperature, turbidity, oxygen, salinity)**

### Typical columns
timestamp,temperature,turbidity,oxygen,salinity

## 🔋 Low-Power Mode (Edge-friendly)
Low-power mode is intended for:
  Raspberry Pi / low-power experiments
  reduced update frequency
  smaller LSTM hidden size

In the CLI version, low-power mode is typically controlled via constants at the top of online_detection.py
(e.g., smaller hidden size in online_lstm.py, higher UPDATE_EVERY, smaller reference window).

Recommended low-power settings:
smaller LSTM (hidden=8)
update every 10–20 steps
smaller visible buffers / windows

## Drift Adaptation ON/OFF (comparison for evidence)
A key part of the project is demonstrating improvement when adaptation is enabled.
Adaptation OFF: model error tends to remain higher during drift windows
Adaptation ON: model updates on drift (or periodically) → error stabilizes faster

### Citation
@misc{moses_driftaware_onlinelearning_2025,
  title  = {Drift-Aware Online Learning for Real-Time Environmental Sensor Monitoring (CLI)},
  author = {Moses, Samuel},
  year   = {2025},
  note   = {Code and live demo},
}
