# Drift-Aware Online Learning for Real-Time Environmental Monitoring

This project demonstrates a **drift-aware online learning framework** for multivariate sensor networks (temperature, turbidity, oxygen, salinity).  
It simulates real marine/atmospheric sensor drift and applies:

- **KL divergence drift detection**
- **Wasserstein distance monitoring**
- **Online LSTM encoder** for lightweight representation learning
- **Real-time adaptive updating** under low-power constraints

## Features
- Concept drift identification
- Online learning (streaming model updates)
- Synthetic marine environmental dataset
- Low-power-friendly LSTM encoder architecture
- Drift visualization

## How To Run
```bash
pip install -r requirements.txt
cd src
python generate_data.py
python online_detection.py
