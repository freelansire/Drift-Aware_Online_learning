import pandas as pd
import numpy as np

np.random.seed(42)

def generate_environmental_data(n=5000):
    timestamps = pd.date_range(start="2025-01-01", periods=n, freq="min")

    # Base distributions
    temp = np.random.normal(18, 1, n)
    turb = np.random.normal(2.5, 0.3, n)
    oxygen = np.random.normal(7, 0.4, n)
    salinity = np.random.normal(31, 0.6, n)

    # Inject gradual drift (warming trend)
    temp += np.linspace(0, 3, n)

    # Inject sudden drift (pollution event)
    turb[2000:2600] += 1.2
    oxygen[2000:2600] -= 0.9

    # Inject anomalies (fixed: match anomaly vector length to number of indices)
    idx = np.arange(0, n, 420)
    temp[idx] += np.random.normal(8, 1.5, size=len(idx))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temp,
            "turbidity": turb,
            "oxygen": oxygen,
            "salinity": salinity,
        }
    )

    df.to_csv("./data/synthetic_environment.csv", index=False)
    print("Saved synthetic environmental data to ./data/synthetic_environment.csv")

if __name__ == "__main__":
    generate_environmental_data()
