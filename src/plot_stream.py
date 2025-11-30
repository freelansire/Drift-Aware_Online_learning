import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/synthetic_environment.csv")

plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["temperature"], label="Temperature")
plt.axvspan(df.index[2000], df.index[2600], alpha=0.15, color='red', label="Drift Event")
plt.title("Real-Time Drift in Environmental Temperature")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
