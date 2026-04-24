import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/labels.csv")

x = df["p_gain_mult"]
y = df["stable"]

# jitter
x_jitter = x + np.random.normal(0, 0.02, size=len(x))

# scatter
colors = y.map({0: "red", 1: "blue"})
plt.scatter(x_jitter, y, c=colors, alpha=0.6, label="samples")

# 평균선
grouped = df.groupby("p_gain_mult")["stable"].mean()
plt.plot(grouped.index, grouped.values, color="black", marker="o", label="mean stability")

plt.xlabel("p_gain_mult")
plt.ylabel("stability (0=unstable, 1=stable)")
plt.title("p_gain vs Stability")

plt.legend()
plt.grid()

plt.savefig("results/plots/p_gain_vs_stability.png")
plt.show()