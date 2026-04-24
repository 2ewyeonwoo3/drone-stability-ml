import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("data/processed/labels.csv")

# 산점도
plt.figure()

# stable=1 파란색, unstable=0 빨간색
colors = df["stable"].map({1: "blue", 0: "red"})

plt.scatter(df["p_gain_mult"], df["stable"], c=colors)

plt.xlabel("p_gain_mult")
plt.ylabel("stability (0=unstable, 1=stable)")
plt.title("p_gain vs Stability")

plt.grid()

plt.savefig("results/plots/p_gain_vs_stability.png")
plt.show()