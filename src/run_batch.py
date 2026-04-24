import pandas as pd
import subprocess

df = pd.read_csv("configs/workloads.csv")

for _, row in df.iterrows():
    cmd = [
        "python",
        "src/simulate.py",
        "--p_gain_mult", str(row["p_gain_mult"]),
        "--seed", str(row["seed"]),
        "--duration", "5",
        "--output", row["output"],
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)