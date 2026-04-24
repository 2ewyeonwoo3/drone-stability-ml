import pandas as pd
import subprocess

df = pd.read_csv("configs/workloads.csv")

for i, row in df.iterrows():
    cmd = [
        "python",
        "src/simulate.py",
        "--p_gain_mult", str(row["p_gain_mult"]),
        "--output", row["output"]
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)