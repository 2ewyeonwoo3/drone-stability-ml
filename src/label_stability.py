import glob
import os
import pandas as pd


def label_file(path: str) -> dict:
    df = pd.read_csv(path)

    p_gain_mult = float(df["p_gain_mult"].iloc[0])

    z_mean = df["z"].mean()
    z_min = df["z"].min()
    max_abs_roll = df["roll"].abs().max()
    max_abs_pitch = df["pitch"].abs().max()

    # 아주 단순한 1차 기준
    # z 평균이 너무 낮으면 안정적으로 떠 있지 못한 것으로 판단
    stable = 1 if z_mean >= 0.5 else 0

    return {
        "file": os.path.basename(path),
        "p_gain_mult": p_gain_mult,
        "z_mean": z_mean,
        "z_min": z_min,
        "max_abs_roll": max_abs_roll,
        "max_abs_pitch": max_abs_pitch,
        "stable": stable,
    }


def main():
    files = sorted(glob.glob("results/logs/run_p*.csv"))
    rows = [label_file(f) for f in files]

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/labels.csv"

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved labels to {out_path}")


if __name__ == "__main__":
    main()