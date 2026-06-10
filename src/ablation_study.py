"""
ablation_study.py — Stage 2-B 조기경보 모델 Ablation 검증.

목적: "어떤 feature 조합이 성능에 기여하는가"를 체계적으로 검증.
      RF PR-AUC=0.997이 특정 feature에 의존하는지, 혹은 R1~R6 조합이 시너지를 만드는지 규명.

실험 설계:
  A. Feature 단일 제거 (leave-one-out): 각 feature를 하나씩 빼고 성능 측정
  B. Feature 단일 사용 (each-only): 각 feature만 단독으로 사용 시 성능
  C. 신호 그룹별: 위치(R1,R6) vs 자세(R2,R3,R4) vs 이벤트(R5) 그룹 비교
  D. Scenario 제거: S1만 / S2만 / S1+S2 비교 → S2 추가의 기여도

결과: results/ablation/ablation_report.csv + ablation_plot.png

실행:
  python src/ablation_study.py
"""

from __future__ import annotations

import argparse
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLS = [
    "alt_rmse_val",      # R1
    "tilt_max_val",      # R2
    "ang_rate_rms_val",  # R3
    "vib_ratio_val",     # R4
    "crash_val",         # R5
    "conv_fail_val",     # R6
]
FEAT_LABELS = ["R1(alt)", "R2(tilt)", "R3(ang)", "R4(vib)", "R5(crash)", "R6(conv)"]
GROUP_COL   = "run_id"
RANDOM_STATE = 42


def train_eval(df: pd.DataFrame, features: list[str],
               scenario_filter: str | None = None) -> dict:
    """지정된 feature 조합으로 RF 학습 후 PR-AUC, F1 반환."""
    if scenario_filter:
        df = df[df["scenario"] == scenario_filter].copy()

    if len(features) == 0 or (df["label"] == 1).sum() == 0:
        return {"pr_auc": 0.0, "f1": 0.0, "n_pos": 0, "n_neg": 0}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    try:
        train_idx, test_idx = next(
            gss.split(df, df["label"], df[GROUP_COL]))
    except ValueError:
        return {"pr_auc": 0.0, "f1": 0.0, "n_pos": 0, "n_neg": 0}

    X_tr = df.iloc[train_idx][features].to_numpy()
    y_tr = df.iloc[train_idx]["label"].to_numpy()
    X_te = df.iloc[test_idx][features].to_numpy()
    y_te = df.iloc[test_idx]["label"].to_numpy()

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = rf.predict(X_te)

    return {
        "pr_auc": average_precision_score(y_te, y_prob),
        "f1":     f1_score(y_te, y_pred, zero_division=0),
        "n_pos":  int((df["label"] == 1).sum()),
        "n_neg":  int((df["label"] == 0).sum()),
    }


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    baseline = train_eval(df, FEATURE_COLS)
    records.append({"실험": "Baseline (R1~R6 전체)", "features": "All",
                    **baseline, "pr_auc_drop": 0.0})
    print(f"[Baseline]       PR-AUC={baseline['pr_auc']:.4f}  F1={baseline['f1']:.4f}")

    # A. Leave-one-out
    print("\n[A] Feature 단일 제거 (leave-one-out)")
    for i, feat in enumerate(FEATURE_COLS):
        sub = [f for f in FEATURE_COLS if f != feat]
        res = train_eval(df, sub)
        drop = baseline["pr_auc"] - res["pr_auc"]
        records.append({"실험": f"A: -{FEAT_LABELS[i]}", "features": f"All-{FEAT_LABELS[i]}",
                        **res, "pr_auc_drop": drop})
        print(f"  -{FEAT_LABELS[i]:12s}: PR-AUC={res['pr_auc']:.4f}  drop={drop:+.4f}")

    # B. Each-only
    print("\n[B] Feature 단독 사용 (each-only)")
    for i, feat in enumerate(FEATURE_COLS):
        res = train_eval(df, [feat])
        drop = baseline["pr_auc"] - res["pr_auc"]
        records.append({"실험": f"B: {FEAT_LABELS[i]}만", "features": FEAT_LABELS[i],
                        **res, "pr_auc_drop": drop})
        print(f"  {FEAT_LABELS[i]:12s}만: PR-AUC={res['pr_auc']:.4f}  drop={drop:+.4f}")

    # C. 그룹별
    print("\n[C] 신호 그룹별")
    groups = {
        "위치 그룹 (R1+R6)":    ["alt_rmse_val", "conv_fail_val"],
        "자세 그룹 (R2+R3+R4)": ["tilt_max_val", "ang_rate_rms_val", "vib_ratio_val"],
        "이벤트 그룹 (R5)":      ["crash_val"],
        "위치+자세 (R1~R4+R6)": [f for f in FEATURE_COLS if f != "crash_val"],
    }
    for name, feats in groups.items():
        res = train_eval(df, feats)
        drop = baseline["pr_auc"] - res["pr_auc"]
        records.append({"실험": f"C: {name}", "features": str([f.replace('_val','') for f in feats]),
                        **res, "pr_auc_drop": drop})
        print(f"  {name}: PR-AUC={res['pr_auc']:.4f}  drop={drop:+.4f}")

    # D. Scenario 기여도
    print("\n[D] Scenario 기여도 (S1 vs S2 vs 전체)")
    for sc_name in ["S1", "S2", None]:
        label = f"D: {sc_name or 'S1+S2(전체)'}"
        res = train_eval(df, FEATURE_COLS, scenario_filter=sc_name)
        drop = baseline["pr_auc"] - res["pr_auc"]
        records.append({"실험": label, "features": "All",
                        **res, "pr_auc_drop": drop})
        print(f"  {label}: PR-AUC={res['pr_auc']:.4f}  "
              f"pos={res['n_pos']} neg={res['n_neg']}  drop={drop:+.4f}")

    return pd.DataFrame(records)


def plot_ablation(result_df: pd.DataFrame, out_path: Path):
    """Ablation 결과 시각화 — PR-AUC drop 막대그래프."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # A. Leave-one-out drop
    ax = axes[0]
    loo = result_df[result_df["실험"].str.startswith("A:")].copy()
    loo["label"] = loo["실험"].str.replace("A: -", "")
    colors = ["crimson" if d > 0.01 else "steelblue" for d in loo["pr_auc_drop"]]
    ax.barh(loo["label"], loo["pr_auc_drop"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("PR-AUC Drop (제거 후 기준 대비)")
    ax.set_title("A. Leave-One-Out: 어떤 feature가 중요한가\n(높을수록 중요)")
    ax.grid(True, alpha=0.3, axis="x")

    # B. Each-only PR-AUC
    ax = axes[1]
    eo = result_df[result_df["실험"].str.startswith("B:")].copy()
    eo["label"] = eo["실험"].str.replace("B: ", "").str.replace("만", "")
    baseline_val = result_df[result_df["실험"] == "Baseline (R1~R6 전체)"]["pr_auc"].iloc[0]
    bar_colors = ["darkorange" if v >= 0.8 else "lightgray" for v in eo["pr_auc"]]
    ax.barh(eo["label"], eo["pr_auc"], color=bar_colors)
    ax.axvline(baseline_val, color="red", ls="--", lw=1.5, label=f"Baseline={baseline_val:.3f}")
    ax.set_xlabel("PR-AUC (단독 사용 시)")
    ax.set_title("B. Each-Only: feature 단독 예측력")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Ablation Study — 조기경보 모델 Feature 기여도 분석 (H=3.5s)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--labels",  default="data/processed/labels.parquet")
    parser.add_argument("--out-dir", default="results/ablation")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"데이터 로드: {args.labels}")
    df = pd.read_parquet(args.labels)
    df = df[df["label"] >= 0].copy()
    print(f"총 {len(df):,}행 / {df['run_id'].nunique()} runs / "
          f"양성:{(df['label']==1).sum():,} 음성:{(df['label']==0).sum():,}")

    print("\n" + "="*60)
    print("Ablation Study 시작 (H=3.5s, RF)")
    print("="*60)

    result_df = run_ablation(df)

    csv_path = Path(args.out_dir) / "ablation_report.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    plot_ablation(result_df, Path(args.out_dir) / "ablation_plot.png")

    print("\n" + "="*60)
    print("Ablation 요약")
    print("="*60)
    print(result_df[["실험", "pr_auc", "f1", "pr_auc_drop"]].to_string(index=False))


if __name__ == "__main__":
    main()