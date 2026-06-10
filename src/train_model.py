"""
train_model.py — Stage 2 조기경보 분류기 학습 및 평가.

★ 핵심 함정 반영 ★
① 라벨 불균형: class_weight='balanced', 평가는 PR-AUC·F1(accuracy 금지)
② 데이터 누수: train/test 반드시 run 단위 분할 (GroupShuffleSplit)
③ 조기경보 특화 평가: 윈도우 분류 지표 + run당 lead time 분포

모델: LogisticRegression(baseline) + RandomForest 비교
Feature: R1~R6 값 (alt_rmse, tilt_max, ang_rate_rms, vib_ratio, crash, conv_fail)

실행:
  python src/train_model.py
  python src/train_model.py --labels data/processed/labels.parquet
  python src/train_model.py --no-plot  # 그래프 없이 숫자만
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 상수 ──────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "alt_rmse_val",     # R1: 고도/추종오차 RMSE
    "tilt_max_val",     # R2: 최대 자세각
    "ang_rate_rms_val", # R3: 각속도 RMS
    "vib_ratio_val",    # R4: 진동 비율 (FFT)
    "crash_val",        # R5: 추락 지시
    "conv_fail_val",    # R6: 수렴 실패
]
TARGET_COL   = "label"
GROUP_COL    = "run_id"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def split_by_run(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    run 단위로 train/test 분할 (GroupShuffleSplit).
    데이터 누수 방지: 한 run의 모든 윈도우는 한쪽에만.
    payload_factor·scenario 분포 검증 포함.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=RANDOM_STATE)
    groups = df[GROUP_COL].values
    train_idx, test_idx = next(gss.split(df, df[TARGET_COL], groups))

    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]

    print(f"\n[Train/Test 분할] run 단위 GroupShuffleSplit (test={test_size:.0%})")
    print(f"  train: {len(train_df):,}행 / {train_df[GROUP_COL].nunique()} runs")
    print(f"  test : {len(test_df):,}행 / {test_df[GROUP_COL].nunique()} runs")

    # 분포 검증 — payload_factor가 양쪽에 골고루 들어갔는지
    print("\n  [payload_factor 분포 검증]")
    for col in ["payload_factor", "scenario"]:
        if col in df.columns:
            tr = train_df.drop_duplicates(GROUP_COL)[col].value_counts(normalize=True)
            te = test_df.drop_duplicates(GROUP_COL)[col].value_counts(normalize=True)
            print(f"    {col}: train={dict(round(tr,2))}, test={dict(round(te,2))}")

    return train_df, test_df


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> dict:
    """윈도우 단위 분류 지표 계산."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*55}")
    print(f"[{model_name}] 윈도우 단위 분류 지표")
    print(f"{'='*55}")
    print(f"  PR-AUC  : {pr_auc:.4f}  (주 평가지표)")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  F1(양성): {f1:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["정상(0)", "위험임박(1)"], zero_division=0))

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1,
            "y_prob": y_prob, "y_pred": y_pred}


def evaluate_lead_time(model, test_df: pd.DataFrame,
                       X_test: np.ndarray, model_name: str):
    """
    조기경보 특화 평가: run당 lead time 측정.
    각 발산 run에서 모델이 t_fail보다 몇 초 먼저 첫 양성 경보를 냈는가.
    이게 조기경보의 실제 가치를 나타내는 핵심 지표.
    """
    test_df = test_df.copy()
    test_df["y_prob"] = model.predict_proba(X_test)[:, 1]
    test_df["y_pred"] = model.predict(X_test)

    # t_fail이 있는 run만 (발산 run)
    fail_runs = test_df[test_df["t_fail"].notna()].copy()
    if len(fail_runs) == 0:
        print(f"\n[{model_name}] lead time: 발산 run 없음")
        return

    # 각 run에서 첫 번째 양성 경보 시점
    first_alarm = (fail_runs[fail_runs["y_pred"] == 1]
                   .groupby("run_id")["window_end"].min()
                   .reset_index()
                   .rename(columns={"window_end": "first_alarm_t"}))

    t_fail_per_run = (fail_runs.drop_duplicates("run_id")
                     [["run_id", "t_fail", "payload_factor", "scenario"]])

    lead_df = t_fail_per_run.merge(first_alarm, on="run_id", how="left")
    lead_df["lead_time"] = lead_df["t_fail"] - lead_df["first_alarm_t"]
    # 경보 못 낸 run은 lead_time = NaN
    n_total   = len(lead_df)
    n_alarmed = lead_df["lead_time"].notna().sum()

    print(f"\n[{model_name}] 조기경보 lead time (발산 run {n_total}개 중 {n_alarmed}개 경보)")
    print(lead_df["lead_time"].describe().to_string())
    print(f"\n  경보율: {100*n_alarmed/n_total:.1f}%")
    if n_alarmed > 0:
        mean_lt = lead_df["lead_time"].mean()
        print(f"  평균 lead time: {mean_lt:.2f}초")
        if mean_lt > 1.5:
            print(f"  ✅ 평균 {mean_lt:.1f}초 전 경보 — 유의미한 조기경보")
        else:
            print(f"  ⚠️  평균 lead time이 짧음({mean_lt:.1f}초) — '추락 직전 감지'에 가까움")

    # payload_factor별 lead time
    print("\n  [payload_factor별 평균 lead time]")
    print(lead_df.groupby("payload_factor")["lead_time"].mean().to_string())

    return lead_df


def plot_results(results: dict, output_dir: Path):
    """PR 곡선 + Feature Importance 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PR 곡선
    ax = axes[0]
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax.plot(rec, prec, label=f"{name} (PR-AUC={res['pr_auc']:.3f})", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (주 평가지표)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 기준선: 랜덤 분류기 = 양성 비율
    if results:
        baseline = results[list(results.keys())[0]]["pos_ratio"]
        ax.axhline(baseline, ls="--", color="gray", alpha=0.5,
                   label=f"기준선(양성비율={baseline:.3f})")

    # Feature Importance (RandomForest만)
    ax = axes[1]
    if "RandomForest" in results and "feature_importances" in results["RandomForest"]:
        imp = results["RandomForest"]["feature_importances"]
        feat_names = [c.replace("_val", "").replace("ang_rate_rms", "R3")
                        .replace("alt_rmse", "R1").replace("tilt_max", "R2")
                        .replace("vib_ratio", "R4").replace("crash", "R5")
                        .replace("conv_fail", "R6")
                      for c in FEATURE_COLS]
        idx = np.argsort(imp)[::-1]
        ax.barh([feat_names[i] for i in idx], [imp[i] for i in idx], color="steelblue")
        ax.set_xlabel("Feature Importance (Gini)")
        ax.set_title("조기경보 예측에 기여한 신호 (R1~R6)")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out = output_dir / "model_eval.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {out}")


def main():
    parser = argparse.ArgumentParser(description="조기경보 분류기 학습")
    parser.add_argument("--labels",  default="data/processed/labels.parquet")
    parser.add_argument("--out-dir", default="results/models")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print(f"데이터 로드: {labels_path}")
    df = pd.read_parquet(labels_path)
    df = df[df["label"] >= 0].copy()   # 제외 행 다시 한번 필터

    print(f"총 윈도우: {len(df):,} / runs: {df['run_id'].nunique()}")
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    pos_ratio = pos / len(df)
    print(f"양성: {pos:,} / 음성: {neg:,} / 비율={pos_ratio:.3f} (음성:양성={neg/pos:.1f}:1)")

    # ── train/test 분할 (run 단위) ────────────────────────────────────────────
    train_df, test_df = split_by_run(df)

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test  = test_df[FEATURE_COLS].to_numpy()
    y_test  = test_df[TARGET_COL].to_numpy()

    # feature 정규화 (LR용 — RF는 불필요하나 같이 적용해도 무방)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 모델 학습 ────────────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'─'*40}")
        print(f"학습: {name}")
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test, name)
        res["y_true"]    = y_test
        res["pos_ratio"] = pos_ratio
        if hasattr(model, "feature_importances_"):
            res["feature_importances"] = model.feature_importances_
        results[name] = res

    # ── lead time 평가 (조기경보 핵심) ───────────────────────────────────────
    print(f"\n{'─'*40}")
    print("조기경보 lead time 평가 (프로젝트의 핵심 지표)")
    for name, model in models.items():
        scaler_X = scaler.transform(test_df[FEATURE_COLS].to_numpy())
        evaluate_lead_time(model, test_df, scaler_X, name)

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_results(results, out_dir)

    # 모델 저장
    import joblib
    for name, model in models.items():
        model_path = out_dir / f"{name.lower()}.pkl"
        joblib.dump({"model": model, "scaler": scaler,
                     "feature_cols": FEATURE_COLS}, model_path)
        print(f"모델 저장: {model_path}")

    print(f"\n{'='*55}")
    print("학습 완료. 다음 단계: results/models/ 모델 확인 + lead time 해석")
    print("  lead time이 H=2초보다 유의미하게 크면 → 진짜 조기경보 ✅")
    print("  lead time이 ~0이면 → '추락 직전 감지'이므로 threshold 재검토 ⚠️")


if __name__ == "__main__":
    main()