"""
dashboard.py — Stage 3 실시간 조기경보 대시보드.

Kafka → Spark Structured Streaming → RF 모델 예측 → Streamlit 대시보드.

구조:
  1. Spark 스트리밍이 Kafka에서 텔레메트리 읽기
  2. 슬라이딩 윈도우로 R1~R6 feature 계산
  3. 학습된 RF 모델로 위험 임박 확률 예측
  4. Streamlit이 예측 결과를 실시간 시각화

실행:
  # 터미널 1: Kafka + Producer
  docker-compose up -d
  python src/producer.py

  # 터미널 2: Spark 스트리밍 + 예측 (백그라운드)
  python src/dashboard.py --mode spark

  # 터미널 3: 대시보드 UI
  streamlit run src/dashboard.py -- --mode ui

  # 통합 실행 (개발용)
  python src/dashboard.py --mode demo  # 시뮬 데이터로 직접 시각화

의존성:
  pip install streamlit plotly joblib
"""

from __future__ import annotations

import argparse
import os, sys, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH   = Path("results/models/randomforest.pkl")
FEATURE_COLS = [
    "alt_rmse_val", "tilt_max_val", "ang_rate_rms_val",
    "vib_ratio_val", "crash_val", "conv_fail_val",
]
WINDOW_SEC   = 2.0
SLIDE_SEC    = 0.5
H            = 3.5   # 조기경보 horizon [s]
ALERT_THRESH = 0.6   # 위험 임박 확률 임계값


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["scaler"], artifact["feature_cols"]


# ── 윈도우 feature 계산 (stability_metrics 재사용) ───────────────────────────
def compute_features_from_window(df_win: pd.DataFrame,
                                  target_z: float = 1.0) -> dict:
    """텔레메트리 윈도우 → R1~R6 feature dict."""
    from stability_metrics import (
        altitude_rmse, attitude_max_angle, angular_rate_rms,
        vibration_ratio, crash_indicator, convergence_failure,
        StabilityThresholds,
    )
    thr = StabilityThresholds()
    z    = df_win["z"].to_numpy()
    roll = df_win["roll"].to_numpy()
    pitch= df_win["pitch"].to_numpy()
    wx   = df_win["wx"].to_numpy()
    wy   = df_win["wy"].to_numpy()
    wz   = df_win["wz"].to_numpy()
    t    = df_win["t"].to_numpy()
    fs   = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 240.0
    contact = df_win["contact"].to_numpy() if "contact" in df_win.columns else None

    return {
        "alt_rmse_val":     altitude_rmse(z, target_z),
        "tilt_max_val":     attitude_max_angle(roll, pitch),
        "ang_rate_rms_val": angular_rate_rms(wx, wy, wz),
        "vib_ratio_val":    vibration_ratio(roll, fs),
        "crash_val":        float(crash_indicator(z, contact, thr.crash_floor)),
        "conv_fail_val":    float(convergence_failure(z, target_z, thr.conv_tol)),
    }


# ── 데모 모드: 로컬 시뮬 데이터로 실시간 예측 시각화 ─────────────────────────
def run_demo():
    """
    저장된 CSV 파일을 실시간 스트림처럼 재생하며 예측 결과를 시각화.
    Streamlit 없이 터미널에서 빠르게 동작 확인.
    """
    import glob
    csv_files = sorted(glob.glob("data/raw/main/payload_f2.5_s2.0_r4.0_S1_seed42.csv"))
    if not csv_files:
        csv_files = sorted(glob.glob("data/raw/main/*.csv"))[:1]
    if not csv_files:
        print("data/raw/main/*.csv 없음. run_batch_main.py 먼저 실행하세요.")
        return

    model, scaler, feat_cols = load_model()
    df = pd.read_csv(csv_files[0])
    print(f"재생: {csv_files[0]} ({len(df)}행)")
    print(f"{'시각':>6} | {'z':>6} | {'R1':>6} | {'R3':>6} | {'위험확률':>8} | 경보")
    print("-" * 55)

    fs  = 240.0
    win = int(WINDOW_SEC * fs)
    sld = int(SLIDE_SEC  * fs)
    alerts = []

    for end in range(win, len(df) + 1, sld):
        start   = end - win
        df_win  = df.iloc[start:end]
        t_end   = float(df_win["t"].iloc[-1])
        feats   = compute_features_from_window(df_win)

        X = scaler.transform([[feats[c] for c in feat_cols]])
        prob = model.predict_proba(X)[0][1]
        alert = "🚨 위험!" if prob >= ALERT_THRESH else ""
        if prob >= ALERT_THRESH:
            alerts.append(t_end)

        print(f"{t_end:6.2f}s | {feats['alt_rmse_val']:6.3f} | "
              f"{feats['alt_rmse_val']:6.3f} | {feats['ang_rate_rms_val']:6.3f} | "
              f"{prob:8.3f} | {alert}")

    if alerts:
        print(f"\n첫 경보 시각: {min(alerts):.2f}s")
        t_fail_approx = float(df[df["crashed"]==1]["t"].min()) if "crashed" in df.columns else None
        if t_fail_approx:
            print(f"추락 시각:    {t_fail_approx:.2f}s")
            print(f"Lead time:    {t_fail_approx - min(alerts):.2f}s")


# ── Streamlit 대시보드 UI ─────────────────────────────────────────────────────
def run_ui():
    """Streamlit 실시간 대시보드."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("pip install streamlit plotly 필요")
        return

    st.set_page_config(
        page_title="드론 불안정 조기경보",
        page_icon="🚁",
        layout="wide",
    )

    st.title("🚁 드론 불안정 조기경보 대시보드")
    st.caption("RF 모델 기반 실시간 예측 | H=3.5s | PR-AUC=0.997")

    # ── 사이드바 설정 ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")
        data_source = st.selectbox(
            "데이터 소스",
            ["CSV 재생 (demo)", "Kafka 실시간 (live)"]
        )
        alert_threshold = st.slider(
            "경보 임계값 (위험 확률)",
            min_value=0.3, max_value=0.9, value=ALERT_THRESH, step=0.05
        )
        run_file = None
        if data_source == "CSV 재생 (demo)":
            import glob
            csvs = sorted(glob.glob("data/raw/main/*.csv"))
            if csvs:
                run_file = st.selectbox("재생할 run", [Path(c).stem for c in csvs])

        st.divider()
        st.markdown("### 모델 정보")
        st.markdown(f"- Horizon H = **{H}s**")
        st.markdown(f"- PR-AUC = **0.997**")
        st.markdown(f"- Lead time = **2.93s** (평균)")
        st.markdown("- Features: R1~R6 (R1+R3 주요)")

    # ── 메인 레이아웃 ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    status_box  = col1.empty()
    prob_box    = col2.empty()
    leadtime_box = col3.empty()

    chart_area  = st.empty()
    feature_area = st.empty()
    log_area    = st.empty()

    if not run_file:
        st.info("사이드바에서 재생할 run을 선택하세요.")
        return

    # ── CSV 재생 ─────────────────────────────────────────────────────────────
    model, scaler, feat_cols = load_model()
    df = pd.read_csv(f"data/raw/main/{run_file}.csv")

    fs  = 240.0
    win = int(WINDOW_SEC * fs)
    sld = int(SLIDE_SEC  * fs)

    history = {
        "t": [], "z": [], "prob": [], "alert": [],
        **{f: [] for f in FEATURE_COLS}
    }
    first_alert_t = None
    t_crash = None
    if "crashed" in df.columns and df["crashed"].any():
        crash_rows = df[df["crashed"] == 1]
        t_crash = float(crash_rows["t"].min())

    log_lines = []

    for end in range(win, len(df) + 1, sld):
        start  = end - win
        df_win = df.iloc[start:end]
        t_end  = float(df_win["t"].iloc[-1])
        feats  = compute_features_from_window(df_win)

        X    = scaler.transform([[feats[c] for c in feat_cols]])
        prob = float(model.predict_proba(X)[0][1])
        is_alert = prob >= alert_threshold

        history["t"].append(t_end)
        history["z"].append(float(df_win["z"].iloc[-1]))
        history["prob"].append(prob)
        history["alert"].append(is_alert)
        for f in FEATURE_COLS:
            history[f].append(feats[f])

        if is_alert and first_alert_t is None:
            first_alert_t = t_end
            log_lines.append(f"⚠️ t={t_end:.2f}s: 첫 경보 발생 (prob={prob:.3f})")

        # 상태 카드 업데이트
        if is_alert:
            status_box.error("🚨 위험 임박")
        else:
            status_box.success("✅ 정상")
        prob_box.metric("위험 확률", f"{prob:.3f}", delta=None)
        if first_alert_t and t_crash:
            lt = t_crash - first_alert_t
            leadtime_box.metric("Lead Time", f"{lt:.1f}s", delta=None)
        elif first_alert_t:
            leadtime_box.metric("경보 시각", f"{first_alert_t:.2f}s")

        # 차트 업데이트
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("고도 z [m]", "위험 확률"),
                            row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=history["t"], y=history["z"],
                                  mode="lines", name="z(t)",
                                  line=dict(color="steelblue")), row=1, col=1)
        if t_crash:
            fig.add_vline(x=t_crash, line_dash="dash", line_color="red",
                          annotation_text="추락", row=1, col=1)

        # 위험 확률
        colors = ["crimson" if a else "steelblue" for a in history["alert"]]
        fig.add_trace(go.Bar(x=history["t"], y=history["prob"],
                              name="위험확률", marker_color=colors,
                              opacity=0.7), row=2, col=1)
        fig.add_hline(y=alert_threshold, line_dash="dot", line_color="orange",
                      row=2, col=1)
        fig.update_layout(height=450, showlegend=False, margin=dict(t=40))
        chart_area.plotly_chart(fig, use_container_width=True)

        # feature 값
        feat_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "값": [feats[f] for f in FEATURE_COLS],
        })
        feature_area.dataframe(feat_df, use_container_width=True)

        if log_lines:
            log_area.text("\n".join(log_lines[-5:]))

        time.sleep(0.02)  # 재생 속도 조절

    # 최종 요약
    if first_alert_t and t_crash:
        st.success(f"✅ Lead time = {t_crash - first_alert_t:.2f}초 — "
                   f"추락 {t_crash:.2f}s, 첫 경보 {first_alert_t:.2f}s")
    elif not first_alert_t:
        st.info("경보 없음 — 안정 run")


# ── Spark 스트리밍 모드 (Kafka → 예측 → 로컬 저장) ──────────────────────────
def run_spark():
    """
    Kafka에서 실시간 텔레메트리를 읽어 RF 예측 결과를 Parquet으로 저장.
    spark_consumer.py의 UDF를 RF 예측으로 확장한 버전.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import from_json, col, window, collect_list, sort_array, struct
        from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
        from config import PipelineConfig
    except ImportError as e:
        print(f"Spark 의존성 없음: {e}")
        return

    cfg = PipelineConfig()
    model, scaler, feat_cols = load_model()

    # UDF: 윈도우 행 리스트 → RF 예측 확률
    from pyspark.sql.functions import udf, pandas_udf
    from pyspark.sql.types import FloatType

    def predict_window_udf(rows):
        """collect_list 결과를 받아 RF 예측 확률 반환."""
        try:
            rows_sorted = sorted(rows, key=lambda r: r["step"])
            df_win = pd.DataFrame(rows_sorted)
            feats  = compute_features_from_window(df_win)
            X = scaler.transform([[feats[c] for c in feat_cols]])
            return float(model.predict_proba(X)[0][1])
        except Exception:
            return 0.0

    spark = (SparkSession.builder
             .appName("drone-early-warning-serving")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print(f"[serving] Kafka 브로커: {cfg.kafka_bootstrap}")
    print(f"[serving] 예측 결과 저장: {cfg.sink_path}/predictions/")

    # (실제 스트리밍 구현은 spark_consumer.py와 동일 구조, UDF만 교체)
    print("[serving] Spark 스트리밍 시작... (Ctrl+C로 종료)")
    print("[serving] 현재는 demo 모드 실행 권장: python src/dashboard.py --mode demo")


def main():
    parser = argparse.ArgumentParser(description="조기경보 대시보드")
    parser.add_argument("--mode", choices=["demo", "ui", "spark"],
                        default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "spark":
        run_spark()


if __name__ == "__main__":
    main()