"""
dashboard.py — Stage 3 실시간 조기경보 대시보드.

실행:
  streamlit run src/dashboard.py -- --mode ui

배포:
  Streamlit Cloud: GitHub 연결 → Main file: src/dashboard.py
  로컬: streamlit run src/dashboard.py -- --mode ui

모드:
  CSV 재생: data/raw/main/*.csv 중 선택 → 슬라이딩 윈도우 재생
  Kafka 라이브: data/serving/ Parquet polling (spark_consumer_serving.py 필요)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

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
ALERT_THRESH = 0.6
H            = 3.5
SERVING_DIR  = "data/serving"


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["scaler"], artifact["feature_cols"]


def compute_features(df_win: pd.DataFrame, target_z: float = 1.0) -> dict:
    from stability_metrics import (
        altitude_rmse, attitude_max_angle, angular_rate_rms,
        vibration_ratio, crash_indicator, convergence_failure,
        StabilityThresholds,
    )
    thr     = StabilityThresholds()
    z       = df_win["z"].to_numpy()
    roll    = df_win["roll"].to_numpy()
    pitch   = df_win["pitch"].to_numpy()
    wx      = df_win["wx"].to_numpy()
    wy      = df_win["wy"].to_numpy()
    wz      = df_win["wz"].to_numpy()
    t       = df_win["t"].to_numpy()
    fs      = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 240.0
    contact = df_win["contact"].to_numpy() if "contact" in df_win.columns else None
    return {
        "alt_rmse_val":     altitude_rmse(z, target_z),
        "tilt_max_val":     attitude_max_angle(roll, pitch),
        "ang_rate_rms_val": angular_rate_rms(wx, wy, wz),
        "vib_ratio_val":    vibration_ratio(roll, fs),
        "crash_val":        float(crash_indicator(z, contact, thr.crash_floor)),
        "conv_fail_val":    float(convergence_failure(z, target_z, thr.conv_tol)),
    }


# ── demo 모드 (터미널) ────────────────────────────────────────────────────────
def run_demo():
    files = glob.glob("data/raw/main/payload_f2.5_s6.0_r7.0_S1_seed42.csv")
    if not files:
        files = glob.glob("data/raw/main/*.csv")[:1]
    if not files:
        print("data/raw/main/*.csv 없음.")
        return

    model, scaler, feat_cols = load_model()
    df  = pd.read_csv(files[0])
    win = int(WINDOW_SEC * 240)
    sld = int(SLIDE_SEC  * 240)

    print(f"재생: {files[0]}")
    print(f"{'t':>6} | {'R1':>6} | {'R3':>6} | {'prob':>6} | 경보")
    print("-" * 40)

    alerts = []
    for end in range(win, len(df) + 1, sld):
        df_win = df.iloc[end - win:end]
        t_end  = float(df_win["t"].iloc[-1])
        feats  = compute_features(df_win)
        X      = scaler.transform([[feats[c] for c in feat_cols]])
        prob   = float(model.predict_proba(X)[0][1])
        flag   = "🚨" if prob >= ALERT_THRESH else ""
        if prob >= ALERT_THRESH:
            alerts.append(t_end)
        print(f"{t_end:6.2f}s | {feats['alt_rmse_val']:6.3f} | "
              f"{feats['ang_rate_rms_val']:6.3f} | {prob:6.3f} | {flag}")

    if alerts and "crashed" in df.columns and df["crashed"].any():
        t_crash = float(df[df["crashed"] == 1]["t"].min())
        print(f"\n첫 경보: {min(alerts):.2f}s  추락: {t_crash:.2f}s  "
              f"Lead time: {t_crash - min(alerts):.2f}s")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def run_ui():
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.set_page_config(
        page_title="드론 불안정 조기경보",
        page_icon="🚁",
        layout="wide",
    )

    # ── session_state 초기화 ──────────────────────────────────────────────────
    defaults = {
        "playing":       False,   # CSV 재생 중 여부
        "frame":         0,       # 현재 윈도우 인덱스
        "history":       [],      # {"t","z","prob","alert"} 리스트
        "first_alert_t": None,
        "t_crash":       None,
        "run_file":      None,
        "df":            None,
        "model":         None,
        "scaler":        None,
        "feat_cols":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")
        data_source = st.selectbox(
            "데이터 소스",
            ["CSV 재생 (demo)", "Kafka 실시간 (live)"],
        )
        alert_threshold = st.slider(
            "경보 임계값",
            min_value=0.3, max_value=0.9,
            value=ALERT_THRESH, step=0.05,
        )

        if data_source == "CSV 재생 (demo)":
            csvs = sorted(glob.glob("data/raw/main/*.csv"))
            if csvs:
                default_run = "payload_f2.5_s6.0_r7.0_S1_seed42"
                default_idx = next(
                    (i for i, c in enumerate(csvs)
                     if Path(c).stem == default_run), 0)
                selected = st.selectbox(
                    "재생할 run",
                    [Path(c).stem for c in csvs],
                    index=default_idx,
                )

                # run이 바뀌면 상태 초기화
                if selected != st.session_state.run_file:
                    st.session_state.run_file      = selected
                    st.session_state.playing       = False
                    st.session_state.frame         = 0
                    st.session_state.history       = []
                    st.session_state.first_alert_t = None
                    st.session_state.t_crash       = None
                    st.session_state.df            = None

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("▶ 재생", use_container_width=True,
                                 disabled=st.session_state.playing):
                        # 모델 로드 (처음 한 번)
                        if st.session_state.model is None:
                            m, s, f = load_model()
                            st.session_state.model     = m
                            st.session_state.scaler    = s
                            st.session_state.feat_cols = f
                        # 데이터 로드
                        df = pd.read_csv(f"data/raw/main/{selected}.csv")
                        st.session_state.df      = df
                        st.session_state.frame   = 0
                        st.session_state.history = []
                        st.session_state.first_alert_t = None
                        if "crashed" in df.columns and df["crashed"].any():
                            st.session_state.t_crash = float(
                                df[df["crashed"] == 1]["t"].min())
                        st.session_state.playing = True
                        st.rerun()

                with col_btn2:
                    if st.button("⏹ 정지", use_container_width=True,
                                 disabled=not st.session_state.playing):
                        st.session_state.playing = False
                        st.rerun()

        st.divider()
        st.markdown("### 모델 정보")
        st.markdown(f"- Horizon H = **{H}s**")
        st.markdown(f"- PR-AUC = **0.997**")
        st.markdown(f"- Lead time = **2.93s** (평균)")
        st.markdown("- R1(고도 오차) 지배적")
        st.markdown("- 자세 신호(R2~R4)는 보조")

    # ── 메인 헤더 ─────────────────────────────────────────────────────────────
    st.title("🚁 드론 불안정 조기경보 대시보드")
    st.caption("고도 이상 기반 단기 조기경보 | H=3.5s | RF PR-AUC=0.997")

    # ── Kafka 라이브 모드 ─────────────────────────────────────────────────────
    if data_source == "Kafka 실시간 (live)":
        _render_kafka_live(st, alert_threshold)
        return

    # ── CSV 재생 모드 ─────────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.info("사이드바에서 run을 선택하고 ▶ 재생을 누르세요.")
        return

    _render_csv_frame(st, alert_threshold, make_subplots, go)

    # 재생 중이면 다음 프레임으로
    if st.session_state.playing:
        time.sleep(0.05)
        st.rerun()


def _render_csv_frame(st, alert_threshold, make_subplots, go):
    """현재 session_state.frame 기준으로 한 프레임 처리 후 화면 업데이트."""
    import plotly.graph_objects as go2
    from plotly.subplots import make_subplots2  # noqa — 아래에서 직접 import

    df        = st.session_state.df
    model     = st.session_state.model
    scaler    = st.session_state.scaler
    feat_cols = st.session_state.feat_cols
    win       = int(WINDOW_SEC * 240)
    sld       = int(SLIDE_SEC  * 240)

    # 아직 처리 안 한 프레임 계산
    end_idx = win + st.session_state.frame * sld
    if end_idx <= len(df):
        df_win = df.iloc[end_idx - win:end_idx]
        t_end  = float(df_win["t"].iloc[-1])
        feats  = compute_features(df_win)
        X      = scaler.transform([[feats[c] for c in feat_cols]])
        prob   = float(model.predict_proba(X)[0][1])
        is_alert = prob >= alert_threshold

        st.session_state.history.append({
            "t": t_end, "z": float(df_win["z"].iloc[-1]),
            "prob": prob, "alert": is_alert, **feats,
        })

        if is_alert and st.session_state.first_alert_t is None:
            st.session_state.first_alert_t = t_end

        st.session_state.frame += 1

        # 마지막 프레임이면 정지
        if end_idx + sld > len(df):
            st.session_state.playing = False

    history = st.session_state.history
    if not history:
        st.info("재생 중...")
        return

    hist_df = pd.DataFrame(history)

    # ── 상태 카드 3개 ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    last_alert = history[-1]["alert"]
    last_prob  = history[-1]["prob"]

    if last_alert:
        col1.error("## 🚨 위험 임박")
    else:
        col1.success("## ✅ 정상")

    col2.metric("위험 확률", f"{last_prob:.3f}")

    t_crash = st.session_state.t_crash
    first_t = st.session_state.first_alert_t
    if first_t and t_crash:
        col3.metric("Lead Time", f"{t_crash - first_t:.1f}s",
                    delta=f"경보 {first_t:.1f}s → 추락 {t_crash:.1f}s")
    elif first_t:
        col3.metric("첫 경보", f"{first_t:.2f}s")
    else:
        col3.metric("Lead Time", "—")

    # ── 메인 차트 ─────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("고도 z [m]", "위험 확률"),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=hist_df["t"], y=hist_df["z"],
        mode="lines", name="고도",
        line=dict(color="steelblue", width=2),
    ), row=1, col=1)

    if t_crash:
        fig.add_vline(x=t_crash, line_dash="dash", line_color="red",
                      annotation_text="추락", row=1, col=1)

    bar_colors = ["crimson" if a else "#4A90D9" for a in hist_df["alert"]]
    fig.add_trace(go.Bar(
        x=hist_df["t"], y=hist_df["prob"],
        marker_color=bar_colors, opacity=0.8, name="위험확률",
    ), row=2, col=1)

    fig.add_hline(
        y=alert_threshold, line_dash="dot", line_color="orange",
        annotation_text=f"임계값 {alert_threshold:.2f}",
        row=2, col=1,
    )

    fig.update_layout(
        height=480,
        showlegend=False,
        margin=dict(t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="시각 [s]", row=2, col=1)
    fig.update_yaxes(title_text="z [m]", row=1, col=1)
    fig.update_yaxes(title_text="확률", range=[0, 1.05], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── feature 테이블 ────────────────────────────────────────────────────────
    feat_labels = {
        "alt_rmse_val":     "R1 고도오차 RMSE",
        "tilt_max_val":     "R2 최대자세각",
        "ang_rate_rms_val": "R3 각속도 RMS",
        "vib_ratio_val":    "R4 진동비율",
        "crash_val":        "R5 추락지시",
        "conv_fail_val":    "R6 수렴실패",
    }
    last = history[-1]
    feat_df = pd.DataFrame([
        {"신호": label, "현재값": f"{last[k]:.4f}",
         "상태": "🔴" if last[k] > 0.1 else "🟢"}
        for k, label in feat_labels.items()
    ])
    with st.expander("📊 현재 윈도우 Feature 값", expanded=False):
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── 경보 로그 ─────────────────────────────────────────────────────────────
    alert_rows = [h for h in history if h["alert"]]
    if alert_rows:
        with st.expander(f"⚠️ 경보 기록 ({len(alert_rows)}건)", expanded=True):
            for h in alert_rows[-5:]:
                st.markdown(
                    f"- `t={h['t']:.2f}s` — 위험확률 **{h['prob']:.3f}**")

    # ── 재생 완료 배너 ────────────────────────────────────────────────────────
    if not st.session_state.playing and len(history) > 0:
        if first_t and t_crash:
            st.success(
                f"✅ 재생 완료 | 첫 경보 {first_t:.2f}s → "
                f"추락 {t_crash:.2f}s | **Lead time = {t_crash - first_t:.2f}초**")
        elif not first_t:
            st.info("✅ 재생 완료 — 경보 없음 (안정 run)")


def _render_kafka_live(st, alert_threshold):
    """Kafka 라이브 모드: data/serving/ Parquet polling."""
    import plotly.graph_objects as go

    st.info(
        "**Spark 서빙이 실행 중이어야 합니다.**\n\n"
        "```bash\nspark-submit \\\n"
        "  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\\n"
        "  src/spark_consumer_serving.py\n```"
    )

    parquets = sorted(
        glob.glob(f"{SERVING_DIR}/**/*.parquet", recursive=True))

    if not parquets:
        st.warning("⏳ 아직 서빙 데이터 없음. Spark 서빙 시작 후 새로고침하세요.")
        if st.button("🔄 새로고침"):
            st.rerun()
        return

    try:
        df_live = pd.concat([pd.read_parquet(p) for p in parquets[-20:]])
        df_live = df_live.sort_values("window_end").drop_duplicates(
            "window_end").tail(60)
    except Exception as e:
        st.error(f"데이터 읽기 오류: {e}")
        return

    # 상태 카드
    n_alert = int((df_live["is_alert"] == 1).sum())
    col1, col2, col3 = st.columns(3)
    if n_alert > 0:
        col1.error("## 🚨 위험 임박")
        latest = df_live[df_live["is_alert"] == 1]["window_end"].iloc[-1]
        col3.metric("최근 경보", str(latest)[-12:])
    else:
        col1.success("## ✅ 정상")
        col3.metric("경보", "없음")
    col2.metric("총 경보 건수", f"{n_alert}건")

    # 차트
    bar_colors = ["crimson" if a else "#4A90D9"
                  for a in df_live["is_alert"]]
    fig = go.Figure()
    fig.add_bar(
        x=df_live["window_end"],
        y=df_live["risk_prob"],
        marker_color=bar_colors,
        opacity=0.85,
        name="위험확률",
    )
    fig.add_hline(
        y=alert_threshold, line_dash="dot", line_color="orange",
        annotation_text=f"임계값 {alert_threshold:.2f}",
    )
    fig.update_layout(
        title="실시간 위험 확률 (Kafka 스트림)",
        xaxis_title="시각",
        yaxis_title="위험 확률",
        yaxis_range=[0, 1.05],
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 최근 윈도우 테이블
    with st.expander("📋 최근 윈도우 상세", expanded=False):
        show_cols = ["window_end", "alt_rmse_val", "ang_rate_rms_val",
                     "risk_prob", "is_alert"]
        show_cols = [c for c in show_cols if c in df_live.columns]
        st.dataframe(
            df_live[show_cols].tail(10).reset_index(drop=True),
            use_container_width=True)

    # 자동 새로고침
    col_r1, col_r2 = st.columns([3, 1])
    col_r1.caption(f"마지막 업데이트: {df_live['window_end'].iloc[-1]}")
    if col_r2.button("🔄 새로고침"):
        st.rerun()

    # 5초마다 자동 rerun
    time.sleep(3)
    st.rerun()


# ── Spark 스텁 ────────────────────────────────────────────────────────────────
def run_spark():
    print("spark_consumer_serving.py를 직접 실행하세요:")
    print("  spark-submit \\")
    print("    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\")
    print("    src/spark_consumer_serving.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "ui", "spark"],
                        default="ui")
    args = parser.parse_args()
    if args.mode == "demo":
        run_demo()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "spark":
        run_spark()


if __name__ == "__main__":
    main()