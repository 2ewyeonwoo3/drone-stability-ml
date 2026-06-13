"""
dashboard.py — Stage 3 실시간 조기경보 대시보드.

실행:
  streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 -- --mode ui

배포 (EC2):
  nohup streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 -- --mode ui &
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "randomforest.pkl"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "main"
SERVING_DIR = PROJECT_ROOT / "data" / "serving" / "output"

FEATURE_COLS = [
    "alt_rmse_val", "tilt_max_val", "ang_rate_rms_val",
    "vib_ratio_val", "crash_val", "conv_fail_val",
]
WINDOW_SEC   = 2.0
SLIDE_SEC    = 0.5
ALERT_THRESH = 0.6
H            = 3.5
REFRESH_SEC  = 3   # Kafka 라이브 자동 새로고침 간격


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
    files = glob.glob(str(RAW_DATA_DIR / "payload_f2.5_s6.0_r7.0_S1_seed42.csv"))
    if not files:
        files = glob.glob(str(RAW_DATA_DIR / "*.csv"))[:1]
    if not files:
        print(f"{RAW_DATA_DIR}/*.csv 없음.")
        return
    model, scaler, feat_cols = load_model()
    df  = pd.read_csv(files[0])
    win = int(WINDOW_SEC * 240)
    sld = int(SLIDE_SEC  * 240)
    print(f"재생: {files[0]}")
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
        print(f"{t_end:6.2f}s | R1={feats['alt_rmse_val']:.3f} | prob={prob:.3f} {flag}")
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

    # session_state 초기화
    for k, v in {
        "playing": False, "frame": 0, "history": [],
        "first_alert_t": None, "t_crash": None,
        "run_file": None, "df": None,
        "model": None, "scaler": None, "feat_cols": None,
    }.items():
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
            csvs = sorted(glob.glob(str(RAW_DATA_DIR / "*.csv")))
            if not csvs:
                st.warning("data/raw/main/ 에 CSV 파일이 없습니다.")
            else:
                # run이 여러 개면 selectbox, 하나면 그냥 표시
                stems = [Path(c).stem for c in csvs]
                default_run = "payload_f2.5_s6.0_r7.0_S1_seed42"
                default_idx = next(
                    (i for i, s in enumerate(stems) if s == default_run), 0)

                if len(stems) == 1:
                    selected = stems[0]
                    st.markdown(f"**선택된 run:**\n`{selected}`")
                else:
                    selected = st.selectbox("재생할 run", stems, index=default_idx)

                # run 변경 시 초기화
                if selected != st.session_state.run_file:
                    st.session_state.run_file      = selected
                    st.session_state.playing       = False
                    st.session_state.frame         = 0
                    st.session_state.history       = []
                    st.session_state.first_alert_t = None
                    st.session_state.t_crash       = None
                    st.session_state.df            = None

                col1, col2 = st.columns(2)
                with col1:
                    play_btn = st.button(
                        "▶ 재생",
                        use_container_width=True,
                        disabled=st.session_state.playing,
                    )
                with col2:
                    stop_btn = st.button(
                        "⏹ 정지",
                        use_container_width=True,
                        disabled=not st.session_state.playing,
                    )

                if play_btn:
                    if st.session_state.model is None:
                        m, s, f = load_model()
                        st.session_state.model     = m
                        st.session_state.scaler    = s
                        st.session_state.feat_cols = f
                    df = pd.read_csv(RAW_DATA_DIR / f"{selected}.csv")
                    st.session_state.df      = df
                    st.session_state.frame   = 0
                    st.session_state.history = []
                    st.session_state.first_alert_t = None
                    if "crashed" in df.columns and df["crashed"].any():
                        st.session_state.t_crash = float(
                            df[df["crashed"] == 1]["t"].min())
                    st.session_state.playing = True
                    st.rerun()

                if stop_btn:
                    st.session_state.playing = False
                    st.rerun()

        st.divider()
        st.markdown("### 모델 정보")
        st.markdown(f"- Horizon H = **{H}s**")
        st.markdown(f"- RF PR-AUC = **0.997**")
        st.markdown(f"- Lead time = **2.93s** (평균)")
        st.markdown("- R1(고도 오차) 지배적")
        st.markdown("- 자세 신호(R2~R4)는 보조")

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    st.title("🚁 드론 불안정 조기경보 대시보드")
    st.caption("고도 이상 기반 단기 조기경보 | H=3.5s | RF PR-AUC=0.997")

    # ── Kafka 라이브 모드 ─────────────────────────────────────────────────────
    if data_source == "Kafka 실시간 (live)":
        _render_kafka(st, alert_threshold, make_subplots, go)
        return

    # ── CSV 재생 모드 ─────────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.info("👈 사이드바에서 **▶ 재생** 버튼을 누르세요.")
        return

    _render_csv_frame(st, alert_threshold, make_subplots, go)

    if st.session_state.playing:
        time.sleep(0.05)
        st.rerun()


# ── CSV 재생 프레임 렌더링 ────────────────────────────────────────────────────
def _render_csv_frame(st, alert_threshold, make_subplots, go):
    df        = st.session_state.df
    model     = st.session_state.model
    scaler    = st.session_state.scaler
    feat_cols = st.session_state.feat_cols
    win       = int(WINDOW_SEC * 240)
    sld       = int(SLIDE_SEC  * 240)

    end_idx = win + st.session_state.frame * sld
    if end_idx <= len(df):
        df_win   = df.iloc[end_idx - win:end_idx]
        t_end    = float(df_win["t"].iloc[-1])
        feats    = compute_features(df_win)
        X        = scaler.transform([[feats[c] for c in feat_cols]])
        prob     = float(model.predict_proba(X)[0][1])
        is_alert = prob >= alert_threshold

        st.session_state.history.append({
            "t": t_end, "z": float(df_win["z"].iloc[-1]),
            "prob": prob, "alert": is_alert, **feats,
        })
        if is_alert and st.session_state.first_alert_t is None:
            st.session_state.first_alert_t = t_end

        st.session_state.frame += 1
        if end_idx + sld > len(df):
            st.session_state.playing = False

    history = st.session_state.history
    if not history:
        return

    hist_df    = pd.DataFrame(history)
    last       = history[-1]
    last_alert = last["alert"]
    last_prob  = last["prob"]
    t_crash    = st.session_state.t_crash
    first_t    = st.session_state.first_alert_t

    # 상태 카드
    c1, c2, c3 = st.columns(3)
    if last_alert:
        c1.error("## 🚨 위험 임박")
    else:
        c1.success("## ✅ 정상")
    c2.metric("위험 확률", f"{last_prob:.3f}")
    if first_t and t_crash:
        c3.metric("Lead Time", f"{t_crash - first_t:.1f}s",
                  delta=f"{first_t:.1f}s → {t_crash:.1f}s")
    elif first_t:
        c3.metric("첫 경보", f"{first_t:.2f}s")
    else:
        c3.metric("Lead Time", "—")

    # 진행 바
    total_frames = (len(hist_df["t"]))
    max_t = float(df["t"].iloc[-1])
    cur_t = last["t"]
    st.progress(min(1.0, cur_t / max_t),
                text=f"재생 중: {cur_t:.1f}s / {max_t:.1f}s")

    # 메인 차트
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("고도 z [m]", "위험 확률"),
        row_heights=[0.6, 0.4], vertical_spacing=0.08,
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
    fig.add_hline(y=alert_threshold, line_dash="dot", line_color="orange",
                  annotation_text=f"임계값 {alert_threshold:.2f}",
                  row=2, col=1)
    fig.update_layout(
        height=480, showlegend=False, margin=dict(t=40, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="시각 [s]", row=2, col=1)
    fig.update_yaxes(title_text="z [m]", row=1, col=1)
    fig.update_yaxes(title_text="확률", range=[0, 1.05], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # feature 테이블
    feat_labels = {
        "alt_rmse_val":     "R1 고도오차",
        "tilt_max_val":     "R2 최대자세각",
        "ang_rate_rms_val": "R3 각속도RMS",
        "vib_ratio_val":    "R4 진동비율",
        "crash_val":        "R5 추락지시",
        "conv_fail_val":    "R6 수렴실패",
    }
    with st.expander("📊 현재 Feature 값", expanded=False):
        st.dataframe(pd.DataFrame([
            {"신호": lbl, "값": f"{last[k]:.4f}",
             "상태": "🔴" if last[k] > 0.1 else "🟢"}
            for k, lbl in feat_labels.items()
        ]), use_container_width=True, hide_index=True)

    # 경보 로그
    alerts = [h for h in history if h["alert"]]
    if alerts:
        with st.expander(f"⚠️ 경보 기록 ({len(alerts)}건)", expanded=True):
            for h in alerts[-5:]:
                st.markdown(f"- `t={h['t']:.2f}s` — 위험확률 **{h['prob']:.3f}**")

    # 재생 완료
    if not st.session_state.playing:
        if first_t and t_crash:
            st.success(
                f"✅ 재생 완료 | 첫 경보 **{first_t:.2f}s** → "
                f"추락 **{t_crash:.2f}s** | Lead time = **{t_crash - first_t:.2f}초**")
        elif not first_t:
            st.info("✅ 재생 완료 — 경보 없음 (안정 run)")


# ── Kafka 라이브 모드 ─────────────────────────────────────────────────────────
def _render_kafka(st, alert_threshold, make_subplots, go):
    # foreachBatch가 생성한 batch_id 하위 Parquet 파일을 수정 시간순으로 찾는다.
    try:
        parquets = sorted(
            (p for p in SERVING_DIR.rglob("*.parquet") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError as e:
        st.error(f"서빙 디렉터리 조회 오류: {e}")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    # serving 데이터 없을 때만 안내 표시
    if not parquets:
        st.warning(
            "⏳ 서빙 데이터가 없습니다. Spark 서빙을 시작하고 데이터를 전송하세요.\n\n"
            "```bash\n"
            "spark-submit \\\n"
            "  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\\n"
            "  src/spark_consumer_serving.py\n"
            "```"
        )
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    # 데이터 로드
    try:
        frames = []
        for parquet_path in parquets[-20:]:
            try:
                frames.append(pd.read_parquet(parquet_path))
            except Exception:
                # Spark가 파일을 쓰는 짧은 순간에는 해당 파일만 건너뛴다.
                continue

        if not frames:
            raise RuntimeError("읽을 수 있는 Parquet 파일이 아직 없습니다.")

        df_live = pd.concat(frames, ignore_index=True)
        df_live = (
            df_live
            .sort_values("window_end")
            .drop_duplicates(
                subset=["window_end", "drone_id"],
                keep="last",
            )
            .tail(60)
        )
    except Exception as e:
        st.error(f"데이터 읽기 오류: {e}")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    required_cols = {"window_end", "drone_id", "risk_prob", "is_alert"}
    missing_cols = required_cols - set(df_live.columns)
    if missing_cols:
        st.error(
            "Parquet 스키마에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_cols))
        )
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    df_live["window_end"] = pd.to_datetime(
        df_live["window_end"],
        format="mixed",
        errors="coerce",
    )
    df_live = df_live.dropna(subset=["window_end"])
    df_live = df_live.sort_values("window_end")
    df_live = df_live.drop_duplicates(
        subset=["window_end", "drone_id"],
        keep="last",
    )
    df_recent = df_live.tail(30)

    if df_recent.empty:
        st.info("⏳ 처리된 윈도우가 아직 없습니다.")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    # 대시보드 슬라이더 기준으로 경보 여부를 재계산한다.
    # (Spark가 저장한 is_alert는 ALERT_THRESH=0.6 고정값 기준이므로
    #  사이드바 슬라이더와 별도 기준이 되어 화면이 일치하지 않는 문제를 방지)
    df_recent = df_recent.copy()
    df_recent["display_alert"] = (
        df_recent["risk_prob"] >= alert_threshold
    ).astype(int)

    # 상태 카드
    n_alert = int(df_recent["display_alert"].sum())
    n_total = len(df_recent)
    last_prob = float(df_recent["risk_prob"].iloc[-1])
    last_alert = bool(df_recent["display_alert"].iloc[-1])

    c1, c2, c3 = st.columns(3)
    if last_alert:
        c1.error("## 🚨 위험 임박")
    else:
        c1.success("## ✅ 정상")
    c2.metric("현재 위험 확률", f"{last_prob:.3f}")
    c3.metric("경보 비율", f"{n_alert}/{n_total}",
              delta=f"최근 {n_total}개 윈도우")

    # 차트
    bar_colors = ["crimson" if a else "#4A90D9"
                  for a in df_recent["display_alert"]]
    fig = go.Figure()
    fig.add_bar(
        x=df_recent["window_end"],
        y=df_recent["risk_prob"],
        marker_color=bar_colors,
        opacity=0.85,
    )
    fig.add_hline(y=alert_threshold, line_dash="dot",
                  line_color="orange",
                  annotation_text=f"임계값 {alert_threshold:.2f}")
    fig.update_layout(
        title="실시간 위험 확률 (Kafka 스트림)",
        xaxis_title="시각", yaxis_title="위험 확률",
        yaxis_range=[0, 1.05], height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 최근 경보 로그
    alert_rows = df_recent[df_recent["display_alert"] == 1]
    if len(alert_rows) > 0:
        with st.expander(f"⚠️ 경보 기록 ({len(alert_rows)}건)", expanded=True):
            for _, row in alert_rows.tail(5).iterrows():
                st.markdown(
                    f"- `{row['window_end']}` — "
                    f"위험확률 **{row['risk_prob']:.3f}** | "
                    f"R1={row.get('alt_rmse_val', 0):.3f}")

    # 상세 테이블
    with st.expander("📋 최근 윈도우 상세", expanded=False):
        show = [c for c in ["window_end", "alt_rmse_val",
                             "ang_rate_rms_val", "risk_prob", "is_alert"]
                if c in df_recent.columns]
        st.dataframe(df_recent[show].tail(10).reset_index(drop=True),
                     use_container_width=True)

    # 자동 새로고침 (버튼 없이)
    st.caption(f"🔄 {REFRESH_SEC}초마다 자동 업데이트 | "
               f"마지막: {df_recent['window_end'].iloc[-1]}")
    time.sleep(REFRESH_SEC)
    st.rerun()


# ── Spark 스텁 ────────────────────────────────────────────────────────────────
def run_spark():
    print("spark_consumer_serving.py를 직접 실행하세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "ui", "spark"],
                        default="ui")
    args = parser.parse_args()
    {"demo": run_demo, "ui": run_ui, "spark": run_spark}[args.mode]()


if __name__ == "__main__":
    main()