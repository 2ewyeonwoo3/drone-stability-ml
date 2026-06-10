"""
spark_consumer_serving.py — Stage 3 실시간 조기경보 서빙.

기존 spark_consumer.py의 윈도우 집계 구조를 그대로 재사용하고,
feature 계산 직후 RF 예측 단계를 추가한 버전.

변경점 (spark_consumer.py 대비):
  - applyInPandas UDF 안에서 feature 계산 + RF 예측을 한 번에 수행
  - 모델·scaler·feature_cols를 모듈 전역 캐시로 worker에서 1회 로드
    (broadcast 대신 — 로컬 모드에서 단순하고 안전)
  - 출력에 risk_prob(위험확률), is_alert(경보 여부) 컬럼 추가
  - 경보 발생 시 콘솔에 🚨 표시

핵심 난관 해결:
  ① worker 모델 전달: _get_model() 함수로 모듈 전역 캐싱
     → applyInPandas 함수가 호출될 때 파일을 로드하고 이후는 캐시 사용
  ② scaler 누락 방지: 모델+scaler+feature_cols를 _MODEL_CACHE에 한 묶음으로 관리
  ③ 기존 UDF 재사용: evaluate_window(stability_metrics)를 그대로 쓰고
     그 결과에 predict_proba만 추가

실행:
  # 터미널 1: Kafka + Zookeeper
  docker-compose up -d

  # 터미널 2: producer (드론 시뮬 → Kafka)
  python src/producer.py

  # 터미널 3: 실시간 서빙 (Kafka → feature → RF 예측 → 콘솔)
  spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:4.1.2 \\
    src/spark_consumer_serving.py

  # S3 싱크도 같이 원하면
  ENV_FILE=.env.s3 spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:4.1.2,\\
               org.apache.hadoop:hadoop-aws:3.3.4 \\
    src/spark_consumer_serving.py

  # 버전: pyspark 3.5.1 + spark-sql-kafka-0-10_2.12:3.5.1
      (Stage 1.5에서 검증된 조합)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pandas as pd

from config import cfg

WINDOW_DURATION = "2 seconds"
SLIDE_DURATION  = "0.5 seconds"
WATERMARK_DELAY = "5 seconds"
ALERT_THRESH    = 0.6
MODEL_PATH      = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "models", "randomforest.pkl"
)

# ── ① worker 모델 캐시 ────────────────────────────────────────────────────────
# applyInPandas는 worker 프로세스에서 실행되므로
# 모듈 전역 dict에 로드한 모델을 캐싱 → 첫 호출 시 1회만 joblib.load
_MODEL_CACHE: dict = {}

def _get_model():
    """모델+scaler+feature_cols를 한 묶음으로 반환 (worker별 1회 로드)."""
    if not _MODEL_CACHE:
        import joblib
        artifact = joblib.load(MODEL_PATH)
        _MODEL_CACHE["model"]    = artifact["model"]
        _MODEL_CACHE["scaler"]   = artifact["scaler"]
        _MODEL_CACHE["features"] = artifact["feature_cols"]
    return (_MODEL_CACHE["model"],
            _MODEL_CACHE["scaler"],
            _MODEL_CACHE["features"])


# ── Kafka 스키마 (spark_consumer.py와 동일) ───────────────────────────────────
TELEMETRY_SCHEMA = T.StructType([
    T.StructField("step",        T.IntegerType(), True),
    T.StructField("t",           T.DoubleType(),  True),
    T.StructField("x",           T.DoubleType(),  True),
    T.StructField("y",           T.DoubleType(),  True),
    T.StructField("z",           T.DoubleType(),  True),
    T.StructField("qx",          T.DoubleType(),  True),
    T.StructField("qy",          T.DoubleType(),  True),
    T.StructField("qz",          T.DoubleType(),  True),
    T.StructField("qw",          T.DoubleType(),  True),
    T.StructField("roll",        T.DoubleType(),  True),
    T.StructField("pitch",       T.DoubleType(),  True),
    T.StructField("yaw",         T.DoubleType(),  True),
    T.StructField("vx",          T.DoubleType(),  True),
    T.StructField("vy",          T.DoubleType(),  True),
    T.StructField("vz",          T.DoubleType(),  True),
    T.StructField("wx",          T.DoubleType(),  True),
    T.StructField("wy",          T.DoubleType(),  True),
    T.StructField("wz",          T.DoubleType(),  True),
    T.StructField("rpm0",        T.DoubleType(),  True),
    T.StructField("rpm1",        T.DoubleType(),  True),
    T.StructField("rpm2",        T.DoubleType(),  True),
    T.StructField("rpm3",        T.DoubleType(),  True),
    T.StructField("target_x",    T.DoubleType(),  True),
    T.StructField("target_y",    T.DoubleType(),  True),
    T.StructField("target_z",    T.DoubleType(),  True),
    T.StructField("mass_factor", T.DoubleType(),  True),
    T.StructField("wind_x",      T.DoubleType(),  True),
    T.StructField("wind_y",      T.DoubleType(),  True),
    T.StructField("wind_z",      T.DoubleType(),  True),
    T.StructField("contact",     T.IntegerType(), True),
    T.StructField("crashed",     T.IntegerType(), True),
    T.StructField("p_gain_mult", T.DoubleType(),  True),
    T.StructField("seed",        T.IntegerType(), True),
    T.StructField("drone_id",    T.IntegerType(), True),
    T.StructField("ctrl_freq",   T.IntegerType(), True),
    T.StructField("disturbance_type", T.StringType(), True),
    T.StructField("event_time",  T.StringType(),  True),
])

# ── applyInPandas 출력 스키마 ─────────────────────────────────────────────────
SERVING_SCHEMA = T.StructType([
    T.StructField("window_start",    T.StringType(),  True),
    T.StructField("window_end",      T.StringType(),  True),
    T.StructField("drone_id",        T.IntegerType(), True),
    T.StructField("sample_count",    T.IntegerType(), True),
    # R1~R6 feature 값
    T.StructField("alt_rmse_val",    T.DoubleType(),  True),
    T.StructField("tilt_max_val",    T.DoubleType(),  True),
    T.StructField("ang_rate_rms_val",T.DoubleType(),  True),
    T.StructField("vib_ratio_val",   T.DoubleType(),  True),
    T.StructField("crash_val",       T.DoubleType(),  True),
    T.StructField("conv_fail_val",   T.DoubleType(),  True),
    # 예측 결과
    T.StructField("risk_prob",       T.DoubleType(),  True),
    T.StructField("is_alert",        T.IntegerType(), True),
])


# ── ③ 핵심 UDF: feature 계산 + RF 예측 ───────────────────────────────────────
def compute_and_predict(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    applyInPandas 함수.
    같은 (window, drone_id) 그룹의 텔레메트리 행들을 받아
    R1~R6 feature 계산 + RF 위험확률 예측 결과를 반환.

    stability_metrics.evaluate_window 재사용 (중복 구현 금지).
    """
    from stability_metrics import evaluate_window, StabilityThresholds

    if len(pdf) < 8:  # 윈도우가 너무 짧으면 스킵
        return pd.DataFrame(columns=[f.name for f in SERVING_SCHEMA])

    # step 기준 정렬 (FFT 시계열 순서 보장 — Stage 1 설계 원칙)
    pdf = pdf.sort_values("step").reset_index(drop=True)

    # window 메타
    window_start = str(pdf["event_time"].iloc[0])
    window_end   = str(pdf["event_time"].iloc[-1])
    drone_id     = int(pdf["drone_id"].iloc[0])

    # target_z: S2는 시점별로 다를 수 있으므로 중앙값 사용
    target_z = float(pdf["target_z"].median()) if "target_z" in pdf.columns else 1.0

    # R1~R6 feature 계산 (evaluate_window 재사용)
    window_dict = {col: pdf[col].tolist()
                   for col in ["z", "roll", "pitch", "wx", "wy", "wz",
                                "contact", "ctrl_freq"]
                   if col in pdf.columns}
    thr = StabilityThresholds()
    result = evaluate_window(window_dict, target_z=target_z, thr=thr)

    feat_vals = {
        "alt_rmse_val":     result.get("alt_rmse_val",    0.0),
        "tilt_max_val":     result.get("tilt_max_val",    0.0),
        "ang_rate_rms_val": result.get("ang_rate_rms_val",0.0),
        "vib_ratio_val":    result.get("vib_ratio_val",   0.0),
        "crash_val":        float(result.get("R5", False)),
        "conv_fail_val":    float(result.get("R6", False)),
    }

    # ② RF 예측 (모델+scaler 캐시에서 로드)
    model, scaler, feature_cols = _get_model()
    X_raw = [[feat_vals[c] for c in feature_cols]]
    X_scaled = scaler.transform(X_raw)
    risk_prob = float(model.predict_proba(X_scaled)[0][1])
    is_alert  = int(risk_prob >= ALERT_THRESH)

    # 경보 시 콘솔 출력 (드라이버가 아닌 worker에서 print — 로컬 모드에선 보임)
    if is_alert:
        print(f"🚨 [ALERT] drone_id={drone_id} "
              f"risk_prob={risk_prob:.3f} "
              f"R1={feat_vals['alt_rmse_val']:.3f} "
              f"window_end={window_end}")

    row = {
        "window_start":    window_start,
        "window_end":      window_end,
        "drone_id":        drone_id,
        "sample_count":    len(pdf),
        **feat_vals,
        "risk_prob":       risk_prob,
        "is_alert":        is_alert,
    }
    return pd.DataFrame([row])


# ── Spark 파이프라인 ──────────────────────────────────────────────────────────
def _build_spark() -> SparkSession:
    builder = (SparkSession.builder
               .appName("drone-early-warning-serving")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled",
                       "false"))
    if not cfg.is_s3:
        builder = builder.master("local[*]")
    return builder.getOrCreate()


def build_serving_pipeline(spark: SparkSession):
    """
    Kafka → 파싱 → 워터마크 → 윈도우 집계 →
    applyInPandas(feature + RF 예측) → 출력 DataFrame.
    """
    # Kafka 읽기 옵션
    opts = {
        "kafka.bootstrap.servers": cfg.kafka_bootstrap,
        "subscribe":               cfg.kafka_topic,
        "startingOffsets":         "latest",
        "failOnDataLoss":          "false",
    }

    df_raw = spark.readStream.format("kafka")
    for k, v in opts.items():
        df_raw = df_raw.option(k, v)
    df_raw = df_raw.load()

    df_parsed = df_raw.select(
        F.from_json(F.col("value").cast("string"), TELEMETRY_SCHEMA).alias("d")
    ).select("d.*")

    df_timed = df_parsed.withColumn(
        "event_time", F.to_timestamp(F.col("event_time"))
    )

    df_watermarked = df_timed.withWatermark("event_time", WATERMARK_DELAY)

    # (window, drone_id) 그룹으로 collect_list
    df_grouped = df_watermarked.groupBy(
        F.window("event_time", WINDOW_DURATION, SLIDE_DURATION),
        F.col("drone_id"),
    ).agg(
        # applyInPandas에 필요한 모든 컬럼을 struct로 수집
        F.collect_list(
            F.struct("step", "t", "z", "roll", "pitch",
                     "wx", "wy", "wz", "contact",
                     "target_z", "ctrl_freq",
                     "drone_id", "event_time")
        ).alias("rows"),
        F.count("*").alias("sample_count"),
    )

    # collect_list를 explode해서 applyInPandas에 넘길 flat DataFrame 재구성
    df_flat = df_grouped.select(
        F.col("window.start").cast("string").alias("win_start"),
        F.col("window.end").cast("string").alias("win_end"),
        F.col("drone_id"),
        F.explode(F.col("rows")).alias("r"),
    ).select(
        F.col("win_start"),
        F.col("win_end"),
        F.col("drone_id"),
        F.col("r.step").alias("step"),
        F.col("r.t").alias("t"),
        F.col("r.z").alias("z"),
        F.col("r.roll").alias("roll"),
        F.col("r.pitch").alias("pitch"),
        F.col("r.wx").alias("wx"),
        F.col("r.wy").alias("wy"),
        F.col("r.wz").alias("wz"),
        F.col("r.contact").alias("contact"),
        F.col("r.target_z").alias("target_z"),
        F.col("r.ctrl_freq").alias("ctrl_freq"),
        F.col("r.event_time").alias("event_time"),
    )

    # applyInPandas: (win_start, win_end, drone_id) 그룹별 예측
    df_predictions = df_flat.groupBy(
        "win_start", "win_end", "drone_id"
    ).applyInPandas(compute_and_predict, schema=SERVING_SCHEMA)

    return df_predictions


def main():
    print(cfg.summary())
    print(f"[serving] 모델: {MODEL_PATH}")
    print(f"[serving] 경보 임계값: {ALERT_THRESH}")

    spark = _build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # worker에 stability_metrics.py 배포 (spark_consumer.py와 동일)
    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stability_metrics.py")
    if os.path.exists(metrics_path):
        spark.sparkContext.addPyFile(metrics_path)

    # 로컬 경로 생성
    serving_sink = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "serving"
    )
    serving_ckpt = os.path.join(serving_sink, "checkpoint")
    if not cfg.is_s3:
        os.makedirs(serving_sink, exist_ok=True)
        os.makedirs(serving_ckpt, exist_ok=True)

    df_predictions = build_serving_pipeline(spark)

    # 콘솔 출력 (실시간 확인용)
    query_console = (
        df_predictions.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .option("numRows", 10)
        .option("checkpointLocation", os.path.join(serving_ckpt, "console"))
        .trigger(processingTime="5 seconds")
        .start()
    )

    # Parquet 싱크 (서빙 결과 저장)
    query_parquet = (
        df_predictions.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", serving_sink)
        .option("checkpointLocation", os.path.join(serving_ckpt, "parquet"))
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("[serving] 실시간 조기경보 스트리밍 시작. Ctrl+C로 종료.")
    print(f"  Kafka  : {cfg.kafka_bootstrap}  토픽: {cfg.kafka_topic}")
    print(f"  윈도우 : {WINDOW_DURATION} / slide {SLIDE_DURATION}")
    print(f"  저장   : {serving_sink}")
    print(f"  경보   : risk_prob >= {ALERT_THRESH}")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\n[serving] 종료 신호, 스트림 정리 중...")
    finally:
        query_console.stop()
        query_parquet.stop()
        spark.stop()


if __name__ == "__main__":
    main()