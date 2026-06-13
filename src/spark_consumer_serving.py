"""
spark_consumer_serving.py — Stage 3 실시간 조기경보 서빙.

Kafka 텔레메트리를 2초 윈도우/0.5초 슬라이드로 집계하고,
R1~R6 feature 계산과 RandomForest 위험확률 예측을 수행한다.

출력 구조:
  data/serving/output/batch_id=<N>/*.parquet
  data/checkpoints/spark-serving/...

출력 데이터와 체크포인트를 분리하고, 단일 foreachBatch 쿼리에서
콘솔 출력과 Parquet 저장을 함께 수행한다.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg


WINDOW_DURATION = "2 seconds"
SLIDE_DURATION = "0.5 seconds"
WATERMARK_DELAY = "5 seconds"
ALERT_THRESH = 0.6

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "randomforest.pkl",
)
SERVING_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "serving",
    "output",
)
SERVING_CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "checkpoints",
    "spark-serving",
)


# applyInPandas worker 프로세스별 모델 캐시
_MODEL_CACHE: dict = {}


def _get_model():
    """모델, scaler, feature_cols를 worker별로 한 번만 로드한다."""
    if not _MODEL_CACHE:
        import joblib

        artifact = joblib.load(MODEL_PATH)
        _MODEL_CACHE["model"] = artifact["model"]
        _MODEL_CACHE["scaler"] = artifact["scaler"]
        _MODEL_CACHE["features"] = artifact["feature_cols"]

    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["scaler"],
        _MODEL_CACHE["features"],
    )


TELEMETRY_SCHEMA = T.StructType([
    T.StructField("step", T.IntegerType(), True),
    T.StructField("t", T.DoubleType(), True),
    T.StructField("x", T.DoubleType(), True),
    T.StructField("y", T.DoubleType(), True),
    T.StructField("z", T.DoubleType(), True),
    T.StructField("qx", T.DoubleType(), True),
    T.StructField("qy", T.DoubleType(), True),
    T.StructField("qz", T.DoubleType(), True),
    T.StructField("qw", T.DoubleType(), True),
    T.StructField("roll", T.DoubleType(), True),
    T.StructField("pitch", T.DoubleType(), True),
    T.StructField("yaw", T.DoubleType(), True),
    T.StructField("vx", T.DoubleType(), True),
    T.StructField("vy", T.DoubleType(), True),
    T.StructField("vz", T.DoubleType(), True),
    T.StructField("wx", T.DoubleType(), True),
    T.StructField("wy", T.DoubleType(), True),
    T.StructField("wz", T.DoubleType(), True),
    T.StructField("rpm0", T.DoubleType(), True),
    T.StructField("rpm1", T.DoubleType(), True),
    T.StructField("rpm2", T.DoubleType(), True),
    T.StructField("rpm3", T.DoubleType(), True),
    T.StructField("target_x", T.DoubleType(), True),
    T.StructField("target_y", T.DoubleType(), True),
    T.StructField("target_z", T.DoubleType(), True),
    T.StructField("mass_factor", T.DoubleType(), True),
    T.StructField("wind_x", T.DoubleType(), True),
    T.StructField("wind_y", T.DoubleType(), True),
    T.StructField("wind_z", T.DoubleType(), True),
    T.StructField("contact", T.IntegerType(), True),
    T.StructField("crashed", T.IntegerType(), True),
    T.StructField("p_gain_mult", T.DoubleType(), True),
    T.StructField("seed", T.IntegerType(), True),
    T.StructField("drone_id", T.IntegerType(), True),
    T.StructField("ctrl_freq", T.IntegerType(), True),
    T.StructField("disturbance_type", T.StringType(), True),
    T.StructField("event_time", T.StringType(), True),
])


SERVING_SCHEMA = T.StructType([
    T.StructField("window_start", T.StringType(), True),
    T.StructField("window_end", T.StringType(), True),
    T.StructField("drone_id", T.IntegerType(), True),
    T.StructField("sample_count", T.IntegerType(), True),
    T.StructField("alt_rmse_val", T.DoubleType(), True),
    T.StructField("tilt_max_val", T.DoubleType(), True),
    T.StructField("ang_rate_rms_val", T.DoubleType(), True),
    T.StructField("vib_ratio_val", T.DoubleType(), True),
    T.StructField("crash_val", T.DoubleType(), True),
    T.StructField("conv_fail_val", T.DoubleType(), True),
    T.StructField("risk_prob", T.DoubleType(), True),
    T.StructField("is_alert", T.IntegerType(), True),
])


def compute_and_predict(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    같은 (window, drone_id) 그룹에 대해 feature와 위험확률을 계산한다.
    """
    from stability_metrics import evaluate_window, StabilityThresholds

    if len(pdf) < 8:
        return pd.DataFrame(columns=[field.name for field in SERVING_SCHEMA])

    pdf = pdf.sort_values("step").reset_index(drop=True)

    # Spark window 경계를 그대로 사용한다.
    window_start = str(pdf["win_start"].iloc[0])
    window_end = str(pdf["win_end"].iloc[0])
    drone_id = int(pdf["drone_id"].iloc[0])

    target_z = (
        float(pdf["target_z"].median())
        if "target_z" in pdf.columns
        else 1.0
    )

    window_dict = {
        col: pdf[col].tolist()
        for col in [
            "z",
            "roll",
            "pitch",
            "wx",
            "wy",
            "wz",
            "contact",
            "ctrl_freq",
        ]
        if col in pdf.columns
    }

    result = evaluate_window(
        window_dict,
        target_z=target_z,
        thr=StabilityThresholds(),
    )

    feat_vals = {
        "alt_rmse_val": float(result.get("alt_rmse_val", 0.0)),
        "tilt_max_val": float(result.get("tilt_max_val", 0.0)),
        "ang_rate_rms_val": float(result.get("ang_rate_rms_val", 0.0)),
        "vib_ratio_val": float(result.get("vib_ratio_val", 0.0)),
        "crash_val": float(result.get("R5", False)),
        "conv_fail_val": float(result.get("R6", False)),
    }

    model, scaler, feature_cols = _get_model()
    x_raw = [[feat_vals[col] for col in feature_cols]]
    x_scaled = scaler.transform(x_raw)

    risk_prob = float(model.predict_proba(x_scaled)[0][1])
    is_alert = int(risk_prob >= ALERT_THRESH)

    if is_alert:
        print(
            f"🚨 [ALERT] drone_id={drone_id} "
            f"risk_prob={risk_prob:.3f} "
            f"R1={feat_vals['alt_rmse_val']:.3f} "
            f"window_end={window_end}"
        )

    return pd.DataFrame([{
        "window_start": window_start,
        "window_end": window_end,
        "drone_id": drone_id,
        "sample_count": int(len(pdf)),
        **feat_vals,
        "risk_prob": risk_prob,
        "is_alert": is_alert,
    }])


def _build_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("drone-early-warning-serving")
        .config("spark.sql.shuffle.partitions", "1")
        .config(
            "spark.sql.streaming.statefulOperator.checkCorrectness.enabled",
            "false",
        )
    )

    # spark-submit --master가 있으면 해당 설정이 우선 적용된다.
    if not cfg.is_s3:
        builder = builder.master("local[1]")

    return builder.getOrCreate()


def build_serving_pipeline(spark: SparkSession):
    """Kafka → 파싱 → 워터마크 → 윈도우 → feature/RF 예측."""
    kafka_options = {
        "kafka.bootstrap.servers": cfg.kafka_bootstrap,
        "subscribe": cfg.kafka_topic,
        "startingOffsets": "latest",
        "failOnDataLoss": "false",
    }

    df_raw = spark.readStream.format("kafka")
    for key, value in kafka_options.items():
        df_raw = df_raw.option(key, value)
    df_raw = df_raw.load()

    df_parsed = (
        df_raw
        .select(
            F.from_json(
                F.col("value").cast("string"),
                TELEMETRY_SCHEMA,
            ).alias("d")
        )
        .select("d.*")
        .filter(F.col("event_time").isNotNull())
        .filter(F.col("drone_id").isNotNull())
    )

    df_timed = df_parsed.withColumn(
        "event_time",
        F.to_timestamp(F.col("event_time")),
    )

    df_watermarked = df_timed.withWatermark(
        "event_time",
        WATERMARK_DELAY,
    )

    df_grouped = (
        df_watermarked
        .groupBy(
            F.window(
                "event_time",
                WINDOW_DURATION,
                SLIDE_DURATION,
            ),
            F.col("drone_id"),
        )
        .agg(
            F.collect_list(
                F.struct(
                    "step",
                    "t",
                    "z",
                    "roll",
                    "pitch",
                    "wx",
                    "wy",
                    "wz",
                    "contact",
                    "target_z",
                    "ctrl_freq",
                    "drone_id",
                    "event_time",
                )
            ).alias("rows"),
            F.count("*").alias("sample_count"),
        )
    )

    df_flat = (
        df_grouped
        .select(
            F.col("window.start").cast("string").alias("win_start"),
            F.col("window.end").cast("string").alias("win_end"),
            F.col("drone_id"),
            F.explode(F.col("rows")).alias("r"),
        )
        .select(
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
    )

    return (
        df_flat
        .groupBy("win_start", "win_end", "drone_id")
        .applyInPandas(
            compute_and_predict,
            schema=SERVING_SCHEMA,
        )
    )


def main() -> None:
    print(cfg.summary())
    print(f"[serving] 모델: {MODEL_PATH}")
    print(f"[serving] 경보 임계값: {ALERT_THRESH}")
    print(f"[serving] Parquet 출력: {SERVING_OUTPUT_DIR}")
    print(f"[serving] 체크포인트: {SERVING_CHECKPOINT_DIR}")

    os.makedirs(SERVING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SERVING_CHECKPOINT_DIR, exist_ok=True)

    spark = _build_spark()
    spark.sparkContext.setLogLevel("WARN")

    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stability_metrics.py",
    )
    if os.path.exists(metrics_path):
        spark.sparkContext.addPyFile(metrics_path)

    df_predictions = build_serving_pipeline(spark)

    def write_serving_batch(batch_df, batch_id: int) -> None:
        """
        micro-batch 하나를 콘솔에 출력하고 Parquet으로 저장한다.
        """
        from pyspark import StorageLevel

        cached = batch_df.persist(StorageLevel.MEMORY_AND_DISK)
        try:
            row_count = cached.count()

            print("-" * 43)
            print(f"Batch: {batch_id} | rows: {row_count}")
            print("-" * 43)

            if row_count == 0:
                return

            cached.show(n=10, truncate=False)

            batch_output_dir = os.path.join(
                SERVING_OUTPUT_DIR,
                f"batch_id={batch_id}",
            )

            (
                cached.write
                .mode("overwrite")
                .parquet(batch_output_dir)
            )

            print(
                f"[serving] batch={batch_id} "
                f"{row_count}행 저장 완료: {batch_output_dir}"
            )
        finally:
            cached.unpersist()

    query = (
        df_predictions.writeStream
        .outputMode("append")
        .foreachBatch(write_serving_batch)
        .option(
            "checkpointLocation",
            SERVING_CHECKPOINT_DIR,
        )
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("[serving] 실시간 조기경보 스트리밍 시작. Ctrl+C로 종료.")
    print(f"  Kafka      : {cfg.kafka_bootstrap}  토픽: {cfg.kafka_topic}")
    print(f"  윈도우     : {WINDOW_DURATION} / slide {SLIDE_DURATION}")
    print(f"  Parquet    : {SERVING_OUTPUT_DIR}")
    print(f"  Checkpoint : {SERVING_CHECKPOINT_DIR}")
    print(f"  경보       : risk_prob >= {ALERT_THRESH}")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[serving] 종료 신호, 스트림 정리 중...")
    finally:
        if query.isActive:
            query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
