"""
spark_consumer.py — Kafka 토픽 구독 → 슬라이딩 윈도우 → R1~R6 feature → Parquet

핵심 설계 결정 (Stage 1 확정, 이 전환에서 절대 변경 없음):
  - collect_list + evaluate_window UDF 단일 경로 (R1~R6 모두)
  - collect_list 후 step 기준 정렬 (FFT 시계열 순서 보장)
  - ctrl_freq를 텔레메트리에서 읽어 FFT fs로 사용 (하드코딩 금지)
  - event_time = 시뮬 시간 기반 합성값 (producer가 생성)
  - window 2초 / slide 0.5초

연결 지점: config.py(환경변수)에서 읽음.
전환 방법: ENV_FILE=.env.s3 spark-submit ... src/spark_consumer.py

hadoop-aws 버전 핀:
  pyspark 3.5.x 번들 Hadoop 버전과 반드시 일치해야 함.
  ClassNotFound 방지를 위해 2단계(S3 전환) 직전에 실측 확인:
    spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
  현재 핀: hadoop-aws:3.3.4 (pyspark 3.5.1 기본값, 실측 후 조정)

실행:
  # 로컬 (기본값)
  spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:4.1.2,\\
               org.apache.hadoop:hadoop-aws:3.3.4 \\
    src/spark_consumer.py

  # B 전환 (S3 싱크)
  ENV_FILE=.env.s3 spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:4.1.2,\\
               org.apache.hadoop:hadoop-aws:3.3.4 \\
    src/spark_consumer.py

  # B+C 전환 (S3 + EC2 Kafka)
  ENV_FILE=.env.aws spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:4.1.2,\\
               org.apache.hadoop:hadoop-aws:3.3.4 \\
    src/spark_consumer.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import cfg

WINDOW_DURATION = "2 seconds"
SLIDE_DURATION  = "0.5 seconds"
WATERMARK_DELAY = "5 seconds"

# ── Kafka 메시지 JSON 스키마 ──────────────────────────────────────────────────
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
    T.StructField("event_time",  T.StringType(),  True),
])

# ── R1~R6 feature UDF ─────────────────────────────────────────────────────────
FEATURE_SCHEMA = T.StructType([
    T.StructField("alt_rmse_val",     T.DoubleType(),  True),
    T.StructField("tilt_max_val",     T.DoubleType(),  True),
    T.StructField("ang_rate_rms_val", T.DoubleType(),  True),
    T.StructField("vib_ratio_val",    T.DoubleType(),  True),
    T.StructField("crash_val",        T.DoubleType(),  True),
    T.StructField("conv_fail_val",    T.DoubleType(),  True),
    T.StructField("R1",               T.BooleanType(), True),
    T.StructField("R2",               T.BooleanType(), True),
    T.StructField("R3",               T.BooleanType(), True),
    T.StructField("R4",               T.BooleanType(), True),
    T.StructField("R5",               T.BooleanType(), True),
    T.StructField("R6",               T.BooleanType(), True),
    T.StructField("severity",         T.IntegerType(), True),
    T.StructField("unstable",         T.BooleanType(), True),
])


@F.udf(returnType=FEATURE_SCHEMA)
def compute_features_udf(samples):
    """
    collect_list로 수집한 윈도우 샘플 배열 → R1~R6 feature.
    - collect_list 순서 비결정적 → step 기준 정렬 필수
    - ctrl_freq 샘플에서 읽어 FFT fs로 사용 (하드코딩 금지)
    - 반환값은 Python 기본 타입 (numpy 타입 직렬화 불가)
    """
    if not samples or len(samples) < 8:
        return None

    samples_sorted = sorted(samples, key=lambda r: r["step"])
    ctrl_freq = float(samples_sorted[0]["ctrl_freq"] or 240)
    target_z  = float(samples_sorted[0]["target_z"]  or 1.0)

    window_data = {
        "z":       [float(r["z"])       for r in samples_sorted],
        "roll":    [float(r["roll"])    for r in samples_sorted],
        "pitch":   [float(r["pitch"])   for r in samples_sorted],
        "wx":      [float(r["wx"])      for r in samples_sorted],
        "wy":      [float(r["wy"])      for r in samples_sorted],
        "wz":      [float(r["wz"])      for r in samples_sorted],
        "contact": [float(r["contact"]) for r in samples_sorted],
    }

    try:
        from stability_metrics import evaluate_window, StabilityThresholds
        result = evaluate_window(
            window_data, target_z=target_z,
            thr=StabilityThresholds(), fs=ctrl_freq,
        )
    except Exception:
        return None

    return (
        float(result["alt_rmse_val"]),   float(result["tilt_max_val"]),
        float(result["ang_rate_rms_val"]),float(result["vib_ratio_val"]),
        float(result["crash_val"]),       float(result["conv_fail_val"]),
        bool(result["R1"]),  bool(result["R2"]),
        bool(result["R3"]),  bool(result["R4"]),
        bool(result["R5"]),  bool(result["R6"]),
        int(result["severity"]), bool(result["unstable"]),
    )


def _build_spark() -> SparkSession:
    """환경(로컬/S3)에 맞는 SparkSession 생성.

    S3 모드일 때 hadoop-aws 설정 추가.
    hadoop-aws 버전은 pyspark 번들 Hadoop 버전과 일치해야 함.
    2단계(S3 전환) 직전에 실측 확인 권장:
      spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
    """
    builder = (
        SparkSession.builder
        .appName("drone-early-warning-stage1")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
    )

    if cfg.is_s3:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    cfg.s3_credentials_provider)
            .config("spark.hadoop.fs.s3a.endpoint",
                    f"s3.{cfg.aws_region}.amazonaws.com")
        )
        # 로컬 키 방식일 때만 직접 주입 (EC2 instance profile이면 불필요)
        if cfg.aws_access_key_id:
            builder = (
                builder
                .config("spark.hadoop.fs.s3a.access.key", cfg.aws_access_key_id)
                .config("spark.hadoop.fs.s3a.secret.key", cfg.aws_secret_access_key)
            )

    return builder.getOrCreate()


def _kafka_read_options() -> dict:
    """
    환경별 Kafka readStream 옵션.
    현재 검증된 경로: PLAINTEXT (로컬 + EC2 자체호스팅).
    SASL 분기는 "있으면 읽는다" 수준 — 현재 실제 검증 안 함.
    """
    opts = {
        "kafka.bootstrap.servers": cfg.kafka_bootstrap,
        "subscribe":               cfg.kafka_topic,
        "startingOffsets":         "latest",
        "failOnDataLoss":          "false",
    }
    # SASL 분기 (스트레치 옵션, 현재 미검증)
    if not cfg.is_plaintext:
        opts["kafka.security.protocol"] = cfg.kafka_security_protocol
    return opts


def build_pipeline(spark: SparkSession):
    """Kafka → 파싱 → 워터마크 → 윈도우 집계 → UDF → feature DataFrame."""

    df_raw = spark.readStream.format("kafka")
    for k, v in _kafka_read_options().items():
        df_raw = df_raw.option(k, v)
    df_raw = df_raw.load()

    df_parsed = df_raw.select(
        F.from_json(F.col("value").cast("string"), TELEMETRY_SCHEMA).alias("d")
    ).select("d.*")

    df_timed = df_parsed.withColumn(
        "event_time", F.to_timestamp(F.col("event_time"))
    )

    df_watermarked = df_timed.withWatermark("event_time", WATERMARK_DELAY)

    df_windowed = df_watermarked.groupBy(
        F.window("event_time", WINDOW_DURATION, SLIDE_DURATION),
        F.col("drone_id"),
    ).agg(
        F.collect_list(
            F.struct("step", "z", "roll", "pitch",
                     "wx", "wy", "wz", "contact",
                     "target_z", "ctrl_freq")
        ).alias("samples"),
        F.count("*").alias("sample_count"),
        F.first("p_gain_mult").alias("p_gain_mult"),
        F.first("mass_factor").alias("mass_factor"),
    )

    df_features = (
        df_windowed
        .withColumn("features", compute_features_udf(F.col("samples")))
        .filter(F.col("features").isNotNull())
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("drone_id"),
            F.col("sample_count"),
            F.col("p_gain_mult"),
            F.col("mass_factor"),
            F.col("features.*"),
        )
    )
    return df_features


def main():
    print(cfg.summary())

    spark = _build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Spark worker에 stability_metrics.py 배포
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "stability_metrics.py")
    if os.path.exists(metrics_path):
        spark.sparkContext.addPyFile(metrics_path)

    # 로컬 경로면 디렉터리 미리 생성 (S3는 불필요)
    if not cfg.is_s3:
        os.makedirs(cfg.sink_path, exist_ok=True)
        os.makedirs(cfg.checkpoint_path, exist_ok=True)

    df_features = build_pipeline(spark)

    query_parquet = (
        df_features.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", cfg.sink_path)
        .option("checkpointLocation", os.path.join(cfg.checkpoint_path, "parquet"))
        .partitionBy("drone_id")
        .trigger(processingTime="5 seconds")
        .start()
    )

    query_console = (
        df_features.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .option("numRows", 5)
        .option("checkpointLocation", os.path.join(cfg.checkpoint_path, "console"))
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("[consumer] 스트리밍 시작. Ctrl+C로 종료.")
    print(f"  Kafka  : {cfg.kafka_bootstrap}  토픽: {cfg.kafka_topic}")
    print(f"  윈도우 : {WINDOW_DURATION} / slide {SLIDE_DURATION}")
    print(f"  싱크   : {cfg.sink_path}")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\n[consumer] 종료 신호 수신, 스트림 정리 중...")
    finally:
        query_parquet.stop()
        query_console.stop()
        spark.stop()


if __name__ == "__main__":
    main()
