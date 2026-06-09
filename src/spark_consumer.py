"""
spark_consumer.py — Kafka 토픽 구독 → 슬라이딩 윈도우 → R1~R6 feature → Parquet

설계 결정:
  1. 단일 스트림 경로 (R1~R6 모두)
     SQL 집계(R1·R2·R3·R6) + applyInPandas(R4·R5) 2경로로 쪼개면
     같은 규칙이 두 군데에 중복 구현되어 미묘하게 어긋날 수 있음.
     대신: collect_list(윈도우 SQL) → evaluate_window UDF 하나로 통일.
     Stage 0의 stability_metrics.py가 그대로 재사용됨 → 배치/스트림 일관성 보장.

  2. collect_list → UDF 패턴 (applyInPandas 대신)
     스트리밍 DataFrame에 groupBy().applyInPandas()를 직접 걸면
     "streaming 미지원" 에러를 만날 수 있음 (상태 없는 grouped-map 제약).
     collect_list는 윈도우 SQL 집계로 스트리밍에서 확실히 지원됨.
     collect_list로 윈도우 샘플을 묶은 뒤, 그 배열 컬럼에 UDF 적용 — 안전한 경로.

  3. collect_list 순서 보장
     셔플 후 collect_list 순서는 비결정적.
     FFT는 시간순 시계열이 필요하므로 UDF 안에서 step 기준 sort_values 필수.
     struct 안에 step을 포함해서 UDF가 정렬 가능하게 함.

  4. event_time = 시뮬 시간 기반 합성값 (producer.py에서 생성)
     window(2초, slide 0.5초)가 물리적으로 의미 있으려면 여기에 맞춰야 함.

  5. ctrl_freq를 텔레메트리에서 읽어 FFT fs로 사용
     하드코딩 fs=240 금지 — pyb_freq != ctrl_freq인 경우 주파수 대역 5배 어긋남.

윈도우 설정:
  window: 2초, slide: 0.5초
  → FFT 주파수 해상도: 1/2 = 0.5Hz (5~15Hz 대역 충분)
  → 탐지 지연: 최대 2초

실행:
  spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\
    src/spark_consumer.py
"""

from __future__ import annotations

import os
import sys

# stability_metrics.py 경로를 Spark worker가 찾을 수 있도록 sys.path 추가
# (spark-submit 실행 위치에 따라 달라질 수 있음)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ── 설정 ──────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "drone-telemetry"

# 출력 경로
PARQUET_OUTPUT = "data/streaming/features"
CHECKPOINT_DIR = "data/streaming/checkpoint"

# 윈도우 / 워터마크 설정
WINDOW_DURATION = "2 seconds"
SLIDE_DURATION = "0.5 seconds"
WATERMARK_DELAY = "5 seconds"   # 최대 5초 늦은 메시지까지 허용


# ── Kafka 메시지 JSON 스키마 ──────────────────────────────────────────────────
# from_json에 명시적 스키마를 주어야 함 (schema inference는 스트리밍에서 느리고 불안정)
TELEMETRY_SCHEMA = T.StructType([
    T.StructField("step",       T.IntegerType(),  True),
    T.StructField("t",          T.DoubleType(),   True),
    T.StructField("x",          T.DoubleType(),   True),
    T.StructField("y",          T.DoubleType(),   True),
    T.StructField("z",          T.DoubleType(),   True),
    T.StructField("qx",         T.DoubleType(),   True),
    T.StructField("qy",         T.DoubleType(),   True),
    T.StructField("qz",         T.DoubleType(),   True),
    T.StructField("qw",         T.DoubleType(),   True),
    T.StructField("roll",       T.DoubleType(),   True),
    T.StructField("pitch",      T.DoubleType(),   True),
    T.StructField("yaw",        T.DoubleType(),   True),
    T.StructField("vx",         T.DoubleType(),   True),
    T.StructField("vy",         T.DoubleType(),   True),
    T.StructField("vz",         T.DoubleType(),   True),
    T.StructField("wx",         T.DoubleType(),   True),
    T.StructField("wy",         T.DoubleType(),   True),
    T.StructField("wz",         T.DoubleType(),   True),
    T.StructField("rpm0",       T.DoubleType(),   True),
    T.StructField("rpm1",       T.DoubleType(),   True),
    T.StructField("rpm2",       T.DoubleType(),   True),
    T.StructField("rpm3",       T.DoubleType(),   True),
    T.StructField("target_x",   T.DoubleType(),   True),
    T.StructField("target_y",   T.DoubleType(),   True),
    T.StructField("target_z",   T.DoubleType(),   True),
    T.StructField("mass_factor",T.DoubleType(),   True),
    T.StructField("wind_x",     T.DoubleType(),   True),
    T.StructField("wind_y",     T.DoubleType(),   True),
    T.StructField("wind_z",     T.DoubleType(),   True),
    T.StructField("contact",    T.IntegerType(),  True),
    T.StructField("crashed",    T.IntegerType(),  True),
    T.StructField("p_gain_mult",T.DoubleType(),   True),
    T.StructField("seed",       T.IntegerType(),  True),
    T.StructField("drone_id",   T.IntegerType(),  True),
    T.StructField("ctrl_freq",  T.IntegerType(),  True),
    # producer가 합성해서 추가하는 필드
    T.StructField("event_time", T.StringType(),   True),
])


# ── R1~R6 feature UDF ─────────────────────────────────────────────────────────
# collect_list가 묶은 윈도우 샘플 배열을 받아 evaluate_window()를 호출.
# 반환 스키마가 Parquet 컬럼 구조를 결정함.
FEATURE_SCHEMA = T.StructType([
    T.StructField("alt_rmse_val",     T.DoubleType(),  True),   # R1
    T.StructField("tilt_max_val",     T.DoubleType(),  True),   # R2
    T.StructField("ang_rate_rms_val", T.DoubleType(),  True),   # R3
    T.StructField("vib_ratio_val",    T.DoubleType(),  True),   # R4
    T.StructField("crash_val",        T.DoubleType(),  True),   # R5
    T.StructField("conv_fail_val",    T.DoubleType(),  True),   # R6
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
    """collect_list로 수집한 윈도우 샘플 배열 → R1~R6 feature 딕셔너리.

    주의사항:
      - collect_list는 순서를 보장하지 않으므로 step 기준으로 정렬 필수
        (정렬 없으면 FFT가 쓰레기값을 반환)
      - ctrl_freq를 샘플에서 읽어 FFT fs로 사용 (하드코딩 금지)
      - 반환값은 Python 기본 타입이어야 함 (np.float64 등 numpy 타입 불가)
      - 샘플 수 부족 시 None 반환 → Parquet에 null로 저장됨
    """
    if not samples or len(samples) < 8:
        # FFT를 돌리기에 샘플이 너무 적으면 null 반환
        return None

    # ── step 기준 정렬 (collect_list 순서 비결정적) ────────────────────────
    samples_sorted = sorted(samples, key=lambda r: r["step"])

    # ctrl_freq: 첫 번째 샘플에서 읽음 (한 윈도우 내에서 동일한 값)
    ctrl_freq = samples_sorted[0]["ctrl_freq"] or 240
    target_z = float(samples_sorted[0]["target_z"] or 1.0)

    # ── evaluate_window 입력 형식으로 변환 ────────────────────────────────
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
            window_data,
            target_z=target_z,
            thr=StabilityThresholds(),
            fs=float(ctrl_freq),   # 텔레메트리에서 읽은 실제 샘플링 레이트
        )
    except Exception:
        return None

    # numpy 타입 → Python 기본 타입 변환 (Spark UDF 직렬화 요건)
    return (
        float(result["alt_rmse_val"]),
        float(result["tilt_max_val"]),
        float(result["ang_rate_rms_val"]),
        float(result["vib_ratio_val"]),
        float(result["crash_val"]),
        float(result["conv_fail_val"]),
        bool(result["R1"]),
        bool(result["R2"]),
        bool(result["R3"]),
        bool(result["R4"]),
        bool(result["R5"]),
        bool(result["R6"]),
        int(result["severity"]),
        bool(result["unstable"]),
    )


# ── 메인 스트리밍 파이프라인 ──────────────────────────────────────────────────
def build_pipeline(spark: SparkSession):
    """Kafka → 파싱 → 워터마크 → 윈도우 집계 → UDF → DataFrame 반환."""

    # 1. Kafka 읽기
    df_raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        # latest: consumer 시작 시점 이후의 메시지만 읽음
        # earliest로 바꾸면 토픽 처음부터 읽음 (재처리용)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # 2. JSON 파싱 — 명시적 스키마 사용 (schema inference 금지)
    df_parsed = df_raw.select(
        F.from_json(F.col("value").cast("string"), TELEMETRY_SCHEMA).alias("d")
    ).select("d.*")

    # 3. event_time 타입 변환 (ISO 문자열 → timestamp)
    #    producer가 합성한 시뮬 시간 기반 타임스탬프
    df_timed = df_parsed.withColumn(
        "event_time", F.to_timestamp(F.col("event_time"))
    )

    # 4. 워터마크 설정 — 최대 5초 늦은 메시지까지 허용
    #    append 출력 모드를 쓰려면 워터마크가 반드시 있어야 함
    df_watermarked = df_timed.withWatermark("event_time", WATERMARK_DELAY)

    # 5. 슬라이딩 윈도우 집계
    #    - window 2초 / slide 0.5초: FFT 주파수 해상도 0.5Hz, 탐지 지연 최대 2초
    #    - collect_list: 윈도우 내 원시 샘플 전체 수집
    #      struct 안에 step을 포함 → UDF 안에서 시간순 정렬 가능
    #    - applyInPandas 대신 collect_list + UDF 패턴:
    #      스트리밍에서 groupBy().applyInPandas() 직접 사용 시 제약이 있음
    df_windowed = df_watermarked.groupBy(
        F.window("event_time", WINDOW_DURATION, SLIDE_DURATION),
        F.col("drone_id"),
    ).agg(
        F.collect_list(
            F.struct(
                "step", "z", "roll", "pitch",
                "wx", "wy", "wz", "contact",
                "target_z", "ctrl_freq",
            )
        ).alias("samples"),
        F.count("*").alias("sample_count"),
        # 디버깅용 추가 집계 (Parquet에도 저장)
        F.first("p_gain_mult").alias("p_gain_mult"),
        F.first("mass_factor").alias("mass_factor"),
    )

    # 6. R1~R6 feature 계산 — Stage 0의 evaluate_window 재사용
    df_features = df_windowed.withColumn(
        "features", compute_features_udf(F.col("samples"))
    ).filter(
        F.col("features").isNotNull()   # 샘플 수 부족 윈도우 제외
    ).select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        F.col("drone_id"),
        F.col("sample_count"),
        F.col("p_gain_mult"),
        F.col("mass_factor"),
        # feature struct를 최상위 컬럼으로 펼침
        F.col("features.*"),
    )

    return df_features


def main():
    # ── SparkSession 초기화 ────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("drone-early-warning-stage1")
        .master("local[*]")
        # 스트리밍 집계에서 중간 상태를 디스크에 spill 허용
        .config("spark.sql.shuffle.partitions", "4")
        # watermark + append 모드 정확성 검사 (개발 중에는 warning 허용)
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # stability_metrics.py를 모든 Spark worker에서 참조 가능하게 등록
    metrics_path = os.path.join(os.path.dirname(__file__), "stability_metrics.py")
    if os.path.exists(metrics_path):
        spark.sparkContext.addPyFile(metrics_path)

    os.makedirs(PARQUET_OUTPUT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df_features = build_pipeline(spark)

    # ── 싱크 1: Parquet (append 모드) ────────────────────────────────────
    # 워터마크가 지나면 윈도우 결과를 확정하고 append
    query_parquet = (
        df_features.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", PARQUET_OUTPUT)
        .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "parquet"))
        # drone_id와 날짜로 파티셔닝 (Stage 2 데이터레이크 설계 대비)
        .partitionBy("drone_id")
        .trigger(processingTime="5 seconds")
        .start()
    )

    # ── 싱크 2: 콘솔 (디버깅용) ──────────────────────────────────────────
    query_console = (
        df_features.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .option("numRows", 5)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("[consumer] 스트리밍 시작. Ctrl+C로 종료.")
    print(f"  Kafka: {KAFKA_BOOTSTRAP}  토픽: {TOPIC}")
    print(f"  윈도우: {WINDOW_DURATION} / slide {SLIDE_DURATION}")
    print(f"  출력 경로: {PARQUET_OUTPUT}")

    try:
        # 두 쿼리 중 하나라도 종료되면 같이 종료
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\n[consumer] 종료 신호 수신, 스트림 정리 중...")
    finally:
        query_parquet.stop()
        query_console.stop()
        spark.stop()


if __name__ == "__main__":
    main()
