# Stage 1 — MVP: 드론 sim → Kafka → Spark → Parquet

## 버전 조합

| 구성 요소                  | 버전                  | 비고                               |
| -------------------------- | --------------------- | ---------------------------------- |
| Kafka (Confluent Platform) | 7.4.0 (= Kafka 3.4.x) | ZooKeeper 포함                     |
| pyspark                    | 3.5.1                 | pip 설치, Spark 컨테이너 불필요    |
| spark-sql-kafka 커넥터     | 3.5.1 (Scala 2.12)    | spark-submit 실행 시 자동 다운로드 |
| kafka-python               | 2.0.2                 | producer용                         |
| Java                       | 11 또는 17            | `java -version`으로 확인 필수      |

## 전제 조건 확인

```bash
# Java 버전 확인 (8/11/17 중 하나여야 함)
java -version

# Apple Silicon이면 Homebrew로 설치
brew install openjdk@17
export JAVA_HOME=$(brew --prefix openjdk@17)
```

## 설치

```bash
pip install kafka-python==2.0.2 pyspark==3.5.1
```

## 실행 순서

### 1. Kafka + ZooKeeper 컨테이너 시작

```bash
docker compose up -d
# 준비 확인 (kafka가 healthy 상태가 될 때까지 20~30초 소요)
docker compose ps
```

### 2. Kafka 토픽 생성

```bash
docker exec drone-kafka \
  kafka-topics --create \
  --topic drone-telemetry \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

토픽 확인:

```bash
docker exec drone-kafka \
  kafka-topics --list --bootstrap-server localhost:9092
```

### 3. Spark consumer 시작 (새 터미널)

```bash
# 첫 실행 시 spark-sql-kafka jar 다운로드 (인터넷 필요, 약 1분)
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  src/spark_consumer.py
```

`[consumer] 스트리밍 시작` 메시지가 뜨면 준비 완료.

### 4. Producer 실행 (다른 터미널)

```bash
# 기본값: 점진 붕괴 시나리오 (payload 2.5×, 3초 ramp)
python src/producer.py

# 안정 시나리오 (외란 없음)
python src/producer.py --payload_factor 1.0 --wind_mode none --duration 8
```

### 5. 결과 확인

consumer 터미널에 윈도우 feature 행이 출력되면 end-to-end 성공.

```bash
# Parquet 파일 생성 확인
ls data/streaming/features/

# Python으로 읽기
python -c "
import pandas as pd
df = pd.read_parquet('data/streaming/features/')
print(df[['window_start','window_end','drone_id','alt_rmse_val','unstable']].head(10))
"
```

## 데이터 흐름 요약

```
simulate.py  (헤드리스, ~6배 실시간)
    │ step/ctrl_freq → event_time 합성 (epoch_base + step/ctrl_freq)
    ▼
producer.py  →  Kafka 토픽: drone-telemetry  (파티션 키: drone_id)
                    │
                    ▼
            spark_consumer.py
            ├─ from_json (명시적 스키마)
            ├─ withWatermark("event_time", "5s")
            ├─ groupBy(window 2s/0.5s, drone_id)
            ├─ collect_list(struct(step, z, roll, pitch, wx, wy, wz, contact, ctrl_freq))
            └─ compute_features_udf → evaluate_window (Stage 0 재사용)
                    │
            ┌───────┴───────┐
            ▼               ▼
        Parquet          console
  data/streaming/features/  (디버깅)
```

## 핵심 설계 결정 (코드에 주석으로도 기재)

**event_time을 시뮬 시간 기반으로 합성하는 이유**
헤드리스 sim이 실시간보다 ~6배 빠르다. `datetime.now()`를 쓰면 수십 초 분량 데이터가
벽시계 몇 초에 압축돼, window(2초)가 "시뮬 2초"가 아닌 "벽시계 2초"가 된다.
FFT 주파수 해석과 윈도우 경계 모두 무의미해짐.

**ctrl_freq를 텔레메트리에 포함하는 이유**
FFT UDF가 `fs=ctrl_freq`로 주파수 대역을 계산한다.
하드코딩 `fs=240`을 쓰면 ctrl_freq != pyb_freq인 경우 대역이 수배 어긋난다.

**collect_list + UDF (applyInPandas 대신)**
스트리밍 DataFrame에 `groupBy().applyInPandas()` 직접 사용 시 버전 의존적 제약 존재.
`collect_list`(윈도우 SQL 집계)는 워터마크 + append 모드에서 확실히 지원된다.

**UDF 안에서 step 기준 정렬**
`collect_list`는 순서를 보장하지 않는다.
FFT는 시간순 시계열이 필요하므로 `sorted(samples, key=lambda r: r["step"])` 필수.

## 로컬 로직 검증 (Kafka/Spark 없이)

```bash
python src/test_pipeline_local.py
# 16개 PASS가 나오면 파이프라인 로직 정상
```
