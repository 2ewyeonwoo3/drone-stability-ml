"""
producer.py — 드론 시뮬레이터를 Kafka producer로 연결.

핵심 설계 결정:
  event_time = epoch_base + step / ctrl_freq  (시뮬 시간 기반 합성)

  datetime.now()를 쓰면 안 되는 이유:
    헤드리스 sim이 실시간보다 ~6배 빠르게 돌아서, 수십 초 분량의 시뮬 데이터가
    벽시계 몇 초에 압축돼 들어옴.
    그러면 Spark의 window(2초)가 "시뮬 2초"가 아니라 "벽시계 2초"가 돼
    FFT 주파수 해석도 윈도우 경계도 전부 물리적으로 무의미해짐.

  ctrl_freq도 메시지에 포함:
    Spark consumer의 FFT UDF가 실제 샘플링 레이트(fs)로 사용.
    pyb_freq != ctrl_freq인 경우 잘못된 주파수 대역을 보게 되는 걸 방지.

실행:
  python src/producer.py
  python src/producer.py --p_gain_mult 1.0 --payload_factor 2.5 \\
      --payload_start 2 --payload_ramp 3 --duration 9
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from simulate import run_simulation

# ── 설정 ──────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "drone-telemetry"

# 시뮬 시간을 실제 epoch에 고정하는 기준점.
# 고정값이어야 여러 런이 같은 타임라인에서 재현 가능.
EPOCH_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()

# 행 전송 사이 인위적 딜레이(초). 0이면 전속력으로 전송.
# Spark consumer의 watermark 지연을 눈으로 확인하고 싶을 때 0.005 정도로 올림.
SEND_DELAY_SEC = 0.0


def _make_producer(retries: int = 5) -> KafkaProducer:
    """Kafka에 연결 가능할 때까지 재시도하며 producer 생성."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                # 전송 실패 시 최대 3회 재시도
                retries=3,
                # acks=all: 브로커 수신 확인 후 진행 (데이터 유실 방지)
                acks="all",
            )
            print(f"[producer] Kafka 연결 성공 ({KAFKA_BOOTSTRAP})")
            return producer
        except NoBrokersAvailable:
            print(f"[producer] Kafka 미응답, {attempt}/{retries} 재시도 (5초 대기)...")
            time.sleep(5)
    raise RuntimeError("Kafka 브로커에 연결할 수 없음. docker compose up 상태 확인 필요")


def stream_to_kafka(
    producer: KafkaProducer,
    drone_id_tag: int = 0,
    **sim_kwargs,
) -> int:
    """시뮬레이터를 돌리고 텔레메트리 행을 Kafka에 전송.

    반환값: 전송한 행 수
    """
    ctrl_freq = sim_kwargs.get("ctrl_freq", 240)

    print(f"[producer] 시뮬 시작: drone_id={drone_id_tag}, ctrl_freq={ctrl_freq}Hz")
    rows = run_simulation(**sim_kwargs, drone_id_tag=drone_id_tag, output_path=None)

    for row in rows:
        # ── event_time 합성 ────────────────────────────────────────────────
        # 시뮬 시간(step / ctrl_freq)을 epoch_base에 더해 ISO 타임스탬프 생성.
        # Spark의 to_timestamp()가 그대로 파싱할 수 있는 형식으로 직렬화.
        sim_t: float = row["step"] / ctrl_freq
        event_time_iso = datetime.fromtimestamp(
            EPOCH_BASE + sim_t, tz=timezone.utc
        ).isoformat()

        msg = {**row, "event_time": event_time_iso}

        # 파티션 키: drone_id — 같은 드론의 메시지가 같은 파티션에 순서대로 들어감
        producer.send(TOPIC, key=drone_id_tag, value=msg)

        if SEND_DELAY_SEC > 0:
            time.sleep(SEND_DELAY_SEC)

    producer.flush()
    print(f"[producer] 전송 완료: {len(rows)}행 → 토픽 '{TOPIC}'")
    return len(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="드론 텔레메트리 Kafka producer")
    parser.add_argument("--p_gain_mult", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drone_id", type=int, default=0)
    # 외란 옵션 (기본값 = 점진 붕괴 시나리오)
    parser.add_argument("--payload_factor", type=float, default=2.5)
    parser.add_argument("--payload_start", type=float, default=2.0)
    parser.add_argument("--payload_ramp", type=float, default=3.0)
    parser.add_argument("--wind_mode", type=str, default="none")
    parser.add_argument("--wind_mag", type=float, default=0.0)
    parser.add_argument("--send_delay", type=float, default=0.0,
                        help="행 간 전송 딜레이(초). 0=전속력")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    SEND_DELAY_SEC = args.send_delay

    producer = _make_producer()
    try:
        stream_to_kafka(
            producer,
            drone_id_tag=args.drone_id,
            p_gain_mult=args.p_gain_mult,
            duration_sec=args.duration,
            seed=args.seed,
            target_pos=(0.0, 0.0, 1.0),
            payload_factor=args.payload_factor,
            payload_start=args.payload_start,
            payload_ramp=args.payload_ramp,
            wind_mode=args.wind_mode,
            wind_mag=args.wind_mag,
        )
    finally:
        producer.close()
