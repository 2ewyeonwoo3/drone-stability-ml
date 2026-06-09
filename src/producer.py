"""
producer.py — 드론 시뮬레이터를 Kafka producer로 연결.

핵심 설계 결정 (Stage 1 확정, 이 전환에서 절대 변경 없음):
  - event_time = EPOCH_BASE + step/ctrl_freq  (시뮬 시간 기반, datetime.now() 금지)
  - ctrl_freq를 메시지에 포함 (Spark consumer FFT UDF가 fs로 사용)
  - 파티션 키 = drone_id

연결 지점: config.py(환경변수)에서 읽음.
전환 방법: ENV_FILE=.env.s3 python src/producer.py

보안:
  현재 PLAINTEXT + 보안그룹(내 IP만 9092 허용) 방식.
  데모/증빙 목적에 충분. SASL_SSL은 현재 미검증(스트레치 옵션).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from config import cfg
from simulate import run_simulation

# 시뮬 시간을 실제 epoch에 고정하는 기준점.
# 고정값이어야 여러 런이 같은 타임라인에서 재현 가능.
EPOCH_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()

SEND_DELAY_SEC = 0.0


def _make_producer(retries: int = 5) -> KafkaProducer:
    """Kafka에 연결 가능할 때까지 재시도하며 producer 생성."""
    print(cfg.summary())

    # 현재 검증된 경로: PLAINTEXT (로컬 + EC2 자체호스팅)
    # SASL_SSL은 보안그룹 방식으로 대체하므로 지금은 단순 연결만
    kafka_kwargs = {
        "bootstrap_servers": cfg.kafka_bootstrap,
        "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
        "key_serializer":   lambda k: str(k).encode("utf-8"),
        "retries": 3,
        "acks": "all",
    }

    # SASL 분기: "있으면 읽는다" 수준 — 현재 실제 검증 안 함 (스트레치)
    if not cfg.is_plaintext:
        import ssl
        kafka_kwargs["security_protocol"] = cfg.kafka_security_protocol
        kafka_kwargs["ssl_context"] = ssl.create_default_context()

    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(**kafka_kwargs)
            print(f"[producer] Kafka 연결 성공 ({cfg.kafka_bootstrap})")
            return producer
        except NoBrokersAvailable:
            print(f"[producer] Kafka 미응답, {attempt}/{retries} 재시도 (5초 대기)...")
            time.sleep(5)
    raise RuntimeError(
        f"Kafka 브로커 연결 실패 ({cfg.kafka_bootstrap}). "
        "로컬: docker compose up 확인 / EC2: 보안그룹 9092 포트 + advertised.listeners 확인"
    )


def stream_to_kafka(producer: KafkaProducer, drone_id_tag: int = 0, **sim_kwargs) -> int:
    """시뮬레이터를 돌리고 텔레메트리 행을 Kafka에 전송. 반환값: 전송 행 수."""
    ctrl_freq = sim_kwargs.get("ctrl_freq", 240)
    print(f"[producer] 시뮬 시작: drone_id={drone_id_tag}, ctrl_freq={ctrl_freq}Hz")

    rows = run_simulation(**sim_kwargs, drone_id_tag=drone_id_tag, output_path=None)

    for row in rows:
        # event_time 합성: 시뮬 시간 기반 (datetime.now() 금지)
        sim_t: float = row["step"] / ctrl_freq
        event_time_iso = datetime.fromtimestamp(
            EPOCH_BASE + sim_t, tz=timezone.utc
        ).isoformat()

        msg = {**row, "event_time": event_time_iso}
        producer.send(cfg.kafka_topic, key=drone_id_tag, value=msg)

        if SEND_DELAY_SEC > 0:
            time.sleep(SEND_DELAY_SEC)

    producer.flush()
    print(f"[producer] 전송 완료: {len(rows)}행 → 토픽 '{cfg.kafka_topic}'")
    return len(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="드론 텔레메트리 Kafka producer")
    parser.add_argument("--p_gain_mult",    type=float, default=1.0)
    parser.add_argument("--duration",       type=float, default=9.0)
    parser.add_argument("--seed",           type=int,   default=7)
    parser.add_argument("--drone_id",       type=int,   default=0)
    parser.add_argument("--payload_factor", type=float, default=2.5)
    parser.add_argument("--payload_start",  type=float, default=2.0)
    parser.add_argument("--payload_ramp",   type=float, default=3.0)
    parser.add_argument("--wind_mode",      type=str,   default="none")
    parser.add_argument("--wind_mag",       type=float, default=0.0)
    parser.add_argument("--send_delay",     type=float, default=0.0)
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
