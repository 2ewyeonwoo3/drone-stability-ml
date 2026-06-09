"""
config.py — 파이프라인 전체 연결 지점을 환경변수로 관리.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(env_file: str | None = None) -> None:
    """
    .env 파일을 읽어 환경변수로 로드.
    python-dotenv가 없으면 조용히 건너뜀(필수 의존성 아님).
    우선순위: 실제 환경변수 > .env 파일 (override=False).
    """
    try:
        from dotenv import load_dotenv
        target = env_file or os.getenv("ENV_FILE", ".env.local")
        if Path(target).exists():
            load_dotenv(target, override=False)
            print(f"[config] 설정 파일 로드: {target}")
        else:
            print(f"[config] 설정 파일 없음 ({target}), 환경변수 직접 사용")
    except ImportError:
        pass


@dataclass
class PipelineConfig:
    """파이프라인 연결 지점 설정. 기본값 = 로컬."""

    # ── Kafka ─────────────────────────────────────────────────────────────
    # .env.local : localhost:9092
    # .env.s3    : localhost:9092  (Kafka는 아직 로컬)
    # .env.aws   : <EC2_PUBLIC_IP>:9092
    kafka_bootstrap: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    )

    # 보안 프로토콜
    # EC2 자체호스팅 + 보안그룹(내 IP만 9092 허용) → PLAINTEXT로 충분
    # SASL_SSL은 인증서·jaas 설정 복잡도가 데모 목적 대비 과함
    # → 모든 환경에서 PLAINTEXT 단일 경로만 실제 검증
    # SASL 분기 코드는 "있으면 읽는다" 수준으로만 유지 (추후 스트레치용)
    kafka_security_protocol: str = field(
        default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    )

    kafka_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_TOPIC", "drone-telemetry")
    )

    # ── 스토리지 ──────────────────────────────────────────────────────────
    # .env.local : data/streaming/features     (로컬 상대경로)
    # .env.s3    : s3a://<버킷>/drone-pipeline/features
    # .env.aws   : s3a://<버킷>/drone-pipeline/features
    sink_path: str = field(
        default_factory=lambda: os.getenv("SINK_PATH", "data/streaming/features")
    )
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_PATH", "data/streaming/checkpoint")
    )

    # ── AWS 자격증명 ──────────────────────────────────────────────────────
    # 로컬에서 s3a 테스트: AWS_ACCESS_KEY_ID/SECRET 환경변수로 주입
    #   → SimpleAWSCredentialsProvider 사용
    # EC2에서 실행: 키 비워두면 IAM role(instance profile) 자동 적용
    #   → DefaultAWSCredentialsProviderChain 사용
    # 두 경우를 aws_access_key_id 유무로 자동 판별
    aws_access_key_id: str = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    )

    # ── 편의 프로퍼티 ─────────────────────────────────────────────────────

    @property
    def is_s3(self) -> bool:
        """싱크가 S3 경로(s3a://)인지 여부."""
        return self.sink_path.startswith("s3")

    @property
    def is_plaintext(self) -> bool:
        """Kafka 보안 없음(PLAINTEXT) 여부. 현재 모든 환경에서 True."""
        return self.kafka_security_protocol == "PLAINTEXT"

    @property
    def s3_credentials_provider(self) -> str:
        """
        S3 자격증명 공급자 결정.
        - 로컬 키 방식: SimpleAWSCredentialsProvider
        - EC2 IAM role: DefaultAWSCredentialsProviderChain (키 없으면 자동 선택)
        """
        if self.aws_access_key_id:
            return "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        return "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"

    def summary(self) -> str:
        lines = [
            "=== 파이프라인 설정 ===",
            f"  Kafka 브로커  : {self.kafka_bootstrap}",
            f"  보안 프로토콜 : {self.kafka_security_protocol}",
            f"  싱크 경로     : {self.sink_path}",
            f"  S3 모드       : {self.is_s3}",
            f"  S3 인증 방식  : {'키(로컬)' if self.aws_access_key_id else 'IAM role(EC2)'}",
            "=" * 22,
        ]
        return "\n".join(lines)


_load_dotenv()
cfg = PipelineConfig()
