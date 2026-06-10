"""
disturbances.py — 드론 시뮬레이터 런타임 외란 주입기.

이 모듈이 존재하는 이유
----------------------
조기경보 파이프라인은 불안정이 수 초에 걸쳐 '서서히 자라야' 의미가 있다.
외란이 드론을 즉시 죽이면 예측할 선행 구간이 없어 조기경보가 성립하지 않는다.
따라서 이 모듈의 외란은 '조율 가능하고 점진적'으로 설계되었다.
외란을 서서히 키워가면서 드론이 "안정 유지 → 진동 성장 → 붕괴"하는 궤적을 만든다.

외란 두 종류:

1. 바람(Wind) → 매 제어 스텝마다 ``p.applyExternalForce``로 외력 [N] 주입.
   모델 세 가지:
     - ConstantWind    : 일정한 방향으로 밀어붙이는 정상 바람 (정적 기울기·위치 편차)
     - GustWind        : 정상 바람 + 주기적 사인 돌풍 (roll/pitch 진동 자극에 효과적)
     - TurbulentWind   : Ornstein-Uhlenbeck(평균 회귀) 랜덤 힘 — 가장 현실적인 난류.
                         '서서히 자라다 붕괴'하는 궤적을 만드는 데 가장 적합.
     - RampedTurbulentWind: TurbulentWind에 선형 ramp를 더한 버전.
                         Stage 2-A 조기경보 데이터 생성 전용.

2. 페이로드(Payload) → ``p.changeDynamics``로 질량·관성 배수 주입.
   중요: DSLPIDControl은 URDF 기본 질량만 알고 페이로드 추가를 모른다.
   따라서 드론이 페이로드만큼 더 무거워지면 서서히 가라앉고,
   적분 제어기가 뒤늦게 보상하는 과정에서 수 초의 lead time이 생긴다.
   이게 바로 조기경보가 '예측'할 수 있는 시간 구간이다.

모든 주입기는 재현 가능(seed 고정)하며 numpy만 의존한다.
PyBullet 호출은 simulate.py에 있고, 이 클래스들은
"t초에 얼마의 힘/질량배수를 줄 것인가"만 답한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 바람 모델
# --------------------------------------------------------------------------- #

class Wind:
    """바람 주입기 기반 클래스. ``force(t)``는 월드 좌표계 힘 벡터 [N, 3]을 반환한다."""

    def force(self, t: float) -> np.ndarray:  # pragma: no cover - interface
        return np.zeros(3)


class NoWind(Wind):
    def force(self, t: float) -> np.ndarray:
        return np.zeros(3)


@dataclass
class ConstantWind(Wind):
    """``start`` 이후 일정한 방향으로 밀어붙이는 정상 바람.

    magnitude : 힘 크기 [N]
    direction : 밀어붙일 방향 벡터 (정규화 불필요). 기본값 +x 방향
    start     : 바람이 시작되는 시각 [s]
    """
    magnitude: float = 0.0
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    start: float = 0.0

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = np.linalg.norm(d)
        self._unit = d / n if n > 0 else np.array([1.0, 0.0, 0.0])

    def force(self, t: float) -> np.ndarray:
        if t < self.start:
            return np.zeros(3)
        return self.magnitude * self._unit


@dataclass
class GustWind(Wind):
    """정상 바람 + 주기적 사인 돌풍.

    드론 고유 진동 주파수 근처의 돌풍은 roll/pitch에 에너지를 서서히 주입해
    "진동이 자라다 붕괴"하는 궤적을 만드는 데 효과적이다.

    base_mag  : 정상 성분 [N]
    gust_mag  : 사인 진동 성분의 최대 크기 [N]
    gust_hz   : 돌풍 주파수 [Hz]
    direction : 밀어붙일 방향 (내부에서 단위벡터로 정규화)
    start     : 시작 시각 [s]
    """
    base_mag: float = 0.0
    gust_mag: float = 0.0
    gust_hz: float = 1.0
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    start: float = 0.0

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = np.linalg.norm(d)
        self._unit = d / n if n > 0 else np.array([1.0, 0.0, 0.0])

    def force(self, t: float) -> np.ndarray:
        if t < self.start:
            return np.zeros(3)
        mag = self.base_mag + self.gust_mag * np.sin(2 * np.pi * self.gust_hz * (t - self.start))
        return mag * self._unit


@dataclass
class TurbulentWind(Wind):
    """Ornstein-Uhlenbeck(평균 회귀) 랜덤 힘 — 현실적인 난류 모사.

    dF = theta * (mean - F) dt + sigma * sqrt(dt) * N(0, I)

    mean   : 평균 바람 벡터 [N]
    sigma  : 노이즈 크기 [N / sqrt(s)] — 클수록 거친 공기
    theta  : 평균 회귀율 [1/s] — 클수록 바람이 빠르게 변하고 드리프트가 줄어듦
    dt     : 제어 주기 [s] (시뮬레이터와 반드시 일치)
    start  : 시작 시각 [s]
    seed   : 재현성 시드
    """
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    theta: float = 1.0
    dt: float = 1.0 / 240.0
    start: float = 0.0
    seed: int = 0

    _state: np.ndarray = field(init=False, default=None)
    _rng: np.random.Generator = field(init=False, default=None)
    _last_t: float = field(init=False, default=-np.inf)

    def __post_init__(self):
        self._mean = np.asarray(self.mean, dtype=float)
        self._state = self._mean.copy()
        self._rng = np.random.default_rng(self.seed)
        self._last_t = None

    def force(self, t: float) -> np.ndarray:
        if t < self.start:
            return np.zeros(3)
        # OU 프로세스를 한 스텝 진행.
        # simulate.py가 시간 순서로 매 제어 스텝마다 한 번씩 호출한다고 가정.
        dt = self.dt
        noise = self._rng.standard_normal(3)
        self._state = (
            self._state
            + self.theta * (self._mean - self._state) * dt
            + self.sigma * np.sqrt(dt) * noise
        )
        return self._state.copy()


# ── Stage 2-A 추가 ────────────────────────────────────────────────────────────

@dataclass
class RampedTurbulentWind(Wind):
    """시간에 따라 mag를 선형으로 키우는 OU 난류. Stage 2-A 조기경보 데이터 생성 전용.

    외란을 '즉사'가 아닌 '점진 발산'으로 만들기 위해 mag를 ramp로 증가시킨다.
    TurbulentWind와 달리 평균 방향 없는 전방향 난류(mean=0)를 사용한다.

    peak_mag : ramp 종료 시점의 최대 힘 크기 [N]
    start    : ramp 시작 시각 [s]
    ramp     : ramp 지속 시간 [s]. start+ramp 이후엔 peak_mag로 고정
    theta    : OU 평균 회귀율 [1/s]
    dt       : 제어 주기 [s] (시뮬레이터와 반드시 일치)
    seed     : 재현성 시드 — 같은 seed = 같은 노이즈 궤적
    """
    peak_mag: float = 0.0
    start: float = 0.0
    ramp: float = 5.0
    theta: float = 2.0
    dt: float = 1.0 / 240.0
    seed: int = 0

    _state: np.ndarray = field(init=False, default=None)
    _rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self):
        self._state = np.zeros(3)
        self._rng = np.random.default_rng(self.seed)

    def _current_mag(self, t: float) -> float:
        """현재 시각 t에서의 목표 mag 계산 (선형 ramp)."""
        if t < self.start:
            return 0.0
        elapsed = t - self.start
        if self.ramp <= 0:
            return self.peak_mag
        return min(self.peak_mag, self.peak_mag * elapsed / self.ramp)

    def force(self, t: float) -> np.ndarray:
        if t < self.start:
            return np.zeros(3)
        mag = self._current_mag(t)
        if mag <= 0:
            # ramp 시작 직후 상태 초기화 — 갑작스러운 튀임 방지
            self._state = np.zeros(3)
            return np.zeros(3)
        # OU 프로세스: 평균 방향 없음(전방향 난류), sigma = 현재 mag
        noise = self._rng.standard_normal(3)
        self._state = (
            self._state
            + self.theta * (np.zeros(3) - self._state) * self.dt
            + mag * np.sqrt(self.dt) * noise
        )
        return self._state.copy()

# ─────────────────────────────────────────────────────────────────────────────


def make_wind(mode: str, *, dt: float, mag: float, seed: int,
              direction=(1.0, 0.0, 0.0), start: float = 0.0,
              gust_hz: float = 1.0) -> Wind:
    """simulate.py / run_batch.py에서 모드 문자열로 바람 객체를 생성하는 팩토리.

    mode: none | constant | gust | turbulent | turbulent_ramp
    mag: 힘 크기의 대표 노브 [N]. 나머지 세부 파라미터는 mag에서 파생됨.

    turbulent_ramp: ramp=5s 고정 fallback.
    정확한 ramp 값이 필요하면 호출부에서 RampedTurbulentWind를 직접 생성할 것
    (simulate.py의 wind_mode=="turbulent_ramp" 분기 참고).
    """
    mode = (mode or "none").lower()
    if mode in ("none", "off", ""):
        return NoWind()
    if mode == "constant":
        return ConstantWind(magnitude=mag, direction=direction, start=start)
    if mode == "gust":
        # 절반은 정상 성분, 절반은 진동 성분
        return GustWind(base_mag=0.5 * mag, gust_mag=0.5 * mag, gust_hz=gust_hz,
                        direction=direction, start=start)
    if mode == "turbulent":
        # 방향으로 mean push + 동일 크기의 OU 노이즈
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        unit = d / n if n > 0 else np.array([1.0, 0.0, 0.0])
        return TurbulentWind(mean=tuple(0.5 * mag * unit), sigma=mag, theta=2.0,
                             dt=dt, start=start, seed=seed)
    if mode == "turbulent_ramp":
        # Stage 2-A 전용. 정확한 ramp 전달이 필요하면 RampedTurbulentWind 직접 생성.
        return RampedTurbulentWind(peak_mag=mag, start=start, ramp=5.0,
                                   theta=2.0, dt=dt, seed=seed)
    raise ValueError(f"알 수 없는 바람 모드: {mode!r}")


# --------------------------------------------------------------------------- #
# 페이로드 스케줄
# --------------------------------------------------------------------------- #

@dataclass
class PayloadSchedule:
    """시간에 따른 질량·관성 배수.

    factor : 최종 배수 (1.0 = 기본 질량, 1.5 = +50%)
    start  : 페이로드가 부착되기 시작하는 시각 [s]
    ramp   : 1.0 → factor까지 선형 증가에 걸리는 시간 [s] (0이면 즉시 적용)

    완만한 ramp는 갑작스러운 추락 대신 서서히 자라는 고도 오차를 만들어
    조기경보 lead time 확보에 유리하다.
    """
    factor: float = 1.0
    start: float = 0.0
    ramp: float = 0.0

    def mass_factor(self, t: float) -> float:
        if self.factor == 1.0 or t < self.start:
            return 1.0
        if self.ramp <= 0:
            return self.factor
        frac = min(1.0, (t - self.start) / self.ramp)
        return 1.0 + (self.factor - 1.0) * frac

    @property
    def is_active(self) -> bool:
        return self.factor != 1.0