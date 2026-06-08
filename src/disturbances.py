"""
disturbances.py — runtime disturbance injectors for the drone simulator.

Why this module exists
----------------------
The early-warning pipeline only makes sense if instability *grows* over a few
seconds (so there is a lead-time window to predict it). A disturbance that kills
the drone instantly leaves nothing to forecast. So disturbances here are designed
to be *tunable and gentle*: you dial them up until the drone goes from "holds fine"
to "oscillates, grows, then collapses".

Two disturbance families:

1. Wind  -> external force [N] applied every control step via
   ``p.applyExternalForce``. Three models:
     - ConstantWind : steady push (steady-state tilt / position bias)
     - GustWind     : steady base + periodic sinusoidal gusts (excites oscillation)
     - TurbulentWind: Ornstein-Uhlenbeck (mean-reverting random walk) force,
                      the most realistic "slowly varying random push" and the
                      best at exciting a *growing* oscillation.

2. Payload -> mass (and inertia) multiplier applied via ``p.changeDynamics``.
   IMPORTANT: DSLPIDControl computes its gravity feed-forward from the URDF
   *nominal* mass, so it does NOT know about the added payload. A heavier drone
   therefore sags, and the slow integral term has to wind up to compensate —
   exactly the kind of persistent, slowly-building stress that produces lead time.

All injectors are reproducible (seeded) and framework-light: they only depend on
numpy. PyBullet calls live in simulate.py; these classes just answer
"what force at time t?" and "what mass factor at time t?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Wind models
# --------------------------------------------------------------------------- #

class Wind:
    """Base class. ``force(t)`` returns a 3-vector force in Newtons (world frame)."""

    def force(self, t: float) -> np.ndarray:  # pragma: no cover - interface
        return np.zeros(3)


class NoWind(Wind):
    def force(self, t: float) -> np.ndarray:
        return np.zeros(3)


@dataclass
class ConstantWind(Wind):
    """Steady push that switches on at ``start``.

    magnitude : force magnitude [N]
    direction : 3-vector (need not be normalised); defaults to +x
    start     : seconds before which there is no wind
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
    """Steady base wind plus periodic sinusoidal gusts.

    A gust band near the drone's natural oscillation frequency is very effective
    at slowly pumping energy into roll/pitch — good for "grows then collapses".

    base_mag  : steady component [N]
    gust_mag  : peak amplitude of the oscillating component [N]
    gust_hz   : gust frequency [Hz]
    direction : push direction (unit-normalised internally)
    start     : onset time [s]
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
        env = self.base_mag + self.gust_mag * np.sin(2 * np.pi * self.gust_hz * (t - self.start))
        return env * self._unit


@dataclass
class TurbulentWind(Wind):
    """Ornstein-Uhlenbeck (mean-reverting) random force — realistic turbulence.

    dF = theta * (mean - F) dt + sigma * sqrt(dt) * N(0, I)

    mean   : 3-vector mean wind [N]
    sigma  : noise scale [N / sqrt(s)] — bigger = rougher air
    theta  : mean-reversion rate [1/s] — bigger = wind changes faster / less drift
    dt     : control timestep [s] (must match the sim)
    start  : onset time [s]
    seed   : RNG seed for reproducibility
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
        # Advance the OU process one step. We assume force() is called once per
        # control step in time order, which is how simulate.py uses it.
        dt = self.dt
        noise = self._rng.standard_normal(3)
        self._state = (
            self._state
            + self.theta * (self._mean - self._state) * dt
            + self.sigma * np.sqrt(dt) * noise
        )
        return self._state.copy()


def make_wind(mode: str, *, dt: float, mag: float, seed: int,
              direction=(1.0, 0.0, 0.0), start: float = 0.0,
              gust_hz: float = 1.0) -> Wind:
    """Factory used by simulate.py / run_batch.py from CLI / config strings.

    mode in {none, constant, gust, turbulent}. ``mag`` is the headline knob
    (force magnitude in N); other shape parameters derive from it so a single
    column in workloads.csv can sweep disturbance strength.
    """
    mode = (mode or "none").lower()
    if mode in ("none", "off", ""):
        return NoWind()
    if mode == "constant":
        return ConstantWind(magnitude=mag, direction=direction, start=start)
    if mode == "gust":
        # half steady, half oscillating
        return GustWind(base_mag=0.5 * mag, gust_mag=0.5 * mag, gust_hz=gust_hz,
                        direction=direction, start=start)
    if mode == "turbulent":
        # mean push along direction + OU noise of comparable scale
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        unit = d / n if n > 0 else np.array([1.0, 0.0, 0.0])
        return TurbulentWind(mean=tuple(0.5 * mag * unit), sigma=mag, theta=2.0,
                             dt=dt, start=start, seed=seed)
    raise ValueError(f"unknown wind mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Payload schedule
# --------------------------------------------------------------------------- #

@dataclass
class PayloadSchedule:
    """Mass/inertia multiplier as a function of time.

    factor    : final multiplier (1.0 = nominal mass, 1.5 = +50%)
    start     : when the payload begins to attach [s]
    ramp      : seconds to linearly ramp from 1.0 -> factor (0 = instant step)

    A gentle ramp produces a *slowly* growing altitude error instead of a sudden
    drop, which is friendlier to early warning.
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
