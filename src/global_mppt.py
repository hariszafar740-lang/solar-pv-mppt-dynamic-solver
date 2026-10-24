"""Global Maximum Power Point Tracking (GMPPT) engines for partially shaded PV strings.

Under non-uniform irradiance, anti-parallel bypass diodes partition a series PV
string's P-V characteristic into multiple local maxima (LMPPs) alongside a
single Global Maximum Power Point (GMPP).  Conventional hill-climbing trackers
(`PerturbAndObserve`, `IncrementalConductance`) rely strictly on local slope
sign and become permanently trapped in whichever local peak lies inside their
initial voltage basin (commonly the rightmost high-voltage peak near ``0.85 V_oc``).

Furthermore, when a string is locked on a low-voltage peak (where shaded
substrings are bypassed) and clouds subsequently clear on the shaded substrings
while the unshaded substring's irradiance stays constant, operating current at
the low-voltage plateau changes by only 1-2% ("bypassed-plateau cloud-clearance
blind spot").  Both GMPPT controllers below address both failure modes:

1. `ParticleSwarmMPPT`:
   Seeded Particle Swarm Optimization (PSO) voltage tracker with linearly
   annealed inertia, velocity clamping, convergence lock with zero steady-state
   perturbation ripple, ``|dP|/P`` event detection, and single-cycle high-voltage
   watchdog probing when locked below ``0.72 * v_max``.

2. `HybridGlobalScanINC`:
   Two-stage industrial GMPPT combining a fast coarse global voltage sweep across
   ``[v_min, v_max]`` followed by Variable-Step Adaptive Incremental Conductance
   (`dI/dV = -I/V`) and low-voltage blind-spot watchdog verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.partial_shading import ShadedPVString, ShadingResult
from src.pv_model import _require_finite_real

__all__ = [
    "ControllerTrajectory",
    "GlobalMPPTBenchmark",
    "HybridGlobalScanINC",
    "ParticleSwarmMPPT",
    "run_dynamic_gmppt_benchmark",
]


def _validate_voltage_bounds(v_min: float, v_max: float) -> tuple[float, float]:
    vmin = _require_finite_real(v_min, "v_min")
    vmax = _require_finite_real(v_max, "v_max")
    if vmin < 0.0:
        raise ValueError(f"v_min must be >= 0.0 V, got {vmin}")
    if vmax <= vmin:
        raise ValueError(f"v_max must exceed v_min, got v_min={vmin}, v_max={vmax}")
    return vmin, vmax


class ParticleSwarmMPPT:
    """Particle Swarm Optimization (PSO) Global MPPT controller with blind-spot watchdog."""

    def __init__(
        self,
        v_min: float = 1.0,
        v_max: float = 40.45,
        *,
        n_particles: int = 8,
        w_max: float = 0.68,
        w_min: float = 0.16,
        c1: float = 1.35,
        c2: float = 1.65,
        max_iters: int = 6,
        convergence_tol_v: float = 0.12,
        reinit_power_delta_rel: float = 0.015,
        watchdog_interval: int = 65,
        seed: int | None = 42,
    ) -> None:
        self.v_min, self.v_max = _validate_voltage_bounds(v_min, v_max)
        if isinstance(n_particles, (bool, np.bool_)) or not isinstance(
            n_particles, (int, np.integer)
        ) or n_particles < 3:
            raise ValueError(f"n_particles must be an integer >= 3, got {n_particles!r}")
        if isinstance(max_iters, (bool, np.bool_)) or not isinstance(
            max_iters, (int, np.integer)
        ) or max_iters < 2:
            raise ValueError(f"max_iters must be an integer >= 2, got {max_iters!r}")
        if isinstance(watchdog_interval, (bool, np.bool_)) or not isinstance(
            watchdog_interval, (int, np.integer)
        ) or watchdog_interval < 5:
            raise ValueError(
                f"watchdog_interval must be an integer >= 5, got {watchdog_interval!r}"
            )

        self.n_particles = int(n_particles)
        self.w_max = _require_finite_real(w_max, "w_max")
        self.w_min = _require_finite_real(w_min, "w_min")
        self.c1 = _require_finite_real(c1, "c1")
        self.c2 = _require_finite_real(c2, "c2")
        self.max_iters = int(max_iters)
        self.convergence_tol_v = _require_finite_real(
            convergence_tol_v, "convergence_tol_v"
        )
        self.reinit_power_delta_rel = _require_finite_real(
            reinit_power_delta_rel, "reinit_power_delta_rel"
        )
        self.watchdog_interval = int(watchdog_interval)

        if not (0.0 <= self.w_min <= self.w_max <= 1.2):
            raise ValueError("Inertia weights must satisfy 0 <= w_min <= w_max <= 1.2")
        if self.c1 < 0.0 or self.c2 < 0.0:
            raise ValueError("Acceleration coefficients c1 and c2 must be >= 0")
        if self.convergence_tol_v <= 0.0:
            raise ValueError("convergence_tol_v must be positive")
        if not (0.005 <= self.reinit_power_delta_rel <= 0.9):
            raise ValueError("reinit_power_delta_rel must lie in [0.005, 0.9]")

        self._rng = np.random.default_rng(seed)
        self._vel_limit = 0.28 * (self.v_max - self.v_min)
        self.reset()

    def reset(self) -> None:
        """Disperse particles across stratified voltage zones in ``[v_min, v_max]``."""
        span = self.v_max - self.v_min
        self.positions = np.linspace(
            self.v_min + 0.08 * span,
            self.v_max - 0.06 * span,
            self.n_particles,
            dtype=np.float64,
        )
        self.velocities = np.zeros(self.n_particles, dtype=np.float64)
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.full(self.n_particles, -np.inf, dtype=np.float64)
        self.gbest_pos = float(self.positions[self.n_particles // 2])
        self.gbest_val = -np.inf
        self.locked = False
        self._locked_power = 0.0
        self._locked_steps = 0
        self._probing_high_v = False
        self._particle_idx = 0
        self._iter_idx = 0
        self._started = False

    @property
    def initial_voltage(self) -> float:
        return float(self.positions[0])

    def step(self, v_curr: float, i_curr: float) -> float:
        """Advance one control cycle from measured ``(v_curr, i_curr)`` and return ``v_next``."""
        v_val = _require_finite_real(v_curr, "v_curr")
        i_val = _require_finite_real(i_curr, "i_curr")
        return self.step_power(v_val, max(0.0, v_val * i_val))

    def step_power(self, v_curr: float, p_curr: float) -> float:
        """Advance one control cycle from measured ``(v_curr, p_curr)`` and return ``v_next``."""
        v_val = _require_finite_real(v_curr, "v_curr")
        p_val = _require_finite_real(p_curr, "p_curr")

        if not self._started:
            self._started = True
            self.gbest_pos = float(np.clip(v_val, self.v_min, self.v_max))
            self.gbest_val = p_val
            self._particle_idx = 0
            return float(self.positions[0])

        if self.locked:
            if self._probing_high_v:
                self._probing_high_v = False
                if p_val > 1.02 * self._locked_power:
                    self.reset()
                    self._started = True
                    self.gbest_pos = float(np.clip(v_val, self.v_min, self.v_max))
                    self.gbest_val = p_val
                    return float(self.positions[0])
                return self.gbest_pos

            denom = max(abs(self._locked_power), 1.0)
            if abs(p_val - self._locked_power) / denom > self.reinit_power_delta_rel:
                self.reset()
                self._started = True
                self.gbest_pos = float(np.clip(v_val, self.v_min, self.v_max))
                self.gbest_val = p_val
                return float(self.positions[0])

            self._locked_steps += 1
            if (
                self.gbest_pos < 0.72 * self.v_max
                and self._locked_steps % self.watchdog_interval == 0
            ):
                self._probing_high_v = True
                return float(self.v_min + 0.80 * (self.v_max - self.v_min))

            return self.gbest_pos

        idx = self._particle_idx
        clamped_v = float(np.clip(v_val, self.v_min, self.v_max))
        if p_val > self.pbest_val[idx]:
            self.pbest_val[idx] = p_val
            self.pbest_pos[idx] = clamped_v
        if p_val > self.gbest_val:
            self.gbest_val = p_val
            self.gbest_pos = clamped_v

        self._particle_idx += 1
        if self._particle_idx < self.n_particles:
            return float(self.positions[self._particle_idx])

        self._particle_idx = 0
        self._iter_idx += 1

        spread = float(np.max(self.positions) - np.min(self.positions))
        if spread <= self.convergence_tol_v or self._iter_idx >= self.max_iters:
            self.locked = True
            self._locked_power = max(0.0, float(self.gbest_val))
            self._locked_steps = 0
            return self.gbest_pos

        progress = min(1.0, self._iter_idx / max(1, self.max_iters - 1))
        w = self.w_max - progress * (self.w_max - self.w_min)
        r1 = self._rng.random(self.n_particles)
        r2 = self._rng.random(self.n_particles)

        cognitive = self.c1 * r1 * (self.pbest_pos - self.positions)
        social = self.c2 * r2 * (self.gbest_pos - self.positions)
        vel = np.clip(
            w * self.velocities + cognitive + social,
            -self._vel_limit,
            self._vel_limit,
        )
        self.velocities = vel
        self.positions = np.clip(self.positions + vel, self.v_min, self.v_max)
        return float(self.positions[0])


class HybridGlobalScanINC:
    """Two-stage Global Voltage Sweep + Variable-Step Adaptive Incremental Conductance."""

    def __init__(
        self,
        v_min: float = 1.0,
        v_max: float = 40.45,
        *,
        n_scan_points: int = 12,
        min_step_v: float = 0.05,
        max_step_v: float = 0.85,
        adaptive_gain: float = 0.045,
        conductance_tol: float = 8e-3,
        rescan_power_delta_rel: float = 0.015,
        watchdog_interval: int = 65,
    ) -> None:
        self.v_min, self.v_max = _validate_voltage_bounds(v_min, v_max)
        if isinstance(n_scan_points, (bool, np.bool_)) or not isinstance(
            n_scan_points, (int, np.integer)
        ) or n_scan_points < 4:
            raise ValueError(
                f"n_scan_points must be an integer >= 4, got {n_scan_points!r}"
            )
        if isinstance(watchdog_interval, (bool, np.bool_)) or not isinstance(
            watchdog_interval, (int, np.integer)
        ) or watchdog_interval < 5:
            raise ValueError(
                f"watchdog_interval must be an integer >= 5, got {watchdog_interval!r}"
            )
        self.n_scan_points = int(n_scan_points)
        self.min_step_v = _require_finite_real(min_step_v, "min_step_v")
        self.max_step_v = _require_finite_real(max_step_v, "max_step_v")
        self.adaptive_gain = _require_finite_real(adaptive_gain, "adaptive_gain")
        self.conductance_tol = _require_finite_real(conductance_tol, "conductance_tol")
        self.rescan_power_delta_rel = _require_finite_real(
            rescan_power_delta_rel, "rescan_power_delta_rel"
        )
        self.watchdog_interval = int(watchdog_interval)

        if self.min_step_v <= 0.0 or self.max_step_v < self.min_step_v:
            raise ValueError("Step bounds must satisfy 0 < min_step_v <= max_step_v")
        if self.adaptive_gain <= 0.0 or self.conductance_tol <= 0.0:
            raise ValueError("adaptive_gain and conductance_tol must be positive")
        if not (0.005 <= self.rescan_power_delta_rel <= 0.9):
            raise ValueError("rescan_power_delta_rel must lie in [0.005, 0.9]")

        self.reset()

    def reset(self) -> None:
        """Start a fresh coarse global voltage scan across ``[v_min, v_max]``."""
        span = self.v_max - self.v_min
        self._scan_grid = np.linspace(
            self.v_min + 0.04 * span,
            self.v_max - 0.03 * span,
            self.n_scan_points,
            dtype=np.float64,
        )
        self._scan_idx = 0
        self._best_scan_v = float(self._scan_grid[0])
        self._best_scan_p = -np.inf
        self.mode = "scan"
        self._started = False
        self._v_prev = float(self._scan_grid[0])
        self._i_prev = 0.0
        self._p_ref = 0.0
        self._settled_steps = 0
        self._probing_high_v = False
        self._recovering_from_probe = False
        self._pre_probe_v = 0.0
        self._pre_probe_i = 0.0

    @property
    def initial_voltage(self) -> float:
        return float(self._scan_grid[0])

    def step(self, v_curr: float, i_curr: float) -> float:
        """Advance one control cycle from measured ``(v_curr, i_curr)`` and return ``v_next``."""
        v_val = _require_finite_real(v_curr, "v_curr")
        i_val = _require_finite_real(i_curr, "i_curr")
        p_val = max(0.0, v_val * i_val)

        if not self._started:
            self._started = True
            if p_val > self._best_scan_p:
                self._best_scan_p = p_val
                self._best_scan_v = float(np.clip(v_val, self.v_min, self.v_max))
            self._scan_idx = 0
            return float(self._scan_grid[0])

        if self.mode == "scan":
            clamped_v = float(np.clip(v_val, self.v_min, self.v_max))
            if p_val > self._best_scan_p:
                self._best_scan_p = p_val
                self._best_scan_v = clamped_v

            self._scan_idx += 1
            if self._scan_idx < self.n_scan_points:
                return float(self._scan_grid[self._scan_idx])

            self.mode = "track"
            self._v_prev = v_val
            self._i_prev = i_val
            self._p_ref = max(0.0, self._best_scan_p)
            self._settled_steps = 0
            return self._best_scan_v

        if self._probing_high_v:
            self._probing_high_v = False
            if p_val > 1.02 * self._p_ref:
                self.reset()
                self._started = True
                self._best_scan_p = p_val
                self._best_scan_v = float(np.clip(v_val, self.v_min, self.v_max))
                return float(self._scan_grid[0])
            self._recovering_from_probe = True
            return self._pre_probe_v

        if self._recovering_from_probe:
            self._recovering_from_probe = False
            self._v_prev = v_val
            self._i_prev = i_val
            return v_val

        if self._settled_steps >= 4:
            denom = max(abs(self._p_ref), 1.0)
            if abs(p_val - self._p_ref) / denom > self.rescan_power_delta_rel:
                self.reset()
                self._started = True
                self._best_scan_p = p_val
                self._best_scan_v = float(np.clip(v_val, self.v_min, self.v_max))
                return float(self._scan_grid[0])

            if (
                v_val < 0.72 * self.v_max
                and self._settled_steps % self.watchdog_interval == 0
            ):
                self._probing_high_v = True
                self._pre_probe_v = v_val
                self._pre_probe_i = i_val
                return float(self.v_min + 0.80 * (self.v_max - self.v_min))

        dv = v_val - self._v_prev
        di = i_val - self._i_prev

        if abs(dv) < 1e-6:
            if abs(di) < 1e-5:
                v_next = v_val
            else:
                direction = 1.0 if di > 0.0 else -1.0
                v_next = v_val + direction * self.min_step_v
        else:
            inc_cond = di / dv
            inst_cond = -i_val / v_val if v_val > 1e-6 else 0.0
            err = inc_cond - inst_cond
            if abs(err) <= self.conductance_tol:
                v_next = v_val
            else:
                dp_dv = i_val + v_val * inc_cond
                step_mag = float(
                    np.clip(
                        self.adaptive_gain * abs(dp_dv),
                        self.min_step_v,
                        self.max_step_v,
                    )
                )
                v_next = v_val + (step_mag if err > 0.0 else -step_mag)

        v_next = float(np.clip(v_next, self.v_min, self.v_max))
        self._v_prev = v_val
        self._i_prev = i_val
        self._settled_steps += 1
        if self._settled_steps <= 4 and p_val > self._p_ref:
            self._p_ref = p_val
        return v_next


@dataclass
class ControllerTrajectory:
    """Time-series telemetry and energy-harvesting efficiency for one MPPT controller."""

    name: str
    voltages: np.ndarray
    currents: np.ndarray
    powers: np.ndarray
    tracking_efficiency_pct: float
    settled_efficiency_pct: float


@dataclass
class GlobalMPPTBenchmark:
    """Comparative dynamic GMPPT benchmark across multi-stage partial shading."""

    time_steps: np.ndarray
    gmpp_powers: np.ndarray
    gmpp_voltages: np.ndarray
    stage_boundaries: tuple[int, ...]
    stage_labels: tuple[str, ...]
    trajectories: dict[str, ControllerTrajectory]


def run_dynamic_gmppt_benchmark(
    string: ShadedPVString | None = None,
    *,
    stages: Sequence[tuple[str, Sequence[float], int]] | None = None,
    v_start_conventional: float | None = None,
) -> GlobalMPPTBenchmark:
    """Compare Conventional P&O, INC, PSO-GMPPT, and Hybrid Scan + Adaptive INC."""
    pv_string = string if string is not None else ShadedPVString()
    if stages is None:
        stages = (
            ("Phase 1: Chimney Shade (1000/750/220 W/m²)", (1000.0, 750.0, 220.0), 80),
            ("Phase 2: Heavy 2/3 Shade (1000/220/220 W/m²)", (1000.0, 220.0, 220.0), 80),
            ("Phase 3: Clearing Sky (1000/950/900 W/m²)", (1000.0, 950.0, 900.0), 80),
        )

    stage_curves: list[tuple[str, ShadingResult, int]] = []
    v_oc_max = 0.0
    for label, irrad, count in stages:
        if count < 20:
            raise ValueError(f"Each stage must run >= 20 steps, got {count}")
        res = pv_string.solve_curve(irrad, temperatures_c=25.0, num_points=700)
        stage_curves.append((label, res, int(count)))
        v_oc_max = max(v_oc_max, res.v_oc_string)

    v_min = 1.0
    v_max = max(v_min + 1.0, v_oc_max)
    v_conv0 = (
        0.85 * stage_curves[0][1].v_oc_string
        if v_start_conventional is None
        else float(v_start_conventional)
    )

    po = PerturbAndObserve(step_size=0.20, v_init=v_conv0, v_min=v_min, v_max=v_max)
    inc = IncrementalConductance(
        step_size=0.20, v_init=v_conv0, v_min=v_min, v_max=v_max, conductance_tol=1e-2
    )
    pso = ParticleSwarmMPPT(v_min=v_min, v_max=v_max, seed=42)
    hybrid = HybridGlobalScanINC(v_min=v_min, v_max=v_max)

    total_steps = sum(cnt for _, _, cnt in stage_curves)
    gmpp_p = np.zeros(total_steps, dtype=np.float64)
    gmpp_v = np.zeros(total_steps, dtype=np.float64)

    names = (
        "Conventional P&O",
        "Conventional INC",
        "PSO Global MPPT",
        "Hybrid Scan + Adaptive INC",
    )
    v_hist = {name: np.zeros(total_steps, dtype=np.float64) for name in names}
    i_hist = {name: np.zeros(total_steps, dtype=np.float64) for name in names}
    p_hist = {name: np.zeros(total_steps, dtype=np.float64) for name in names}

    v_state = {
        "Conventional P&O": v_conv0,
        "Conventional INC": v_conv0,
        "PSO Global MPPT": v_conv0,
        "Hybrid Scan + Adaptive INC": v_conv0,
    }

    boundaries: list[int] = []
    labels: list[str] = []
    settled_mask = np.zeros(total_steps, dtype=bool)

    k = 0
    for label, curve, count in stage_curves:
        boundaries.append(k)
        labels.append(label)
        settled_start = k + max(15, int(0.60 * count))
        settled_mask[settled_start : k + count] = True

        for _ in range(count):
            gmpp_p[k] = curve.gmpp.power_w
            gmpp_v[k] = curve.gmpp.voltage_v

            for name in names:
                v_now = v_state[name]
                i_now, p_now = curve.sample_at_voltage(v_now)
                v_hist[name][k] = v_now
                i_hist[name][k] = i_now
                p_hist[name][k] = p_now

                if name == "Conventional P&O":
                    v_state[name] = po.step(v_now, p_now)
                elif name == "Conventional INC":
                    v_state[name] = inc.step(v_now, i_now)
                elif name == "PSO Global MPPT":
                    v_state[name] = pso.step(v_now, i_now)
                else:
                    v_state[name] = hybrid.step(v_now, i_now)
            k += 1

    total_ideal_energy = float(np.sum(gmpp_p))
    settled_ideal_energy = float(np.sum(gmpp_p[settled_mask]))

    trajectories: dict[str, ControllerTrajectory] = {}
    for name in names:
        eff_total = (
            100.0 * float(np.sum(p_hist[name])) / total_ideal_energy
            if total_ideal_energy > 0.0
            else 0.0
        )
        eff_settled = (
            100.0 * float(np.sum(p_hist[name][settled_mask])) / settled_ideal_energy
            if settled_ideal_energy > 0.0
            else 0.0
        )
        trajectories[name] = ControllerTrajectory(
            name=name,
            voltages=v_hist[name],
            currents=i_hist[name],
            powers=p_hist[name],
            tracking_efficiency_pct=eff_total,
            settled_efficiency_pct=eff_settled,
        )

    return GlobalMPPTBenchmark(
        time_steps=np.arange(total_steps),
        gmpp_powers=gmpp_p,
        gmpp_voltages=gmpp_v,
        stage_boundaries=tuple(boundaries),
        stage_labels=tuple(labels),
        trajectories=trajectories,
    )
