"""CENELEC EN 50530 dynamic MPPT efficiency standard & coupled electro-thermal PV transient solver.

Physical & Standards Formulation
--------------------------------
1. EN 50530 Trapezoidal Irradiance Ramps:
   Standardized dynamic MPPT testing subjects PV trackers to trapezoidal
   irradiance profiles oscillating between ``G_low`` and ``G_high`` (the
   100-500 W/m^2 low-to-medium band and 300-1000 W/m^2 medium-to-high band) at
   prescribed slew rates ``s`` [W/m^2/s] separated by dwell intervals.  Dynamic
   MPPT efficiency is integrated over the test horizon ``[0, T]``:

       eta_MPPT,dyn = 100 * (integral_0^T P_dc(t) dt) / (integral_0^T P_mpp(G(t), T_c(t)) dt)

2. Coupled Lumped Electro-Thermal Cell Transient ODE:
   Rather than holding cell temperature constant during irradiance ramps, the
   module laminate's areal heat capacity ``C_th`` [J/(m^2*K)] and overall heat
   loss coefficient ``U_th`` [W/(m^2*K)] govern cell temperature ``T_c(t)``:

       C_th * dT_c/dt = alpha_abs * G(t) - P_elec(t) / A_mod - U_th * (T_c(t) - T_amb)

   Extracting electrical power ``P_elec(t)`` cools the module relative to
   open-circuit operation (real electro-thermal coupling), calibrated to
   nominal NOCT ~ 45 degC at 800 W/m^2, 20 degC.

3. Sera `dP-P&O` Drift-Free Ramp Tracker (`DeltaPPO`):
   Conventional P&O and INC suffer severe upward/downward voltage drift during
   fast irradiance ramps because ``(dP/dG)*dG`` masks ``(dP/dV)*dV``.  `DeltaPPO`
   (Sera et al., IEEE TIE) interleaves a constant-voltage hold sample to measure
   pure irradiance-induced power change ``dP_irr`` and subtracts it from the
   subsequent perturbation sample ``dP_v = dP_raw - dP_irr``, cancelling linear
   irradiance ramp drift without requiring PV model parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.pv_model import (
    MAX_OPERATING_TEMP_C,
    MIN_OPERATING_TEMP_C,
    SolarPVModel,
    _require_finite_real,
)

__all__ = [
    "DeltaPPO",
    "EN50530BenchmarkResult",
    "EN50530ControllerSummary",
    "EN50530RampSegment",
    "LumpedThermalModel",
    "build_en50530_composite_profile",
    "generate_trapezoidal_irradiance_profile",
    "run_en50530_dynamic_benchmark",
]

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


@dataclass(frozen=True)
class EN50530RampSegment:
    """Specification for one EN 50530 trapezoidal irradiance test sequence."""

    label: str
    g_low_wm2: float
    g_high_wm2: float
    slew_rate_wm2_s: float
    dwell_s: float = 8.0
    repetitions: int = 1

    def __post_init__(self) -> None:
        glow = _require_finite_real(self.g_low_wm2, "g_low_wm2")
        ghigh = _require_finite_real(self.g_high_wm2, "g_high_wm2")
        slew = _require_finite_real(self.slew_rate_wm2_s, "slew_rate_wm2_s")
        dwell = _require_finite_real(self.dwell_s, "dwell_s")
        if glow <= 0.0 or ghigh <= glow:
            raise ValueError(
                f"Require 0 < g_low_wm2 < g_high_wm2, got {glow} and {ghigh}"
            )
        if slew <= 0.0:
            raise ValueError(f"slew_rate_wm2_s must be > 0, got {slew}")
        if dwell <= 0.0:
            raise ValueError(f"dwell_s must be > 0, got {dwell}")
        if isinstance(self.repetitions, (bool, np.bool_)) or not isinstance(
            self.repetitions, (int, np.integer)
        ) or self.repetitions < 1:
            raise ValueError(
                f"repetitions must be an integer >= 1, got {self.repetitions!r}"
            )


class LumpedThermalModel:
    """First-order lumped electro-thermal PV cell temperature model."""

    def __init__(
        self,
        *,
        c_th_j_m2k: float = 1800.0,
        u_th_w_m2k: float = 29.0,
        absorptance: float = 0.90,
        module_area_m2: float = 1.64,
        t_ambient_c: float = 25.0,
    ) -> None:
        self.c_th = _require_finite_real(c_th_j_m2k, "c_th_j_m2k")
        self.u_th = _require_finite_real(u_th_w_m2k, "u_th_w_m2k")
        self.absorptance = _require_finite_real(absorptance, "absorptance")
        self.module_area_m2 = _require_finite_real(module_area_m2, "module_area_m2")
        self.t_ambient_c = _require_finite_real(t_ambient_c, "t_ambient_c")

        if self.c_th <= 0.0 or self.u_th <= 0.0 or self.module_area_m2 <= 0.0:
            raise ValueError("c_th_j_m2k, u_th_w_m2k, and module_area_m2 must be > 0")
        if not (0.1 <= self.absorptance <= 1.0):
            raise ValueError(
                f"absorptance must lie in [0.1, 1.0], got {self.absorptance}"
            )
        if not (MIN_OPERATING_TEMP_C <= self.t_ambient_c <= MAX_OPERATING_TEMP_C):
            raise ValueError(f"t_ambient_c out of range: {self.t_ambient_c}")

    @property
    def tau_seconds(self) -> float:
        """Thermal time constant ``C_th / U_th`` in seconds."""
        return self.c_th / self.u_th

    def steady_state_temp_c(
        self, irradiance_wm2: float, p_elec_w: float = 0.0, t_amb_c: float | None = None
    ) -> float:
        """Return equilibrium cell temperature (degC) at constant ``(G, P_elec)``."""
        g_val = _require_finite_real(irradiance_wm2, "irradiance_wm2")
        p_val = _require_finite_real(p_elec_w, "p_elec_w")
        if g_val < 0.0 or p_val < 0.0:
            raise ValueError("irradiance_wm2 and p_elec_w must be >= 0")
        tamb = (
            self.t_ambient_c
            if t_amb_c is None
            else _require_finite_real(t_amb_c, "t_amb_c")
        )
        q_net = max(0.0, self.absorptance * g_val - p_val / self.module_area_m2)
        t_eq = tamb + q_net / self.u_th
        return float(np.clip(t_eq, MIN_OPERATING_TEMP_C, MAX_OPERATING_TEMP_C))

    def step_temperature(
        self,
        t_cell_c: float,
        irradiance_wm2: float,
        p_elec_w: float,
        dt_s: float,
        *,
        t_amb_c: float | None = None,
    ) -> float:
        """Advance cell temperature over ``dt_s`` via exact exponential integration."""
        tc = _require_finite_real(t_cell_c, "t_cell_c")
        dt = _require_finite_real(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt}")
        t_inf = self.steady_state_temp_c(irradiance_wm2, p_elec_w, t_amb_c=t_amb_c)
        decay = float(np.exp(-dt / self.tau_seconds))
        t_next = t_inf + (tc - t_inf) * decay
        return float(np.clip(t_next, MIN_OPERATING_TEMP_C, MAX_OPERATING_TEMP_C))


class DeltaPPO:
    """Sera `dP-P&O` irradiance-compensated drift-free MPPT for EN 50530 ramps."""

    def __init__(
        self,
        step_size: float = 0.25,
        v_init: float = 25.0,
        v_min: float = 1.0,
        v_max: float = 40.45,
    ) -> None:
        self.step_size = _require_finite_real(step_size, "step_size")
        v0 = _require_finite_real(v_init, "v_init")
        self.v_min = _require_finite_real(v_min, "v_min")
        self.v_max = _require_finite_real(v_max, "v_max")
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.v_min < 0.0 or self.v_max <= self.v_min:
            raise ValueError("Require 0 <= v_min < v_max")

        self.v_ref = float(np.clip(v0, self.v_min, self.v_max))
        self._direction = 1.0
        self._p_prev = 0.0
        self._dp_irr = 0.0
        self._phase = "init"

    def step(self, v_curr: float, p_curr: float) -> float:
        """Advance one control cycle from ``(v_curr, p_curr)`` and return ``v_next``."""
        v_val = _require_finite_real(v_curr, "v_curr")
        p_val = _require_finite_real(p_curr, "p_curr")

        if self._phase == "init":
            self._p_prev = p_val
            self._phase = "hold"
            return float(np.clip(v_val, self.v_min, self.v_max))

        if self._phase == "hold":
            self._dp_irr = p_val - self._p_prev
            self._p_prev = p_val
            self._phase = "perturb"
            v_next = v_val + self._direction * self.step_size
            return float(np.clip(v_next, self.v_min, self.v_max))

        dp_total = p_val - self._p_prev
        dp_voltage = dp_total - self._dp_irr
        if dp_voltage < 0.0:
            self._direction = -self._direction

        self._p_prev = p_val
        self._phase = "hold"
        return float(np.clip(v_val, self.v_min, self.v_max))


def generate_trapezoidal_irradiance_profile(
    segment: EN50530RampSegment,
    *,
    dt_s: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize ``(time_s, irradiance_wm2)`` for one EN 50530 trapezoidal segment."""
    dt = _require_finite_real(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}")

    ramp_dur = (segment.g_high_wm2 - segment.g_low_wm2) / segment.slew_rate_wm2_s
    cycle_dur = 2.0 * segment.dwell_s + 2.0 * ramp_dur
    total_dur = segment.repetitions * cycle_dur + segment.dwell_s
    n_steps = max(5, int(np.round(total_dur / dt)) + 1)
    t_arr = np.linspace(0.0, total_dur, n_steps, dtype=np.float64)
    g_arr = np.empty_like(t_arr)

    for idx, t_val in enumerate(t_arr):
        if t_val >= segment.repetitions * cycle_dur:
            g_arr[idx] = segment.g_low_wm2
            continue
        tau = t_val % cycle_dur
        if tau < segment.dwell_s:
            g_arr[idx] = segment.g_low_wm2
        elif tau < segment.dwell_s + ramp_dur:
            g_arr[idx] = segment.g_low_wm2 + segment.slew_rate_wm2_s * (
                tau - segment.dwell_s
            )
        elif tau < 2.0 * segment.dwell_s + ramp_dur:
            g_arr[idx] = segment.g_high_wm2
        else:
            g_arr[idx] = segment.g_high_wm2 - segment.slew_rate_wm2_s * (
                tau - 2.0 * segment.dwell_s - ramp_dur
            )

    return t_arr, g_arr


def build_en50530_composite_profile(
    segments: Sequence[EN50530RampSegment] | None = None,
    *,
    dt_s: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], tuple[str, ...]]:
    """Concatenate multiple EN 50530 ramp segments onto a unified time base."""
    if segments is None:
        segments = (
            EN50530RampSegment(
                label="EN 50530 Low-Med Ramp (100-500 W/m² @ 25 W/m²/s)",
                g_low_wm2=100.0,
                g_high_wm2=500.0,
                slew_rate_wm2_s=25.0,
                dwell_s=6.0,
                repetitions=1,
            ),
            EN50530RampSegment(
                label="EN 50530 High-Slew Ramp (300-1000 W/m² @ 50 W/m²/s)",
                g_low_wm2=300.0,
                g_high_wm2=1000.0,
                slew_rate_wm2_s=50.0,
                dwell_s=8.0,
                repetitions=1,
            ),
        )
    if len(segments) == 0:
        raise ValueError("segments must contain at least one EN50530RampSegment")

    t_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    boundaries: list[float] = []
    labels: list[str] = []
    t_offset = 0.0

    for idx, seg in enumerate(segments):
        t_seg, g_seg = generate_trapezoidal_irradiance_profile(seg, dt_s=dt_s)
        boundaries.append(t_offset)
        labels.append(seg.label)
        if idx > 0:
            t_seg = t_seg[1:] + t_offset
            g_seg = g_seg[1:]
        t_parts.append(t_seg)
        g_parts.append(g_seg)
        t_offset = float(t_seg[-1])

    return (
        np.concatenate(t_parts),
        np.concatenate(g_parts),
        tuple(boundaries),
        tuple(labels),
    )


@dataclass
class EN50530ControllerSummary:
    """Time-domain telemetry and EN 50530 dynamic MPPT efficiency for one controller."""

    name: str
    voltages: np.ndarray
    currents: np.ndarray
    powers: np.ndarray
    cell_temps_c: np.ndarray
    en50530_efficiency_pct: float
    rms_voltage_error_v: float
    peak_voltage_excursion_v: float


@dataclass
class EN50530BenchmarkResult:
    """Complete EN 50530 dynamic ramp & coupled electro-thermal benchmark output."""

    time_s: np.ndarray
    irradiance_wm2: np.ndarray
    ideal_mpp_power_w: np.ndarray
    ideal_mpp_voltage_v: np.ndarray
    ideal_cell_temp_c: np.ndarray
    open_circuit_cell_temp_c: np.ndarray
    segment_boundaries_s: tuple[float, ...]
    segment_labels: tuple[str, ...]
    controllers: dict[str, EN50530ControllerSummary]


def run_en50530_dynamic_benchmark(
    pv_model: SolarPVModel | None = None,
    thermal_model: LumpedThermalModel | None = None,
    *,
    segments: Sequence[EN50530RampSegment] | None = None,
    dt_s: float = 0.20,
    step_size_v: float = 0.22,
) -> EN50530BenchmarkResult:
    """Execute EN 50530 trapezoidal ramp test with coupled electro-thermal cell ODE."""
    pv = pv_model if pv_model is not None else SolarPVModel()
    th = thermal_model if thermal_model is not None else LumpedThermalModel()
    dt = _require_finite_real(dt_s, "dt_s")
    step_v = _require_finite_real(step_size_v, "step_size_v")
    if dt <= 0.0 or step_v <= 0.0:
        raise ValueError("dt_s and step_size_v must be positive")

    t_arr, g_arr, boundaries, labels = build_en50530_composite_profile(
        segments, dt_s=dt
    )
    n_steps = len(t_arr)

    v_min = 1.0
    v_max = pv.V_oc * 1.05

    v0_grid, _, p0_grid = pv.calculate_iv_curve(
        G=float(g_arr[0]), T_c_celsius=th.t_ambient_c, num_points=300
    )
    v_init = float(v0_grid[int(np.argmax(p0_grid))])
    t_init_c = th.steady_state_temp_c(float(g_arr[0]), float(np.max(p0_grid)))

    po = PerturbAndObserve(step_size=step_v, v_init=v_init, v_min=v_min, v_max=v_max)
    inc = IncrementalConductance(
        step_size=step_v, v_init=v_init, v_min=v_min, v_max=v_max, conductance_tol=1e-2
    )
    dppo = DeltaPPO(step_size=step_v * 1.35, v_init=v_init, v_min=v_min, v_max=v_max)

    names = ("Standard P&O", "Incremental Conductance", "Sera dP-P&O (Drift-Free)")
    v_hist = {name: np.zeros(n_steps, dtype=np.float64) for name in names}
    i_hist = {name: np.zeros(n_steps, dtype=np.float64) for name in names}
    p_hist = {name: np.zeros(n_steps, dtype=np.float64) for name in names}
    t_hist = {name: np.zeros(n_steps, dtype=np.float64) for name in names}

    v_state = dict.fromkeys(names, v_init)
    t_state = dict.fromkeys(names, t_init_c)

    ideal_p = np.zeros(n_steps, dtype=np.float64)
    ideal_v = np.zeros(n_steps, dtype=np.float64)
    ideal_t = np.zeros(n_steps, dtype=np.float64)
    oc_t = np.zeros(n_steps, dtype=np.float64)

    t_ideal_curr = t_init_c
    t_oc_curr = th.steady_state_temp_c(float(g_arr[0]), 0.0)

    for k in range(n_steps):
        g_now = float(g_arr[k])

        v_ref, _, p_ref = pv.calculate_iv_curve(
            G=g_now, T_c_celsius=t_ideal_curr, num_points=260
        )
        best_idx = int(np.argmax(p_ref))
        ideal_p[k] = float(p_ref[best_idx])
        ideal_v[k] = float(v_ref[best_idx])
        ideal_t[k] = t_ideal_curr
        oc_t[k] = t_oc_curr

        if k < n_steps - 1:
            t_ideal_curr = th.step_temperature(t_ideal_curr, g_now, ideal_p[k], dt)
            t_oc_curr = th.step_temperature(t_oc_curr, g_now, 0.0, dt)

        for name in names:
            tc_now = t_state[name]
            v_now = v_state[name]
            v_c, i_c, _ = pv.calculate_iv_curve(
                G=g_now, T_c_celsius=tc_now, num_points=260
            )
            if v_now <= 0.0:
                i_now = float(i_c[0])
            elif v_now >= float(v_c[-1]):
                i_now = 0.0
            else:
                i_now = float(np.interp(v_now, v_c, i_c))
            p_now = max(0.0, v_now * i_now)

            v_hist[name][k] = v_now
            i_hist[name][k] = i_now
            p_hist[name][k] = p_now
            t_hist[name][k] = tc_now

            if name == "Standard P&O":
                v_state[name] = po.step(v_now, p_now)
            elif name == "Incremental Conductance":
                v_state[name] = inc.step(v_now, i_now)
            else:
                v_state[name] = dppo.step(v_now, p_now)

            if k < n_steps - 1:
                t_state[name] = th.step_temperature(tc_now, g_now, p_now, dt)

    e_ideal = float(_trapz(ideal_p, t_arr))
    summaries: dict[str, EN50530ControllerSummary] = {}
    for name in names:
        e_ctrl = float(_trapz(p_hist[name], t_arr))
        eta = 100.0 * e_ctrl / e_ideal if e_ideal > 0.0 else 0.0
        v_diff = v_hist[name] - ideal_v
        rms_v = float(np.sqrt(np.mean(v_diff**2)))
        peak_excursion = float(np.max(np.abs(v_diff)))
        summaries[name] = EN50530ControllerSummary(
            name=name,
            voltages=v_hist[name],
            currents=i_hist[name],
            powers=p_hist[name],
            cell_temps_c=t_hist[name],
            en50530_efficiency_pct=eta,
            rms_voltage_error_v=rms_v,
            peak_voltage_excursion_v=peak_excursion,
        )

    return EN50530BenchmarkResult(
        time_s=t_arr,
        irradiance_wm2=g_arr,
        ideal_mpp_power_w=ideal_p,
        ideal_mpp_voltage_v=ideal_v,
        ideal_cell_temp_c=ideal_t,
        open_circuit_cell_temp_c=oc_t,
        segment_boundaries_s=boundaries,
        segment_labels=labels,
        controllers=summaries,
    )
