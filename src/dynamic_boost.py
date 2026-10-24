"""State-space averaged DC-DC boost converter, cascaded current-mode PI control & loss map.

Physical Formulation
--------------------
1. **State-Space Averaged Boost Power Stage (`[v_pv, i_L]`)**:
   Couples the nonlinear PV module current ``i_pv(v_pv, G, T_c)`` to an input
   DC filter capacitor ``C_in`` and boost inductor ``L`` feeding a stiff DC-link
   bus ``V_bus``:

       C_in * dv_pv/dt = i_pv(v_pv, G, T_c) - i_L
       L    * di_L/dt  = v_pv - R_eq(D) * i_L - (1 - D) * (V_bus + V_fd)

   where ``R_eq(D) = R_L + D * R_dson + (1 - D) * R_d`` models parasitic
   conduction drops, with discontinuous-conduction diode blocking (``i_L >= 0``).

2. **Cascaded Average-Current-Mode PI Control (`CascadedBoostPIController`)**:
   Direct single-loop duty-to-``v_pv`` control excites the lightly damped input
   ``L-C_in`` tank.  Industrial PV boost stages decouple the plant into two
   cascaded first-order loops (Erickson & Maksimovic; Villalva et al., IEEE TPEL):
   - **Outer PV-Voltage PI Loop** (acts on ``e_v = v_pv - v_ref`` with ``i_pv``
     feedforward to command inductor current reference ``i_L,ref``):
         i_L,ref = i_pv + K_pv * (v_pv - v_ref) + K_iv * integral(v_pv - v_ref) dt
   - **Inner Inductor-Current PI Loop** (tracks ``i_L,ref`` with ``1 - v_pv/V_bus``
     duty feedforward and anti-windup clamping):
         D = (1 - v_pv / V_bus) + K_pi * (i_L,ref - i_L) + K_ii * integral(e_i) dt

3. **Semiconductor & Magnetic Power-Loss Model (`BoostLossBreakdown`)**:
   Resolves inductor copper loss (``I_L,rms^2 * R_L``), MOSFET conduction and
   switching/``C_oss`` overlap losses, freewheeling diode forward/resistive +
   reverse-recovery losses, inductor core ripple losses, and European
   (``eta_Euro``) / CEC (``eta_CEC``) weighted efficiencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.mppt_algorithms import IncrementalConductance
from src.pv_model import SolarPVModel, _require_finite_real

__all__ = [
    "BoostLossBreakdown",
    "BoostPowerStageParams",
    "BoostTransientResult",
    "CascadedBoostPIController",
    "ConverterEfficiencyCurve",
    "DynamicBoostConverter",
    "compute_weighted_efficiencies",
    "evaluate_converter_efficiency_curve",
    "run_closed_loop_boost_transient",
]


@dataclass(frozen=True)
class BoostPowerStageParams:
    """Physical electrical parameters of a SiC/MOSFET DC-DC boost power stage."""

    v_bus_v: float = 60.0
    c_in_f: float = 220e-6
    inductance_h: float = 380e-6
    r_inductor_ohm: float = 0.045
    r_dson_ohm: float = 0.018
    v_diode_drop_v: float = 0.48
    r_diode_ohm: float = 0.016
    q_rr_c: float = 45e-9
    f_sw_hz: float = 50_000.0
    t_sw_overlap_s: float = 42e-9
    c_oss_f: float = 220e-12
    k_core_loss: float = 0.42
    duty_min: float = 0.05
    duty_max: float = 0.92
    i_l_max_a: float = 18.0

    def __post_init__(self) -> None:
        vbus = _require_finite_real(self.v_bus_v, "v_bus_v")
        cin = _require_finite_real(self.c_in_f, "c_in_f")
        ind = _require_finite_real(self.inductance_h, "inductance_h")
        rl = _require_finite_real(self.r_inductor_ohm, "r_inductor_ohm")
        rdson = _require_finite_real(self.r_dson_ohm, "r_dson_ohm")
        vfd = _require_finite_real(self.v_diode_drop_v, "v_diode_drop_v")
        rd = _require_finite_real(self.r_diode_ohm, "r_diode_ohm")
        qrr = _require_finite_real(self.q_rr_c, "q_rr_c")
        fsw = _require_finite_real(self.f_sw_hz, "f_sw_hz")
        tsw = _require_finite_real(self.t_sw_overlap_s, "t_sw_overlap_s")
        coss = _require_finite_real(self.c_oss_f, "c_oss_f")
        kcore = _require_finite_real(self.k_core_loss, "k_core_loss")
        dmin = _require_finite_real(self.duty_min, "duty_min")
        dmax = _require_finite_real(self.duty_max, "duty_max")
        ilmax = _require_finite_real(self.i_l_max_a, "i_l_max_a")

        if vbus <= 5.0:
            raise ValueError(f"v_bus_v must exceed 5.0 V, got {vbus}")
        if cin <= 0.0 or ind <= 0.0 or fsw <= 0.0 or ilmax <= 0.0:
            raise ValueError(
                "c_in_f, inductance_h, f_sw_hz, and i_l_max_a must be positive"
            )
        if any(x < 0.0 for x in (rl, rdson, vfd, rd, qrr, tsw, coss, kcore)):
            raise ValueError("Parasitic loss parameters must be non-negative")
        if not (0.0 <= dmin < dmax < 1.0):
            raise ValueError(
                f"Require 0.0 <= duty_min < duty_max < 1.0, got [{dmin}, {dmax}]"
            )


@dataclass(frozen=True)
class BoostLossBreakdown:
    """Detailed component-level power loss and efficiency breakdown at one point."""

    p_in_w: float
    p_out_w: float
    efficiency_pct: float
    duty_cycle: float
    delta_i_ripple_a: float
    p_inductor_copper_w: float
    p_mosfet_cond_w: float
    p_mosfet_sw_w: float
    p_diode_cond_w: float
    p_core_w: float

    @property
    def total_loss_w(self) -> float:
        return (
            self.p_inductor_copper_w
            + self.p_mosfet_cond_w
            + self.p_mosfet_sw_w
            + self.p_diode_cond_w
            + self.p_core_w
        )


class CascadedBoostPIController:
    """Two-loop Average-Current-Mode PI servo (Outer ``v_pv`` -> Inner ``i_L`` -> ``D``)."""

    def __init__(
        self,
        *,
        kp_v: float = 0.65,
        ki_v: float = 180.0,
        kp_i: float = 0.085,
        ki_i: float = 140.0,
        v_bus_v: float = 60.0,
        duty_min: float = 0.05,
        duty_max: float = 0.92,
        i_l_max_a: float = 18.0,
    ) -> None:
        self.kp_v = _require_finite_real(kp_v, "kp_v")
        self.ki_v = _require_finite_real(ki_v, "ki_v")
        self.kp_i = _require_finite_real(kp_i, "kp_i")
        self.ki_i = _require_finite_real(ki_i, "ki_i")
        self.v_bus_v = _require_finite_real(v_bus_v, "v_bus_v")
        self.duty_min = _require_finite_real(duty_min, "duty_min")
        self.duty_max = _require_finite_real(duty_max, "duty_max")
        self.i_l_max_a = _require_finite_real(i_l_max_a, "i_l_max_a")

        if any(g < 0.0 for g in (self.kp_v, self.ki_v, self.kp_i, self.ki_i)):
            raise ValueError("PI gains must be non-negative")
        if (self.kp_v == 0.0 and self.ki_v == 0.0) or (
            self.kp_i == 0.0 and self.ki_i == 0.0
        ):
            raise ValueError("Outer and inner PI loops must each have positive gain")
        if self.v_bus_v <= 5.0 or self.i_l_max_a <= 0.0:
            raise ValueError("v_bus_v > 5.0 V and i_l_max_a > 0 A required")
        if not (0.0 <= self.duty_min < self.duty_max < 1.0):
            raise ValueError("Require 0.0 <= duty_min < duty_max < 1.0")

        self.int_v = 0.0
        self.int_i = 0.0

    def reset(self) -> None:
        self.int_v = 0.0
        self.int_i = 0.0

    def update(
        self,
        v_pv: float,
        v_ref: float,
        i_l: float,
        i_pv_ff: float,
        dt_s: float,
    ) -> tuple[float, float]:
        """Return ``(duty_cycle, i_l_ref)`` for step ``dt_s``."""
        vpv = _require_finite_real(v_pv, "v_pv")
        vref = _require_finite_real(v_ref, "v_ref")
        il = _require_finite_real(i_l, "i_l")
        iff = _require_finite_real(i_pv_ff, "i_pv_ff")
        dt = _require_finite_real(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt}")

        # Outer loop: v_pv > v_ref -> command higher inductor current to discharge C_in
        err_v = vpv - vref
        cand_int_v = self.int_v + err_v * dt
        il_unsat = iff + self.kp_v * err_v + self.ki_v * cand_int_v
        il_ref = float(np.clip(il_unsat, 0.0, self.i_l_max_a))
        if not (
            (il_unsat > self.i_l_max_a and err_v > 0.0)
            or (il_unsat < 0.0 and err_v < 0.0)
        ):
            self.int_v = cand_int_v

        # Inner loop: i_L,ref > i_L -> increase duty D to ramp inductor current up
        err_i = il_ref - il
        d_ff = float(np.clip(1.0 - vpv / self.v_bus_v, self.duty_min, self.duty_max))
        cand_int_i = self.int_i + err_i * dt
        d_unsat = d_ff + self.kp_i * err_i + self.ki_i * cand_int_i
        duty = float(np.clip(d_unsat, self.duty_min, self.duty_max))
        if not (
            (d_unsat > self.duty_max and err_i > 0.0)
            or (d_unsat < self.duty_min and err_i < 0.0)
        ):
            self.int_i = cand_int_i

        return duty, il_ref


class DynamicBoostConverter:
    """State-space averaged DC-DC boost converter with semiconductor/magnetic loss model."""

    def __init__(self, params: BoostPowerStageParams | None = None) -> None:
        self.params = params if params is not None else BoostPowerStageParams()

    def compute_losses(self, v_pv: float, i_l: float, duty: float) -> BoostLossBreakdown:
        """Evaluate component-level losses and DC-bus output power at ``(v_pv, i_L, D)``."""
        vpv = max(0.0, _require_finite_real(v_pv, "v_pv"))
        il = max(0.0, _require_finite_real(i_l, "i_l"))
        p = self.params
        d = float(np.clip(_require_finite_real(duty, "duty"), p.duty_min, p.duty_max))

        p_in = vpv * il
        if p_in <= 1e-9 or il <= 1e-9:
            return BoostLossBreakdown(
                p_in_w=0.0,
                p_out_w=0.0,
                efficiency_pct=0.0,
                duty_cycle=d,
                delta_i_ripple_a=0.0,
                p_inductor_copper_w=0.0,
                p_mosfet_cond_w=0.0,
                p_mosfet_sw_w=0.0,
                p_diode_cond_w=0.0,
                p_core_w=0.0,
            )

        delta_i = (vpv * d) / (p.inductance_h * p.f_sw_hz)
        i_rms_sq = il * il + (delta_i * delta_i) / 12.0

        p_cu = i_rms_sq * p.r_inductor_ohm
        p_mos_cond = d * i_rms_sq * p.r_dson_ohm
        p_mos_sw = (
            0.5 * p.v_bus_v * il * p.t_sw_overlap_s * p.f_sw_hz
            + 0.5 * p.c_oss_f * (p.v_bus_v**2) * p.f_sw_hz
            + p.q_rr_c * p.v_bus_v * p.f_sw_hz
        )
        p_diode = (1.0 - d) * (p.v_diode_drop_v * il + i_rms_sq * p.r_diode_ohm)
        p_core = p.k_core_loss * (delta_i**2.1)

        total_loss = p_cu + p_mos_cond + p_mos_sw + p_diode + p_core
        p_out = max(0.0, p_in - total_loss)
        eta = 100.0 * p_out / p_in
        return BoostLossBreakdown(
            p_in_w=p_in,
            p_out_w=p_out,
            efficiency_pct=eta,
            duty_cycle=d,
            delta_i_ripple_a=delta_i,
            p_inductor_copper_w=p_cu,
            p_mosfet_cond_w=p_mos_cond,
            p_mosfet_sw_w=p_mos_sw,
            p_diode_cond_w=p_diode,
            p_core_w=p_core,
        )

    def equilibrium_duty(self, v_pv: float, i_l: float) -> float:
        """Solve steady-state CCM duty ratio accounting for parasitic resistive/diode drops."""
        vpv = _require_finite_real(v_pv, "v_pv")
        il = max(0.0, _require_finite_real(i_l, "i_l"))
        p = self.params
        denom = p.v_bus_v + p.v_diode_drop_v + il * (p.r_diode_ohm - p.r_dson_ohm)
        num = (
            p.v_bus_v
            + p.v_diode_drop_v
            + il * (p.r_inductor_ohm + p.r_diode_ohm)
            - vpv
        )
        d_eq = num / max(1.0, denom)
        return float(np.clip(d_eq, p.duty_min, p.duty_max))

    def step_rk4(
        self,
        v_pv: float,
        i_l: float,
        duty: float,
        dt_s: float,
        v_curve: np.ndarray,
        i_curve: np.ndarray,
    ) -> tuple[float, float, float]:
        """Advance ``(v_pv, i_L)`` by ``dt_s`` via 4th-order Runge-Kutta integration."""
        p = self.params
        d = float(np.clip(duty, p.duty_min, p.duty_max))
        r_eq = p.r_inductor_ohm + d * p.r_dson_ohm + (1.0 - d) * p.r_diode_ohm
        v_off = (1.0 - d) * (p.v_bus_v + p.v_diode_drop_v)
        v_oc = float(v_curve[-1])

        def pv_i(v_val: float) -> float:
            if v_val <= 0.0:
                return float(i_curve[0])
            if v_val >= v_oc:
                return 0.0
            return float(np.interp(v_val, v_curve, i_curve))

        def deriv(vp: float, il: float) -> tuple[float, float]:
            il_pos = max(0.0, il)
            ipv = pv_i(vp)
            dv = (ipv - il_pos) / p.c_in_f
            dil = (vp - r_eq * il_pos - v_off) / p.inductance_h
            if il_pos <= 0.0 and dil < 0.0:
                dil = 0.0
            return dv, dil

        k1_v, k1_i = deriv(v_pv, i_l)
        k2_v, k2_i = deriv(v_pv + 0.5 * dt_s * k1_v, max(0.0, i_l + 0.5 * dt_s * k1_i))
        k3_v, k3_i = deriv(v_pv + 0.5 * dt_s * k2_v, max(0.0, i_l + 0.5 * dt_s * k2_i))
        k4_v, k4_i = deriv(v_pv + dt_s * k3_v, max(0.0, i_l + dt_s * k3_i))

        v_next = max(0.5, v_pv + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v))
        i_next = max(0.0, i_l + (dt_s / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i))
        return v_next, i_next, pv_i(v_next)


@dataclass
class ConverterEfficiencyCurve:
    """Efficiency & loss-breakdown sweep across operating power levels."""

    v_bus_v: float
    load_fractions: np.ndarray
    p_in_w: np.ndarray
    p_out_w: np.ndarray
    efficiencies_pct: np.ndarray
    breakdowns: list[BoostLossBreakdown]
    euro_efficiency_pct: float
    cec_efficiency_pct: float
    peak_efficiency_pct: float


def compute_weighted_efficiencies(
    load_fractions: np.ndarray, efficiencies_pct: np.ndarray
) -> tuple[float, float]:
    """Compute standard European (``eta_Euro``) and California CEC (``eta_CEC``) weights."""
    euro_weights = (
        (0.05, 0.03),
        (0.10, 0.06),
        (0.20, 0.13),
        (0.30, 0.10),
        (0.50, 0.48),
        (1.00, 0.20),
    )
    cec_weights = (
        (0.10, 0.04),
        (0.20, 0.05),
        (0.30, 0.12),
        (0.50, 0.21),
        (0.75, 0.53),
        (1.00, 0.05),
    )
    eta_euro = sum(
        w * float(np.interp(frac, load_fractions, efficiencies_pct))
        for frac, w in euro_weights
    )
    eta_cec = sum(
        w * float(np.interp(frac, load_fractions, efficiencies_pct))
        for frac, w in cec_weights
    )
    return eta_euro, eta_cec


def evaluate_converter_efficiency_curve(
    pv_model: SolarPVModel | None = None,
    params: BoostPowerStageParams | None = None,
    *,
    num_points: int = 24,
) -> ConverterEfficiencyCurve:
    """Sweep irradiance from 5% to 100% STC and compute MPP boost loss breakdown."""
    if isinstance(num_points, (bool, np.bool_)) or not isinstance(
        num_points, (int, np.integer)
    ) or num_points < 6:
        raise ValueError(f"num_points must be an integer >= 6, got {num_points!r}")

    pv = pv_model if pv_model is not None else SolarPVModel()
    conv = DynamicBoostConverter(params)
    fracs = np.linspace(0.05, 1.00, int(num_points), dtype=np.float64)

    p_in_arr = np.zeros_like(fracs)
    p_out_arr = np.zeros_like(fracs)
    eta_arr = np.zeros_like(fracs)
    breakdowns: list[BoostLossBreakdown] = []

    for idx, frac in enumerate(fracs):
        g_val = 1000.0 * float(frac)
        v_c, i_c, p_c = pv.calculate_iv_curve(G=g_val, T_c_celsius=25.0, num_points=300)
        mpp_idx = int(np.argmax(p_c))
        v_mpp = float(v_c[mpp_idx])
        i_mpp = float(i_c[mpp_idx])
        d_eq = conv.equilibrium_duty(v_mpp, i_mpp)
        bd = conv.compute_losses(v_mpp, i_mpp, d_eq)
        p_in_arr[idx] = bd.p_in_w
        p_out_arr[idx] = bd.p_out_w
        eta_arr[idx] = bd.efficiency_pct
        breakdowns.append(bd)

    eta_euro, eta_cec = compute_weighted_efficiencies(fracs, eta_arr)
    return ConverterEfficiencyCurve(
        v_bus_v=conv.params.v_bus_v,
        load_fractions=fracs,
        p_in_w=p_in_arr,
        p_out_w=p_out_arr,
        efficiencies_pct=eta_arr,
        breakdowns=breakdowns,
        euro_efficiency_pct=eta_euro,
        cec_efficiency_pct=eta_cec,
        peak_efficiency_pct=float(np.max(eta_arr)),
    )


@dataclass
class BoostTransientResult:
    """Closed-loop cascaded MPPT + PI boost power-stage time-domain waveforms."""

    time_ms: np.ndarray
    irradiance_wm2: np.ndarray
    v_pv_v: np.ndarray
    v_ref_v: np.ndarray
    v_mpp_true_v: np.ndarray
    i_pv_a: np.ndarray
    i_inductor_a: np.ndarray
    i_inductor_ref_a: np.ndarray
    duty_cycle: np.ndarray
    p_pv_w: np.ndarray
    p_bus_out_w: np.ndarray
    avg_converter_efficiency_pct: float
    rms_servo_tracking_error_v: float


def run_closed_loop_boost_transient(
    pv_model: SolarPVModel | None = None,
    params: BoostPowerStageParams | None = None,
    *,
    stages: Sequence[tuple[float, float]] | None = None,
    dt_inner_s: float = 50e-6,
    mppt_decimation: int = 60,
) -> BoostTransientResult:
    """Simulate cascaded MPPT + 2-loop current-mode PI boost stage across irradiance steps."""
    pv = pv_model if pv_model is not None else SolarPVModel()
    conv = DynamicBoostConverter(params)
    p = conv.params

    dt = _require_finite_real(dt_inner_s, "dt_inner_s")
    if dt <= 0.0 or dt > 5e-4:
        raise ValueError(f"dt_inner_s must lie in (0, 5e-4] s, got {dt}")
    if isinstance(mppt_decimation, (bool, np.bool_)) or not isinstance(
        mppt_decimation, (int, np.integer)
    ) or mppt_decimation < 1:
        raise ValueError("mppt_decimation must be an integer >= 1")

    if stages is None:
        stages = (
            (1000.0, 0.045),
            (550.0, 0.045),
            (900.0, 0.045),
        )
    if len(stages) == 0:
        raise ValueError("stages must be non-empty")

    stage_data = []
    for g_val, dur_s in stages:
        g_clean = _require_finite_real(g_val, "irradiance")
        d_clean = _require_finite_real(dur_s, "duration_s")
        if g_clean <= 0.0 or d_clean <= 0.0:
            raise ValueError("Stage irradiance and duration_s must be positive")
        n_sub = max(10, int(np.round(d_clean / dt)))
        v_c, i_c, p_c = pv.calculate_iv_curve(G=g_clean, T_c_celsius=25.0, num_points=350)
        mpp_idx = int(np.argmax(p_c))
        stage_data.append((g_clean, n_sub, v_c, i_c, float(v_c[mpp_idx])))

    total_n = sum(item[1] for item in stage_data)
    t_ms = np.arange(total_n, dtype=np.float64) * (dt * 1e3)
    g_hist = np.zeros(total_n, dtype=np.float64)
    v_pv_hist = np.zeros(total_n, dtype=np.float64)
    v_ref_hist = np.zeros(total_n, dtype=np.float64)
    v_mpp_hist = np.zeros(total_n, dtype=np.float64)
    i_pv_hist = np.zeros(total_n, dtype=np.float64)
    i_l_hist = np.zeros(total_n, dtype=np.float64)
    i_lref_hist = np.zeros(total_n, dtype=np.float64)
    d_hist = np.zeros(total_n, dtype=np.float64)
    p_pv_hist = np.zeros(total_n, dtype=np.float64)
    p_out_hist = np.zeros(total_n, dtype=np.float64)

    _, _, v0_c, i0_c, vmpp0 = stage_data[0]
    v_pv = vmpp0 - 3.2
    i_pv = float(np.interp(v_pv, v0_c, i0_c))
    i_l = i_pv
    v_ref = v_pv

    ctrl = CascadedBoostPIController(
        kp_v=0.65,
        ki_v=180.0,
        kp_i=0.085,
        ki_i=140.0,
        v_bus_v=p.v_bus_v,
        duty_min=p.duty_min,
        duty_max=p.duty_max,
        i_l_max_a=p.i_l_max_a,
    )
    d_eq0 = conv.equilibrium_duty(v_pv, i_l)
    d_ff0 = float(np.clip(1.0 - v_pv / p.v_bus_v, p.duty_min, p.duty_max))
    ctrl.int_i = (d_eq0 - d_ff0) / ctrl.ki_i

    v_seed = max(5.0, v_pv - 0.15)
    mppt = IncrementalConductance(
        step_size=0.32,
        v_init=v_seed,
        v_min=5.0,
        v_max=min(pv.V_oc, p.v_bus_v - 2.0),
        conductance_tol=1.5e-2,
    )
    mppt.i_prev = float(np.interp(v_seed, v0_c, i0_c))

    k = 0
    for g_val, n_sub, v_c, i_c, v_mpp in stage_data:
        i_pv = float(np.interp(v_pv, v_c, i_c))
        for _ in range(n_sub):
            if k > 0 and k % int(mppt_decimation) == 0:
                v_ref = mppt.step(v_pv, i_pv)

            duty, il_ref = ctrl.update(v_pv, v_ref, i_l, i_pv, dt)
            bd = conv.compute_losses(v_pv, i_l, duty)

            g_hist[k] = g_val
            v_pv_hist[k] = v_pv
            v_ref_hist[k] = v_ref
            v_mpp_hist[k] = v_mpp
            i_pv_hist[k] = i_pv
            i_l_hist[k] = i_l
            i_lref_hist[k] = il_ref
            d_hist[k] = duty
            p_pv_hist[k] = v_pv * i_pv
            p_out_hist[k] = bd.p_out_w

            v_pv, i_l, i_pv = conv.step_rk4(v_pv, i_l, duty, dt, v_c, i_c)
            k += 1

    e_in = float(np.sum(p_pv_hist))
    e_out = float(np.sum(p_out_hist))
    avg_eta = 100.0 * e_out / e_in if e_in > 0.0 else 0.0
    rms_servo = float(np.sqrt(np.mean((v_pv_hist - v_ref_hist) ** 2)))

    return BoostTransientResult(
        time_ms=t_ms,
        irradiance_wm2=g_hist,
        v_pv_v=v_pv_hist,
        v_ref_v=v_ref_hist,
        v_mpp_true_v=v_mpp_hist,
        i_pv_a=i_pv_hist,
        i_inductor_a=i_l_hist,
        i_inductor_ref_a=i_lref_hist,
        duty_cycle=d_hist,
        p_pv_w=p_pv_hist,
        p_bus_out_w=p_out_hist,
        avg_converter_efficiency_pct=avg_eta,
        rms_servo_tracking_error_v=rms_servo,
    )
