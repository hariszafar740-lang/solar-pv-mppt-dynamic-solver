"""Tests for state-space averaged DC-DC boost stage, cascaded PI servo & loss breakdown."""

from __future__ import annotations

import numpy as np
import pytest

from src.dynamic_boost import (
    BoostLossBreakdown,
    BoostPowerStageParams,
    CascadedBoostPIController,
    DynamicBoostConverter,
    compute_weighted_efficiencies,
    evaluate_converter_efficiency_curve,
    run_closed_loop_boost_transient,
)
from src.pv_model import SolarPVModel
from src.visualizer import plot_dynamic_boost_converter


def test_boost_power_stage_params_validation() -> None:
    p = BoostPowerStageParams()
    assert p.v_bus_v == pytest.approx(60.0)

    with pytest.raises(ValueError, match="v_bus_v"):
        BoostPowerStageParams(v_bus_v=4.0)
    with pytest.raises(ValueError, match="positive"):
        BoostPowerStageParams(c_in_f=0.0)
    with pytest.raises(ValueError, match="positive"):
        BoostPowerStageParams(inductance_h=-1e-4)
    with pytest.raises(ValueError, match="non-negative"):
        BoostPowerStageParams(r_dson_ohm=-0.01)
    with pytest.raises(ValueError, match="duty_min"):
        BoostPowerStageParams(duty_min=0.9, duty_max=0.5)
    with pytest.raises(ValueError, match="finite"):
        BoostPowerStageParams(f_sw_hz=float("nan"))


def test_loss_breakdown_energy_conservation_and_zero_load() -> None:
    conv = DynamicBoostConverter()
    bd_zero = conv.compute_losses(0.0, 0.0, 0.5)
    assert bd_zero.p_in_w == 0.0
    assert bd_zero.p_out_w == 0.0
    assert bd_zero.total_loss_w == 0.0
    assert bd_zero.efficiency_pct == 0.0

    d_eq = conv.equilibrium_duty(32.48, 9.24)
    # Parasitic conduction + diode drop require slightly more duty than ideal 1 - V_pv/V_bus
    assert d_eq > (1.0 - 32.48 / 60.0)

    bd = conv.compute_losses(32.48, 9.24, d_eq)
    assert isinstance(bd, BoostLossBreakdown)
    assert bd.p_in_w == pytest.approx(32.48 * 9.24)
    assert bd.p_out_w == pytest.approx(bd.p_in_w - bd.total_loss_w, rel=1e-12)
    assert 95.0 < bd.efficiency_pct < 99.5
    assert bd.p_inductor_copper_w > 0.0
    assert bd.p_diode_cond_w > 0.0
    assert bd.p_mosfet_cond_w > 0.0
    assert bd.p_mosfet_sw_w > 0.0
    assert bd.p_core_w > 0.0


def test_cascaded_pi_controller_clamping_anti_windup_and_reset() -> None:
    ctrl = CascadedBoostPIController(
        kp_v=0.65,
        ki_v=180.0,
        kp_i=0.085,
        ki_i=140.0,
        v_bus_v=60.0,
        duty_min=0.05,
        duty_max=0.92,
        i_l_max_a=18.0,
    )
    # Huge positive voltage error -> saturate current ref at i_l_max_a and duty at duty_max
    for _ in range(30):
        duty, il_ref = ctrl.update(v_pv=55.0, v_ref=20.0, i_l=0.5, i_pv_ff=9.0, dt_s=1e-4)
    assert il_ref == pytest.approx(18.0)
    assert duty == pytest.approx(0.92)
    # Anti-windup must keep integrators bounded rather than growing unboundedly
    assert abs(ctrl.int_v) < 0.05
    assert abs(ctrl.int_i) < 0.05

    ctrl.reset()
    assert ctrl.int_v == 0.0
    assert ctrl.int_i == 0.0

    with pytest.raises(ValueError, match="dt_s"):
        ctrl.update(30.0, 30.0, 5.0, 5.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        CascadedBoostPIController(kp_v=-0.1)
    with pytest.raises(ValueError, match="positive gain"):
        CascadedBoostPIController(kp_v=0.0, ki_v=0.0)


def test_rk4_step_dcm_diode_blocking_keeps_inductor_current_non_negative() -> None:
    pv = SolarPVModel()
    conv = DynamicBoostConverter()
    v_c, i_c, _ = pv.calculate_iv_curve(G=200.0, T_c_celsius=25.0, num_points=200)

    # Low duty (D = 0.05) with near-zero initial inductor current -> diode blocks reverse current
    v_next, i_next, i_pv_next = conv.step_rk4(30.0, 0.0, 0.05, 50e-6, v_c, i_c)
    assert i_next >= 0.0
    assert v_next > 30.0  # PV current charges C_in when i_L == 0
    assert i_pv_next >= 0.0


def test_efficiency_curves_euro_cec_and_bus_voltage_scaling() -> None:
    pv = SolarPVModel()
    c60 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=60.0))
    c120 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=120.0))

    assert len(c60.load_fractions) == 24
    assert 96.5 < c60.euro_efficiency_pct < 99.0
    assert 96.5 < c60.cec_efficiency_pct < 99.0
    assert c60.peak_efficiency_pct >= c60.euro_efficiency_pct

    # Higher DC-bus voltage incurs larger switching + C_oss + Q_rr loss at 5% light load
    assert c60.efficiencies_pct[0] > c120.efficiencies_pct[0] + 1.5
    assert c60.euro_efficiency_pct > c120.euro_efficiency_pct

    # Uniform 97.5% curve must yield exact 97.5% Euro and CEC weights (weights sum to 1.0)
    flat_euro, flat_cec = compute_weighted_efficiencies(
        np.linspace(0.05, 1.0, 20), np.full(20, 97.5)
    )
    assert flat_euro == pytest.approx(97.5, abs=1e-9)
    assert flat_cec == pytest.approx(97.5, abs=1e-9)

    with pytest.raises(ValueError, match="num_points"):
        evaluate_converter_efficiency_curve(pv, num_points=3)


def test_closed_loop_boost_transient_tracks_vmpp_tightly_and_plots(tmp_path) -> None:
    pv = SolarPVModel()
    tr = run_closed_loop_boost_transient(pv)

    assert len(tr.time_ms) == 2700
    assert np.all(tr.i_inductor_a >= 0.0)
    assert np.all((tr.duty_cycle >= 0.05) & (tr.duty_cycle <= 0.92))
    assert tr.rms_servo_tracking_error_v < 0.12
    assert 95.5 < tr.avg_converter_efficiency_pct < 99.0

    # End of each 45 ms stage must settle within 0.45 V of true instantaneous V_mpp
    for end_idx in (899, 1799, 2699):
        assert abs(tr.v_pv_v[end_idx] - tr.v_mpp_true_v[end_idx]) < 0.45

    c60 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=60.0))
    c120 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=120.0))
    out_png = tmp_path / "dynamic_boost.png"
    plot_dynamic_boost_converter(tr, c60, c120, save_path=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 40_000

    with pytest.raises(ValueError, match="dt_inner_s"):
        run_closed_loop_boost_transient(pv, dt_inner_s=0.0)
    with pytest.raises(ValueError, match="mppt_decimation"):
        run_closed_loop_boost_transient(pv, mppt_decimation=0)
    with pytest.raises(ValueError, match="stages"):
        run_closed_loop_boost_transient(pv, stages=[])
