"""Verification tests for EN 50530 dynamic irradiance ramps & electro-thermal transient."""

import numpy as np
import pytest

from src.en50530_benchmark import (
    DeltaPPO,
    EN50530RampSegment,
    LumpedThermalModel,
    build_en50530_composite_profile,
    generate_trapezoidal_irradiance_profile,
    run_en50530_dynamic_benchmark,
)
from src.visualizer import plot_en50530_benchmark


def test_trapezoidal_irradiance_profile_endpoints_and_slew_bounds():
    seg = EN50530RampSegment(
        label="Test Ramp",
        g_low_wm2=200.0,
        g_high_wm2=800.0,
        slew_rate_wm2_s=40.0,
        dwell_s=5.0,
        repetitions=2,
    )
    t_arr, g_arr = generate_trapezoidal_irradiance_profile(seg, dt_s=0.25)

    assert np.isclose(g_arr[0], 200.0)
    assert np.isclose(g_arr[-1], 200.0)
    assert np.isclose(np.max(g_arr), 800.0, atol=1e-6)
    assert np.min(g_arr) >= 200.0 - 1e-9

    dt = np.diff(t_arr)
    dg_dt = np.abs(np.diff(g_arr) / dt)
    assert np.max(dg_dt) <= 40.0 + 1.0


def test_lumped_electro_thermal_coupling_cools_mppt_module_vs_open_circuit():
    th = LumpedThermalModel(
        c_th_j_m2k=2000.0,
        u_th_w_m2k=29.0,
        absorptance=0.90,
        module_area_m2=1.64,
        t_ambient_c=25.0,
    )
    t_eq_oc = th.steady_state_temp_c(1000.0, p_elec_w=0.0)
    t_eq_mppt = th.steady_state_temp_c(1000.0, p_elec_w=300.0)

    assert t_eq_oc > t_eq_mppt + 5.0

    t_curr = 25.0
    for _ in range(400):
        t_next = th.step_temperature(t_curr, 1000.0, 300.0, dt_s=2.0)
        assert t_next >= t_curr
        t_curr = t_next
    assert np.isclose(t_curr, t_eq_mppt, atol=1e-2)


def test_delta_ppo_eliminates_ramp_drift_and_achieves_high_en50530_efficiency(tmp_path):
    res = run_en50530_dynamic_benchmark()

    po = res.controllers["Standard P&O"]
    inc = res.controllers["Incremental Conductance"]
    dppo = res.controllers["Sera dP-P&O (Drift-Free)"]

    assert dppo.en50530_efficiency_pct >= 99.5
    assert dppo.rms_voltage_error_v < 0.60
    assert dppo.en50530_efficiency_pct > po.en50530_efficiency_pct + 8.0
    assert dppo.en50530_efficiency_pct > inc.en50530_efficiency_pct + 5.0
    assert dppo.peak_voltage_excursion_v < po.peak_voltage_excursion_v
    assert np.max(res.open_circuit_cell_temp_c) > np.max(res.ideal_cell_temp_c) + 1.5

    out_png = tmp_path / "en50530_dynamic_benchmark.png"
    plot_en50530_benchmark(res, save_path=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 10000


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "label": "x",
                "g_low_wm2": 0.0,
                "g_high_wm2": 500.0,
                "slew_rate_wm2_s": 10.0,
            },
            "g_low",
        ),
        (
            {
                "label": "x",
                "g_low_wm2": 600.0,
                "g_high_wm2": 500.0,
                "slew_rate_wm2_s": 10.0,
            },
            "g_low",
        ),
        (
            {
                "label": "x",
                "g_low_wm2": 100.0,
                "g_high_wm2": 500.0,
                "slew_rate_wm2_s": -5.0,
            },
            "slew",
        ),
        (
            {
                "label": "x",
                "g_low_wm2": 100.0,
                "g_high_wm2": 500.0,
                "slew_rate_wm2_s": 10.0,
                "repetitions": 0,
            },
            "repetitions",
        ),
    ],
)
def test_ramp_segment_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EN50530RampSegment(**kwargs)


def test_thermal_and_dppo_validation():
    with pytest.raises(ValueError, match="absorptance"):
        LumpedThermalModel(absorptance=1.5)
    with pytest.raises(ValueError, match="dt_s"):
        LumpedThermalModel().step_temperature(25.0, 1000.0, 200.0, dt_s=0.0)
    with pytest.raises(ValueError, match="step_size"):
        DeltaPPO(step_size=0.0)
    with pytest.raises(ValueError, match="segments"):
        build_en50530_composite_profile(())
    ctrl = DeltaPPO()
    with pytest.raises(ValueError, match="finite"):
        ctrl.step(float("nan"), 100.0)
