"""Verification tests for series-string bypass-diode partial-shading physics."""

import numpy as np
import pytest

from src.mppt_algorithms import PerturbAndObserve
from src.partial_shading import (
    BypassDiodeSpec,
    ShadedPVString,
    get_benchmark_shading_profiles,
)
from src.pv_model import SolarPVModel
from src.visualizer import plot_partial_shading_curves


def test_uniform_stc_string_matches_single_diode_module_reference():
    pv = SolarPVModel()
    string = ShadedPVString(pv, n_modules=1, n_bypass_per_module=3)
    res = string.solve_curve([1000.0, 1000.0, 1000.0], temperatures_c=25.0, num_points=600)

    assert res.num_peaks == 1
    assert res.active_bypass_count_at_gmpp == 0
    assert np.isclose(res.i_sc_string, pv.I_sc, rtol=1e-4)
    assert np.isclose(res.v_oc_string, pv.V_oc, rtol=1e-4)
    assert np.isclose(res.gmpp.power_w, pv.REFERENCE_MODULE["cec_stc_power_W"], rtol=5e-4)
    assert res.mismatch_loss_pct < 0.1


def test_three_level_partial_shading_produces_three_peaks_and_bypass_steps():
    string = ShadedPVString(n_modules=1, n_bypass_per_module=3)
    res = string.solve_curve([1000.0, 650.0, 250.0], temperatures_c=25.0, num_points=700)

    assert res.num_peaks == 3
    assert sum(1 for pk in res.peaks if pk.is_global) == 1
    assert res.active_bypass_count_at_gmpp >= 1
    assert res.mismatch_loss_pct > 10.0

    peak_voltages = sorted(pk.voltage_v for pk in res.peaks)
    assert 8.0 < peak_voltages[0] < 14.5
    assert 19.0 < peak_voltages[1] < 27.0
    assert 30.0 < peak_voltages[2] < 39.5


def test_conventional_po_gets_trapped_in_high_voltage_local_maximum():
    string = ShadedPVString(n_modules=1, n_bypass_per_module=3)
    res = string.solve_curve([1000.0, 750.0, 220.0], temperatures_c=25.0, num_points=700)
    rightmost_local = res.highest_voltage_local_peak
    assert not rightmost_local.is_global
    assert res.gmpp.power_w > rightmost_local.power_w * 1.35

    po = PerturbAndObserve(step_size=0.15, v_init=36.0, v_min=0.0, v_max=res.v_oc_string)
    v_curr = 36.0
    for _ in range(150):
        p_curr = float(np.interp(v_curr, res.voltages, res.powers))
        v_curr = po.step(v_curr, p_curr)

    trapped_power = float(np.interp(v_curr, res.voltages, res.powers))
    assert abs(trapped_power - rightmost_local.power_w) < 3.0
    assert trapped_power < 0.75 * res.gmpp.power_w


def test_multi_module_series_string_scales_voltage_and_power():
    pv = SolarPVModel()
    two_mod_string = ShadedPVString(pv, n_modules=2, n_bypass_per_module=3)
    res_uniform = two_mod_string.solve_curve([1000.0] * 6, temperatures_c=25.0, num_points=600)

    assert np.isclose(res_uniform.v_oc_string, 2.0 * pv.V_oc, rtol=1e-3)
    assert np.isclose(
        res_uniform.gmpp.power_w,
        2.0 * pv.REFERENCE_MODULE["cec_stc_power_W"],
        rtol=1e-3,
    )


def test_zero_irradiance_and_plot_export(tmp_path):
    string = ShadedPVString()
    zero_res = string.solve_curve([0.0, 0.0, 0.0], temperatures_c=25.0, num_points=100)
    assert zero_res.gmpp.power_w == 0.0
    assert np.all(zero_res.powers == 0.0)

    profiles = get_benchmark_shading_profiles()
    results_map = {
        "Uniform STC (1000 / 1000 / 1000 W/m²)": string.solve_curve(profiles["uniform_stc"]),
        "Chimney Shade (1000 / 700 / 350 W/m²)": string.solve_curve(
            profiles["moderate_chimney_shade"]
        ),
        "Cloud + Soiling (850 / 520 / 200 W/m²)": string.solve_curve(
            profiles["severe_cloud_edge"]
        ),
    }
    out_png = tmp_path / "partial_shading_curves.png"
    plot_partial_shading_curves(results_map, save_path=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 5000


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_modules": 0}, "n_modules"),
        ({"n_bypass_per_module": 7}, "divisible"),
    ],
)
def test_shaded_string_constructor_rejects_invalid_topology(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ShadedPVString(**kwargs)


@pytest.mark.parametrize(
    "irradiances, temps, num_points, match",
    [
        ([1000.0, 800.0], 25.0, 200, "3 substring irradiances"),
        ([1000.0, -50.0, 800.0], 25.0, 200, "non-negative"),
        ([1000.0, 800.0, 600.0], [25.0, 120.0, 25.0], 200, "degC"),
        ([1000.0, 800.0, 600.0], 25.0, 5, "num_points"),
    ],
)
def test_solve_curve_rejects_invalid_inputs(irradiances, temps, num_points, match):
    string = ShadedPVString()
    with pytest.raises(ValueError, match=match):
        string.solve_curve(irradiances, temperatures_c=temps, num_points=num_points)


def test_bypass_diode_spec_validation():
    with pytest.raises(ValueError, match="i_sat_a"):
        BypassDiodeSpec(i_sat_a=-1e-5)
    with pytest.raises(ValueError, match="ideality_factor"):
        BypassDiodeSpec(ideality_factor=5.0)


def test_sample_at_voltage_and_package_exports():
    import src

    assert hasattr(src, "ShadedPVString")
    string = src.ShadedPVString()
    res = string.solve_curve([1000.0, 700.0, 350.0], num_points=500)

    i0, p0 = res.sample_at_voltage(0.0)
    assert np.isclose(i0, res.i_sc_string, rtol=1e-6)
    assert p0 == 0.0

    i_gmpp, p_gmpp = res.sample_at_voltage(res.gmpp.voltage_v)
    assert np.isclose(i_gmpp, res.gmpp.current_a, rtol=1e-3)
    assert np.isclose(p_gmpp, res.gmpp.power_w, rtol=1e-3)

    i_over, p_over = res.sample_at_voltage(res.v_oc_string + 5.0)
    assert i_over == 0.0 and p_over == 0.0
