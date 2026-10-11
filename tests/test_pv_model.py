"""Regression and behavior tests for the single-diode PV module solver.

Tolerances
----------
* Grid-resolution tolerance for power/voltage on the default 200-point grid:
  near the (quadratic) maximum, the discrete grid loses at most
  ~|P''|*(dV/2)^2/2 < 0.01 W, so +/-0.25 W is conservative.  Precision
  assertions use a 4001-point grid with +/-0.02 W.
* The STC constraints I(0) = I_sc and I(V_mp) = I_mp are enforced exactly by
  the constructor's parameter extraction, so they are asserted to 1e-6.
"""

import numpy as np
import pytest

from src.pv_model import (
    MAX_OPERATING_TEMP_C,
    MIN_OPERATING_TEMP_C,
    SolarPVModel,
)

# The original (pre-repair) default parameter set.  These datasheet values are
# mutually inconsistent with the single-diode model: satisfying all of
# (I_sc, V_oc, I_mp, V_mp, R_s, R_sh) simultaneously requires the
# non-physical ideality factor n ~= 0.47.  The old implementation silently
# ignored I_mp/V_mp, fixed n = 1.3, and under-predicted STC power as
# 266.898 W against the configured nameplate 300.080 W (= 8.80 A * 34.1 V).
LEGACY_INCONSISTENT_PANEL = dict(
    I_sc=9.35, V_oc=38.2, I_mp=8.80, V_mp=34.1,
    N_s=60, R_s=0.221, R_sh=415.6, n=1.3, K_i=0.005,
)


@pytest.fixture(scope="module")
def pv():
    return SolarPVModel()


def mpp(v, i, p):
    j = int(np.argmax(p))
    return v[j], i[j], p[j]


# ----------------------------------------------------------------------
# STC reference condition (regression for the 300.08 W vs 266.898 W defect)
# ----------------------------------------------------------------------
def test_stc_max_power_equals_nameplate(pv):
    """The solver must reproduce the configured nameplate STC power."""
    v, i, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=4001)
    v_m, i_m, p_m = mpp(v, i, p)
    nameplate = pv.I_mp * pv.V_mp
    assert p_m == pytest.approx(nameplate, abs=0.02)
    assert v_m == pytest.approx(pv.V_mp, abs=0.05)
    assert i_m == pytest.approx(pv.I_mp, abs=0.01)


def test_stc_max_power_default_grid(pv):
    """Even on the default coarse grid the nameplate power must be met."""
    _, _, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0)
    assert np.max(p) == pytest.approx(pv.I_mp * pv.V_mp, abs=0.25)


def test_stc_constraints_exact(pv):
    """I(V=0) = I_sc and I(V_mp) = I_mp are enforced by extraction."""
    v, i, _ = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=4001)
    assert i[0] == pytest.approx(pv.I_sc, abs=1e-6)
    i_at_vmp = float(np.interp(pv.V_mp, v, i))
    assert i_at_vmp == pytest.approx(pv.I_mp, abs=1e-3)


def test_open_circuit_endpoints(pv):
    """The voltage grid spans past V_oc and current vanishes there."""
    v, i, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=4001)
    assert v[0] == 0.0
    assert v[-1] >= pv.V_oc
    assert i[-1] == 0.0
    assert np.all(p >= 0.0)
    assert np.all(np.diff(v) > 0.0)


def test_legacy_inconsistent_panel_is_rejected():
    """Root-cause regression: the historical default parameter set is
    inconsistent with the single-diode model at any physical ideality factor
    and must now be rejected loudly instead of silently under-producing."""
    with pytest.raises(ValueError, match="non-physical ideality factor"):
        SolarPVModel(**LEGACY_INCONSISTENT_PANEL)


# ----------------------------------------------------------------------
# Invalid input handling (regression for complex arrays / ZeroDivisionError)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("G", [-1.0, -1000.0, -1e-9])
def test_negative_irradiance_rejected(pv, G):
    with pytest.raises(ValueError, match="[Ii]rradiance"):
        pv.calculate_iv_curve(G=G)


@pytest.mark.parametrize("G", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_irradiance_rejected(pv, G):
    with pytest.raises(ValueError):
        pv.calculate_iv_curve(G=G)


def test_complex_irradiance_rejected(pv):
    with pytest.raises(ValueError):
        pv.calculate_iv_curve(G=complex(1000.0, 0.0))


@pytest.mark.parametrize(
    "T", [-273.15, -273.0, MIN_OPERATING_TEMP_C - 0.01, MAX_OPERATING_TEMP_C + 0.01,
          float("nan"), float("inf")]
)
def test_invalid_temperature_rejected(pv, T):
    with pytest.raises(ValueError):
        pv.calculate_iv_curve(G=1000.0, T_c_celsius=T)


def test_complex_temperature_rejected(pv):
    with pytest.raises(ValueError):
        pv.calculate_iv_curve(G=1000.0, T_c_celsius=complex(25.0, 0.0))


@pytest.mark.parametrize("num_points", [0, 1, -5, 2.5, "200", None, True])
def test_invalid_num_points_rejected(pv, num_points):
    with pytest.raises(ValueError):
        pv.calculate_iv_curve(G=1000.0, num_points=num_points)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(I_sc=0.0), "positive"),
        (dict(V_oc=-1.0), "positive"),
        (dict(I_mp=9.95), "I_mp"),
        (dict(V_mp=41.0), "V_mp"),
        (dict(R_s=-0.1), "R_s"),
        (dict(R_sh=0.0), "R_sh"),
        (dict(N_s=0), "N_s"),
        (dict(N_s=60.5), "N_s"),
        (dict(I_sc=float("nan")), "finite"),
        (dict(K_i=float("inf")), "finite"),
    ],
)
def test_invalid_constructor_parameters_rejected(kwargs, match):
    base = dict(I_sc=9.91, V_oc=40.45, I_mp=9.24, V_mp=32.48,
                N_s=60, R_s=0.358977, R_sh=171.705124, n=1.0, K_i=0.004162)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SolarPVModel(**base)


# ----------------------------------------------------------------------
# Zero / low irradiance and operating envelope
# ----------------------------------------------------------------------
def test_zero_irradiance_no_generation(pv):
    v, i, p = pv.calculate_iv_curve(G=0.0)
    assert v.dtype == np.float64 and i.dtype == np.float64 and p.dtype == np.float64
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(i)) and np.all(np.isfinite(p))
    assert v[0] == 0.0 and v[-1] > 0.0
    assert np.all(np.diff(v) > 0.0)
    assert np.all(i == 0.0)
    assert np.all(p == 0.0)


@pytest.mark.parametrize("G", [1.0, 5.0, 50.0, 200.0])
def test_low_irradiance_finite_and_physical(pv, G):
    v, i, p = pv.calculate_iv_curve(G=G, num_points=800)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(i)) and np.all(np.isfinite(p))
    assert np.isrealobj(v) and np.isrealobj(i) and np.isrealobj(p)
    assert np.max(p) > 0.0
    assert i[0] == pytest.approx(pv.I_sc * G / pv.G_ref, rel=0.02)


@pytest.mark.parametrize(
    "G", [0.0, 1.0, 100.0, 400.0, 600.0, 900.0, 1000.0, 1100.0, 1300.0]
)
@pytest.mark.parametrize("T", [MIN_OPERATING_TEMP_C, -10.0, 0.0, 25.0, 45.0,
                               65.0, MAX_OPERATING_TEMP_C])
def test_outputs_real_finite_across_envelope(pv, G, T):
    v, i, p = pv.calculate_iv_curve(G=G, T_c_celsius=T)
    for arr in (v, i, p):
        assert arr.dtype == np.float64
        assert np.isrealobj(arr)
        assert np.all(np.isfinite(arr))
    assert np.all(i >= 0.0) and np.all(p >= 0.0)


def test_irradiance_scaling_monotonic(pv):
    powers = []
    for G in [1000.0, 900.0, 600.0, 400.0, 200.0, 50.0]:
        _, _, p = pv.calculate_iv_curve(G=G, num_points=2001)
        powers.append(np.max(p))
    assert all(a > b for a, b in zip(powers, powers[1:], strict=False))


def test_short_circuit_current_linear_in_irradiance(pv):
    _, i_1000, _ = pv.calculate_iv_curve(G=1000.0, num_points=4001)
    _, i_500, _ = pv.calculate_iv_curve(G=500.0, num_points=4001)
    assert i_500[0] == pytest.approx(0.5 * i_1000[0], rel=5e-3)


def _voc_from_curve(v, i):
    """Voltage where the generating-quadrant current reaches zero (grid res)."""
    positive = np.nonzero(i > 0.0)[0]
    assert positive.size > 0
    return float(v[positive[-1]])


def test_thermal_derating_and_beta_voc(pv):
    """Power must derate with temperature; dVoc/dT must be negative and of
    physically plausible magnitude for c-Si (about -0.08..-0.16 V/K)."""
    _, _, p_25 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=2001)
    _, _, p_45 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=45.0, num_points=2001)
    _, _, p_75 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=75.0, num_points=2001)
    assert np.max(p_25) > np.max(p_45) > np.max(p_75)

    v_lo, i_lo, _ = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=8001)
    v_hi, i_hi, _ = pv.calculate_iv_curve(G=1000.0, T_c_celsius=75.0, num_points=8001)
    beta_voc = (_voc_from_curve(v_hi, i_hi) - _voc_from_curve(v_lo, i_lo)) / 50.0
    assert -0.16 < beta_voc < -0.08


def test_extracted_ideality_factor_is_physical(pv):
    assert 0.5 <= pv.n <= 2.5
    assert pv.I_ph_ref > 0.0 and pv.I_o_ref > 0.0
    assert pv.a_ref == pytest.approx(
        pv.n * pv.N_s * pv.k * pv.T_ref / pv.q, rel=1e-12
    )


def test_open_circuit_voltage_bound(pv):
    v, i, _ = pv.calculate_iv_curve(G=600.0, T_c_celsius=35.0, num_points=4001)
    bound = pv.open_circuit_voltage_bound(G=600.0, T_c_celsius=35.0)
    assert v[-1] == pytest.approx(bound, rel=1e-9)
    voc = _voc_from_curve(v, i)
    assert voc <= bound + 1e-9
    assert bound < 1.05 * pv.V_oc  # sanity: not wildly above nameplate


def test_public_interface_preserved(pv):
    """Attributes and call signature of the historical API still exist."""
    for attr in ("I_sc", "V_oc", "I_mp", "V_mp", "N_s", "R_s", "R_sh", "n",
                 "K_i", "q", "k", "T_ref", "G_ref"):
        assert hasattr(pv, attr)
    v, i, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=101)
    assert v.shape == i.shape == p.shape == (101,)
