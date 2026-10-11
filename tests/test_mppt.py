"""Tests for the P&O and Incremental Conductance MPPT control engines.

These exercise the real controllers against the real single-diode solver: the
plant response is sampled from the model's own I-V/P-V curves (via linear
interpolation on a fine grid), not from duplicated analytic formulas.
"""

import numpy as np
import pytest

from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.pv_model import SolarPVModel

STEP = 0.15


@pytest.fixture(scope="module")
def pv():
    return SolarPVModel()


def curve(pv, G, T=25.0, num=4001):
    return pv.calculate_iv_curve(G=G, T_c_celsius=T, num_points=num)


def run_po(pv, profile, v_init=20.0, step=STEP, bounds=None):
    """Run P&O across an irradiance profile; return (p_true, p_track, v_ref)."""
    kwargs = {} if bounds is None else {"v_min": bounds[0], "v_max": bounds[1]}
    po = PerturbAndObserve(step_size=step, v_init=v_init, **kwargs)
    v = v_init
    p_true, p_track, v_refs = [], [], []
    for G in profile:
        v_arr, _, p_arr = curve(pv, G)
        p_true.append(float(np.max(p_arr)))
        p_meas = float(np.interp(v, v_arr, p_arr))
        p_track.append(p_meas)
        v_refs.append(v)
        v = po.step(v, p_meas)
    return np.array(p_true), np.array(p_track), np.array(v_refs)


def run_inc(pv, profile, v_init=20.0, step=STEP, bounds=None):
    kwargs = {} if bounds is None else {"v_min": bounds[0], "v_max": bounds[1]}
    inc = IncrementalConductance(step_size=step, v_init=v_init, **kwargs)
    v = v_init
    p_true, p_track, v_refs = [], [], []
    for G in profile:
        v_arr, i_arr, p_arr = curve(pv, G)
        p_true.append(float(np.max(p_arr)))
        p_meas = float(np.interp(v, v_arr, p_arr))
        i_meas = float(np.interp(v, v_arr, i_arr))
        p_track.append(p_meas)
        v_refs.append(v)
        v = inc.step(v, i_meas)
    return np.array(p_true), np.array(p_track), np.array(v_refs)


# ----------------------------------------------------------------------
# Convergence under stable STC conditions
# ----------------------------------------------------------------------
def test_po_mppt_efficiency(pv):
    """Historical acceptance test: >= 98% of true MPP power."""
    v_arr, _, p_arr = curve(pv, 1000.0)
    p_true_mpp = np.max(p_arr)
    po = PerturbAndObserve(step_size=STEP, v_init=20.0)
    v_track = 20.0
    for _ in range(150):
        p_track = np.interp(v_track, v_arr, p_arr)
        v_track = po.step(v_track, p_track)
    assert (p_track / p_true_mpp) * 100.0 >= 98.0


def test_inc_mppt_efficiency(pv):
    v_arr, i_arr, p_arr = curve(pv, 1000.0)
    p_true_mpp = np.max(p_arr)
    inc = IncrementalConductance(step_size=STEP, v_init=20.0)
    v_track = 20.0
    for _ in range(150):
        p_track = np.interp(v_track, v_arr, p_arr)
        i_track = np.interp(v_track, v_arr, i_arr)
        v_track = inc.step(v_track, i_track)
    assert (p_track / p_true_mpp) * 100.0 >= 98.0


def test_po_converges_to_mpp_voltage(pv):
    _, p_track, v_refs = run_po(pv, [1000.0] * 200)
    v_arr, _, p_arr = curve(pv, 1000.0)
    v_mpp = float(v_arr[np.argmax(p_arr)])
    assert abs(v_refs[-1] - v_mpp) <= 2 * STEP
    assert p_track[-1] / np.max(p_arr) >= 0.995


def test_po_steady_state_oscillation_bounded(pv):
    """P&O inherently oscillates, but within +/- one step of the MPP."""
    _, _, v_refs = run_po(pv, [1000.0] * 300)
    tail = v_refs[-50:]
    assert np.max(tail) - np.min(tail) <= 2 * STEP + 1e-9


def test_inc_steady_state_is_stable(pv):
    """INC must come to rest (no sustained oscillation) at the MPP."""
    _, p_track, v_refs = run_inc(pv, [1000.0] * 300)
    tail = v_refs[-50:]
    assert np.max(tail) - np.min(tail) <= STEP + 1e-9
    v_arr, _, p_arr = curve(pv, 1000.0)
    assert p_track[-1] / np.max(p_arr) >= 0.995


# ----------------------------------------------------------------------
# Dynamic response to changing irradiance and temperature
# ----------------------------------------------------------------------
def test_po_tracks_irradiance_steps(pv):
    profile = [1000.0] * 60 + [600.0] * 60 + [900.0] * 60
    p_true, p_track, _ = run_po(pv, profile)
    for start in (60, 120):  # just after each irradiance step
        settled = slice(start + 25, start + 60)
        ratio = p_track[settled] / p_true[settled]
        assert np.min(ratio) >= 0.97
    assert p_track[-1] / p_true[-1] >= 0.99


def test_inc_tracks_irradiance_steps(pv):
    profile = [1000.0] * 60 + [600.0] * 60 + [900.0] * 60
    p_true, p_track, _ = run_inc(pv, profile)
    for start in (60, 120):
        settled = slice(start + 25, start + 60)
        ratio = p_track[settled] / p_true[settled]
        assert np.min(ratio) >= 0.97
    assert p_track[-1] / p_true[-1] >= 0.99


def test_inc_tracks_temperature_step(pv):
    """MPP voltage shifts with temperature; INC must follow."""
    inc = IncrementalConductance(step_size=STEP, v_init=20.0)
    v = 20.0
    schedule = [(1000.0, 25.0)] * 80 + [(1000.0, 65.0)] * 80
    last_ratios = []
    for k, (G, T) in enumerate(schedule):
        v_arr, i_arr, p_arr = curve(pv, G, T)
        i_meas = float(np.interp(v, v_arr, i_arr))
        p_meas = float(np.interp(v, v_arr, p_arr))
        if k >= len(schedule) - 30:
            last_ratios.append(p_meas / float(np.max(p_arr)))
        v = inc.step(v, i_meas)
    assert np.min(last_ratios) >= 0.98


# ----------------------------------------------------------------------
# Voltage bounds and validation
# ----------------------------------------------------------------------
def test_bounds_confine_reference_voltage(pv):
    _, _, v_po = run_po(pv, [1000.0] * 50 + [200.0] * 50, bounds=(0.0, pv.V_oc))
    _, _, v_inc = run_inc(pv, [1000.0] * 50 + [200.0] * 50, bounds=(0.0, pv.V_oc))
    assert np.all(v_po >= 0.0) and np.all(v_po <= pv.V_oc)
    assert np.all(v_inc >= 0.0) and np.all(v_inc <= pv.V_oc)


def test_bounds_clamp_initial_state():
    po = PerturbAndObserve(step_size=0.1, v_init=200.0, v_min=0.0, v_max=40.0)
    assert po.v_prev == 40.0
    v_next = po.step(39.0, 100.0)
    assert v_next <= 40.0


@pytest.mark.parametrize("cls", [PerturbAndObserve, IncrementalConductance])
def test_controller_validation(cls):
    with pytest.raises(ValueError):
        cls(step_size=0.0)
    with pytest.raises(ValueError):
        cls(step_size=-1.0)
    with pytest.raises(ValueError):
        cls(v_min=10.0, v_max=5.0)
    with pytest.raises(ValueError):
        cls(v_init=float("nan"))
    ctrl = cls()
    with pytest.raises(ValueError):
        ctrl.step(float("nan"), 1.0)
    with pytest.raises(ValueError):
        ctrl.step(10.0, float("inf"))


def test_incremental_conductance_tol_parameter(pv):
    inc = IncrementalConductance(step_size=STEP, v_init=20.0, conductance_tol=1e-3)
    v_arr, i_arr, p_arr = curve(pv, 1000.0)
    v = 20.0
    for _ in range(400):
        i_meas = float(np.interp(v, v_arr, i_arr))
        v = inc.step(v, i_meas)
    p_final = float(np.interp(v, v_arr, p_arr))
    assert p_final / np.max(p_arr) >= 0.995
    with pytest.raises(ValueError):
        IncrementalConductance(conductance_tol=0.0)
