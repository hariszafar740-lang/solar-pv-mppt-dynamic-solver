"""Verification tests for Global MPPT controllers (PSO & Hybrid Scan + Adaptive INC)."""

import numpy as np
import pytest

from src.global_mppt import (
    HybridGlobalScanINC,
    ParticleSwarmMPPT,
    run_dynamic_gmppt_benchmark,
)
from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.partial_shading import ShadedPVString
from src.visualizer import plot_global_mppt_comparison


@pytest.fixture(scope="module")
def shaded_string():
    return ShadedPVString(n_modules=1, n_bypass_per_module=3)


def test_pso_and_hybrid_escape_high_voltage_local_trap(shaded_string):
    curve = shaded_string.solve_curve([1000.0, 750.0, 220.0], num_points=700)
    v_start = 0.86 * curve.v_oc_string

    po = PerturbAndObserve(step_size=0.20, v_init=v_start, v_min=1.0, v_max=curve.v_oc_string)
    inc = IncrementalConductance(
        step_size=0.20, v_init=v_start, v_min=1.0, v_max=curve.v_oc_string
    )
    pso = ParticleSwarmMPPT(v_min=1.0, v_max=curve.v_oc_string, seed=42)
    hybrid = HybridGlobalScanINC(v_min=1.0, v_max=curve.v_oc_string)

    v_po = v_inc = v_pso = v_hyb = v_start
    pso_hist, hyb_hist = [], []
    for _ in range(90):
        _, p_po = curve.sample_at_voltage(v_po)
        v_po = po.step(v_po, p_po)

        i_inc, _ = curve.sample_at_voltage(v_inc)
        v_inc = inc.step(v_inc, i_inc)

        i_pso, p_pso = curve.sample_at_voltage(v_pso)
        pso_hist.append((v_pso, p_pso))
        v_pso = pso.step(v_pso, i_pso)

        i_hyb, p_hyb = curve.sample_at_voltage(v_hyb)
        hyb_hist.append((v_hyb, p_hyb))
        v_hyb = hybrid.step(v_hyb, i_hyb)

    _, p_po_final = curve.sample_at_voltage(v_po)
    _, p_inc_final = curve.sample_at_voltage(v_inc)
    _, p_pso_final = curve.sample_at_voltage(v_pso)
    _, p_hyb_final = curve.sample_at_voltage(v_hyb)

    assert p_po_final < 0.60 * curve.gmpp.power_w
    assert p_inc_final < 0.60 * curve.gmpp.power_w
    assert p_pso_final >= 0.99 * curve.gmpp.power_w
    assert p_hyb_final >= 0.99 * curve.gmpp.power_w

    settled_pso_v = [v for v, _ in pso_hist[-15:]]
    assert max(settled_pso_v) - min(settled_pso_v) < 1e-9


def test_dynamic_three_stage_benchmark_outperforms_conventional(shaded_string, tmp_path):
    bench = run_dynamic_gmppt_benchmark(shaded_string)

    po_traj = bench.trajectories["Conventional P&O"]
    inc_traj = bench.trajectories["Conventional INC"]
    pso_traj = bench.trajectories["PSO Global MPPT"]
    hyb_traj = bench.trajectories["Hybrid Scan + Adaptive INC"]

    assert pso_traj.settled_efficiency_pct >= 99.0
    assert hyb_traj.settled_efficiency_pct >= 99.0
    assert po_traj.settled_efficiency_pct < 85.0
    assert inc_traj.settled_efficiency_pct < 85.0
    assert hyb_traj.tracking_efficiency_pct > po_traj.tracking_efficiency_pct + 10.0

    out_png = tmp_path / "global_mppt_comparison.png"
    plot_global_mppt_comparison(bench, save_path=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 8000


def test_bypassed_plateau_cloud_clearance_blind_spot_recovery(shaded_string):
    """Verify watchdog probe escapes low-V plateau when shaded substrings clear."""
    c_shaded = shaded_string.solve_curve([1000.0, 350.0, 350.0], num_points=600)
    c_clear = shaded_string.solve_curve([1000.0, 1000.0, 1000.0], num_points=600)

    pso = ParticleSwarmMPPT(
        v_min=1.0, v_max=c_clear.v_oc_string, watchdog_interval=20, seed=7
    )
    hybrid = HybridGlobalScanINC(
        v_min=1.0, v_max=c_clear.v_oc_string, watchdog_interval=20
    )

    v_pso = v_hyb = 34.0
    for curve in ([c_shaded] * 70) + ([c_clear] * 90):
        i_p, _ = curve.sample_at_voltage(v_pso)
        v_pso = pso.step(v_pso, i_p)

        i_h, _ = curve.sample_at_voltage(v_hyb)
        v_hyb = hybrid.step(v_hyb, i_h)

    _, p_pso_end = c_clear.sample_at_voltage(v_pso)
    _, p_hyb_end = c_clear.sample_at_voltage(v_hyb)
    assert p_pso_end >= 0.99 * c_clear.gmpp.power_w
    assert p_hyb_end >= 0.99 * c_clear.gmpp.power_w


def test_pso_seeded_reproducibility(shaded_string):
    curve = shaded_string.solve_curve([1000.0, 680.0, 260.0], num_points=500)
    runs = []
    for _ in range(2):
        ctrl = ParticleSwarmMPPT(v_min=1.0, v_max=curve.v_oc_string, seed=123)
        v = 30.0
        seq = []
        for _ in range(40):
            i_val, _ = curve.sample_at_voltage(v)
            v = ctrl.step(v, i_val)
            seq.append(v)
        runs.append(seq)
    assert np.allclose(runs[0], runs[1])


@pytest.mark.parametrize(
    "cls, kwargs, match",
    [
        (ParticleSwarmMPPT, {"v_min": -1.0, "v_max": 40.0}, "v_min"),
        (ParticleSwarmMPPT, {"v_min": 30.0, "v_max": 20.0}, "v_max"),
        (ParticleSwarmMPPT, {"n_particles": 2}, "n_particles"),
        (ParticleSwarmMPPT, {"w_min": 0.8, "w_max": 0.4}, "Inertia"),
        (ParticleSwarmMPPT, {"watchdog_interval": 2}, "watchdog_interval"),
        (HybridGlobalScanINC, {"n_scan_points": 2}, "n_scan_points"),
        (HybridGlobalScanINC, {"min_step_v": 0.5, "max_step_v": 0.1}, "Step bounds"),
        (HybridGlobalScanINC, {"watchdog_interval": 0}, "watchdog_interval"),
    ],
)
def test_global_mppt_validation_rejects_bad_configs(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(**kwargs)


def test_global_mppt_rejects_nonfinite_step_inputs():
    pso = ParticleSwarmMPPT()
    hyb = HybridGlobalScanINC()
    with pytest.raises(ValueError, match="finite"):
        pso.step(float("nan"), 5.0)
    with pytest.raises(ValueError, match="finite"):
        hyb.step(20.0, float("inf"))
