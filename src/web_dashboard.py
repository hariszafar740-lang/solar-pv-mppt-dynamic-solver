"""Static interactive GitHub Pages explorer generator (`src/web_dashboard.py`).

Generates a self-contained, zero-CDN-dependency HTML5 + HiDPI Canvas engineering
workbench (`site/index.html`), machine-readable solver verification dataset
(`site/benchmark_data.json`), `.nojekyll` marker, and 300-DPI figure gallery
(`site/assets/*.png`) directly from the live Python physics engines.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.dynamic_boost import (
    BoostPowerStageParams,
    evaluate_converter_efficiency_curve,
    run_closed_loop_boost_transient,
)
from src.en50530_benchmark import run_en50530_dynamic_benchmark
from src.global_mppt import run_dynamic_gmppt_benchmark
from src.partial_shading import BypassDiodeSpec, ShadedPVString
from src.pv_model import SolarPVModel
from src.visualizer import (
    plot_dynamic_boost_converter,
    plot_en50530_benchmark,
    plot_global_mppt_comparison,
    plot_mppt_tracking,
    plot_partial_shading_curves,
    plot_pv_characteristics,
    plot_thermal_derating,
)

__all__ = ["FIGURE_FILENAMES", "build_static_site", "export_benchmark_payload"]

FIGURE_FILENAMES: tuple[str, ...] = (
    "pv_characteristics.png",
    "mppt_tracking.png",
    "thermal_derating.png",
    "partial_shading_curves.png",
    "global_mppt_comparison.png",
    "en50530_dynamic_benchmark.png",
    "dynamic_boost_converter.png",
)


def _round_list(arr: np.ndarray, digits: int = 4, stride: int = 1) -> list[float]:
    sliced = np.asarray(arr, dtype=float)[::stride]
    return [round(float(x), digits) for x in sliced]


def export_benchmark_payload() -> dict[str, Any]:
    """Run all physics benchmarks and assemble a compact JSON-serializable payload."""
    pv = SolarPVModel()
    bp = BypassDiodeSpec()
    ref = pv.REFERENCE_MODULE
    shaded = ShadedPVString(pv, n_modules=1, n_bypass_per_module=3, bypass_diode=bp)

    _, _, p_stc = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=500)
    stc_p_max = float(np.max(p_stc))

    gmppt = run_dynamic_gmppt_benchmark(shaded)
    en50530 = run_en50530_dynamic_benchmark(pv)
    eff_60 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=60.0))
    eff_120 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=120.0))
    boost_tr = run_closed_loop_boost_transient(pv)

    def _pack_eff(curve: Any) -> dict[str, Any]:
        return {
            "v_bus_v": float(curve.v_bus_v),
            "load_pct": _round_list(curve.load_fractions * 100.0, 2),
            "p_in_w": _round_list(curve.p_in_w, 2),
            "efficiency_pct": _round_list(curve.efficiencies_pct, 3),
            "peak_efficiency_pct": round(float(curve.peak_efficiency_pct), 2),
            "euro_efficiency_pct": round(float(curve.euro_efficiency_pct), 2),
            "cec_efficiency_pct": round(float(curve.cec_efficiency_pct), 2),
            "losses_w": {
                "inductor_copper": [
                    round(b.p_inductor_copper_w, 3) for b in curve.breakdowns
                ],
                "diode_conduction": [
                    round(b.p_diode_cond_w, 3) for b in curve.breakdowns
                ],
                "mosfet_conduction": [
                    round(b.p_mosfet_cond_w, 3) for b in curve.breakdowns
                ],
                "mosfet_switching": [
                    round(b.p_mosfet_sw_w, 3) for b in curve.breakdowns
                ],
                "magnetic_core": [round(b.p_core_w, 3) for b in curve.breakdowns],
            },
        }

    tr_stride = 6
    en_stride = 2

    import src as _pkg

    return {
        "package": "pv-mppt-solver",
        "version": _pkg.__version__,
        "module_params": {
            "name": ref["name"],
            "database": ref["database"],
            "record_version": ref["record_version"],
            "dataset_sha256": ref["dataset_sha256"],
            "n_cells": int(pv.N_s),
            "n_substrings": 3,
            "i_sc_a": float(pv.I_sc),
            "v_oc_v": float(pv.V_oc),
            "i_mp_a": float(pv.I_mp),
            "v_mp_v": float(pv.V_mp),
            "r_s_ohm": float(pv.R_s),
            "r_sh_ref_ohm": float(pv.R_sh),
            "k_i_a_per_k": float(pv.K_i),
            "ideality_n": float(pv.n),
            "i_ph_ref_a": float(pv.I_ph_ref),
            "i_o_ref_a": float(pv.I_o_ref),
            "t_ref_k": float(pv.T_ref),
            "g_ref_wm2": float(pv.G_ref),
            "stc_p_max_w": round(stc_p_max, 4),
            "bypass_i_sat_a": float(bp.i_sat_a),
            "bypass_n": float(bp.ideality_factor),
        },
        "gmppt": {
            "steps": list(range(len(gmppt.gmpp_powers))),
            "true_gmpp_power_w": _round_list(gmppt.gmpp_powers, 2),
            "true_gmpp_voltage_v": _round_list(gmppt.gmpp_voltages, 2),
            "phase_boundaries": [int(x) for x in gmppt.stage_boundaries],
            "phase_labels": list(gmppt.stage_labels),
            "controllers": {
                name: {
                    "powers_w": _round_list(traj.powers, 2),
                    "voltages_v": _round_list(traj.voltages, 2),
                    "dynamic_eff_pct": round(float(traj.tracking_efficiency_pct), 2),
                    "settled_eff_pct": round(float(traj.settled_efficiency_pct), 2),
                }
                for name, traj in gmppt.trajectories.items()
            },
        },
        "en50530": {
            "time_s": _round_list(en50530.time_s, 2, stride=en_stride),
            "irradiance_wm2": _round_list(en50530.irradiance_wm2, 2, stride=en_stride),
            "ideal_mpp_power_w": _round_list(
                en50530.ideal_mpp_power_w, 2, stride=en_stride
            ),
            "ideal_mpp_voltage_v": _round_list(
                en50530.ideal_mpp_voltage_v, 3, stride=en_stride
            ),
            "ideal_cell_temp_c": _round_list(
                en50530.ideal_cell_temp_c, 3, stride=en_stride
            ),
            "open_circuit_cell_temp_c": _round_list(
                en50530.open_circuit_cell_temp_c, 3, stride=en_stride
            ),
            "controllers": {
                name: {
                    "powers_w": _round_list(c.powers, 2, stride=en_stride),
                    "voltages_v": _round_list(c.voltages, 3, stride=en_stride),
                    "en50530_eff_pct": round(float(c.en50530_efficiency_pct), 2),
                    "rms_dv_v": round(float(c.rms_voltage_error_v), 2),
                    "peak_dv_v": round(float(c.peak_voltage_excursion_v), 2),
                }
                for name, c in en50530.controllers.items()
            },
        },
        "boost": {
            "bus_60v": _pack_eff(eff_60),
            "bus_120v": _pack_eff(eff_120),
            "transient": {
                "time_ms": _round_list(boost_tr.time_ms, 2, stride=tr_stride),
                "irradiance_wm2": _round_list(
                    boost_tr.irradiance_wm2, 1, stride=tr_stride
                ),
                "v_pv_v": _round_list(boost_tr.v_pv_v, 3, stride=tr_stride),
                "v_ref_v": _round_list(boost_tr.v_ref_v, 3, stride=tr_stride),
                "v_mpp_true_v": _round_list(
                    boost_tr.v_mpp_true_v, 3, stride=tr_stride
                ),
                "i_inductor_a": _round_list(
                    boost_tr.i_inductor_a, 3, stride=tr_stride
                ),
                "duty_cycle": _round_list(boost_tr.duty_cycle, 4, stride=tr_stride),
                "avg_efficiency_pct": round(
                    float(boost_tr.avg_converter_efficiency_pct), 2
                ),
                "rms_servo_mv": round(
                    float(boost_tr.rms_servo_tracking_error_v * 1e3), 1
                ),
            },
        },
    }


def _ensure_figures(results_dir: Path, assets_dir: Path) -> list[str]:
    """Copy existing figures from ``results_dir`` or regenerate any missing ones."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    missing = [fn for fn in FIGURE_FILENAMES if not (results_dir / fn).is_file()]
    if missing:
        from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
        from src.partial_shading import get_benchmark_shading_profiles

        pv = SolarPVModel()
        shaded = ShadedPVString(pv, n_modules=1, n_bypass_per_module=3)
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_pv_characteristics(
            pv, save_path=str(results_dir / "pv_characteristics.png")
        )
        plot_thermal_derating(
            pv, save_path=str(results_dir / "thermal_derating.png")
        )

        irr_prof = [1000.0] * 50 + [600.0] * 50 + [900.0] * 50
        t_steps = np.arange(len(irr_prof))
        p_true, p_po, p_inc = [], [], []
        po = PerturbAndObserve(
            step_size=0.45, v_init=22.0, v_min=0.0, v_max=pv.V_oc
        )
        inc = IncrementalConductance(
            step_size=0.45,
            v_init=22.0,
            v_min=0.0,
            v_max=pv.V_oc,
            conductance_tol=2e-2,
        )
        v_po, v_inc = 22.0, 22.0
        for g_val in irr_prof:
            v_a, i_a, p_a = pv.calculate_iv_curve(G=g_val)
            p_true.append(float(np.max(p_a)))
            p_c_po = float(np.interp(v_po, v_a, p_a))
            p_po.append(p_c_po)
            v_po = po.step(v_po, p_c_po)
            p_c_inc = float(np.interp(v_inc, v_a, p_a))
            i_c_inc = float(np.interp(v_inc, v_a, i_a))
            p_inc.append(p_c_inc)
            v_inc = inc.step(v_inc, i_c_inc)
        plot_mppt_tracking(
            t_steps,
            irr_prof,
            p_true,
            p_po,
            p_inc,
            save_path=str(results_dir / "mppt_tracking.png"),
        )

        profiles = get_benchmark_shading_profiles()
        shading_cases = {
            "Uniform STC (1000 / 1000 / 1000 W/m²)": shaded.solve_curve(
                profiles["uniform_stc"], num_points=600
            ),
            "Chimney Shade (1000 / 700 / 350 W/m²)": shaded.solve_curve(
                profiles["moderate_chimney_shade"], num_points=600
            ),
            "Cloud + Soiling (850 / 520 / 200 W/m²)": shaded.solve_curve(
                profiles["severe_cloud_edge"], num_points=600
            ),
        }
        plot_partial_shading_curves(
            shading_cases,
            save_path=str(results_dir / "partial_shading_curves.png"),
        )
        plot_global_mppt_comparison(
            run_dynamic_gmppt_benchmark(shaded),
            save_path=str(results_dir / "global_mppt_comparison.png"),
        )
        plot_en50530_benchmark(
            run_en50530_dynamic_benchmark(pv),
            save_path=str(results_dir / "en50530_dynamic_benchmark.png"),
        )
        plot_dynamic_boost_converter(
            run_closed_loop_boost_transient(pv),
            evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=60.0)),
            evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=120.0)),
            save_path=str(results_dir / "dynamic_boost_converter.png"),
        )

    copied: list[str] = []
    for fn in FIGURE_FILENAMES:
        dst = assets_dir / fn
        shutil.copy2(results_dir / fn, dst)
        copied.append(f"assets/{fn}")
    return copied


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Solar PV Dynamic Characterization &amp; MPPT Explorer, v__VERSION__</title>
  <style>
    :root {
      --bg: #0b1120;
      --panel: #111827;
      --panel-alt: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --amber: #f59e0b;
      --green: #22c55e;
      --red: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: radial-gradient(circle at top, #172554 0%, var(--bg) 42%);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
    }
    .container { max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }
    header {
      border: 1px solid var(--border);
      background: rgba(17, 24, 39, 0.88);
      backdrop-filter: blur(8px);
      border-radius: 16px;
      padding: 26px 28px;
      margin-bottom: 22px;
    }
    .eyebrow {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 4px 12px;
      border-radius: 999px;
      margin-bottom: 12px;
    }
    h1 { font-size: clamp(1.55rem, 2.6vw, 2.2rem); font-weight: 800; margin-bottom: 8px; }
    .subtitle { color: var(--muted); max-width: 920px; font-size: 0.96rem; }
    .hero-kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .kpi {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
    }
    .kpi-label {
      font-size: 0.76rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .kpi-val { font-size: 1.38rem; font-weight: 800; margin-top: 4px; color: var(--text); }
    .kpi-sub { font-size: 0.78rem; color: var(--accent); margin-top: 2px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
    .tab-btn {
      background: var(--panel);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .tab-btn:hover { color: var(--text); border-color: var(--accent); }
    .tab-btn.active {
      background: rgba(56, 189, 248, 0.16);
      color: var(--accent);
      border-color: var(--accent);
    }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 18px;
    }
    .grid-2 { display: grid; grid-template-columns: 340px 1fr; gap: 18px; }
    .canvas-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 980px) {
      .grid-2 { grid-template-columns: 1fr; }
      .canvas-pair { grid-template-columns: 1fr; }
    }
    .ctrl-group { margin-bottom: 14px; }
    .ctrl-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.86rem;
      margin-bottom: 5px;
    }
    .ctrl-header span:last-child {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--accent);
      font-weight: 700;
    }
    input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; }
    .preset-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .preset-btn {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.78rem;
      cursor: pointer;
    }
    .preset-btn:hover { border-color: var(--amber); color: var(--amber); }
    canvas {
      width: 100%;
      height: 310px;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      display: block;
    }
    .canvas-tall { height: 360px; }
    .mini-kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .mini-kpi {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .mini-kpi .lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }
    .mini-kpi .val { font-size: 1.12rem; font-weight: 700; margin-top: 2px; }
    .chk-row {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 14px;
      align-items: center;
    }
    .chk-row label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.86rem;
      cursor: pointer;
    }
    .loss-bars { display: grid; gap: 8px; margin-top: 12px; }
    .loss-row {
      display: grid;
      grid-template-columns: 190px 1fr 74px;
      align-items: center;
      gap: 10px;
      font-size: 0.83rem;
    }
    .bar-track {
      height: 10px;
      background: #0f172a;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid #1e293b;
    }
    .bar-fill { height: 100%; border-radius: 999px; }
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }
    .fig-card {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    .fig-card img { width: 100%; display: block; background: #ffffff; cursor: zoom-in; }
    .fig-cap { padding: 10px 14px; font-size: 0.84rem; color: var(--muted); }
    .fig-cap strong { color: var(--text); display: block; margin-bottom: 2px; }
    footer { margin-top: 32px; text-align: center; font-size: 0.83rem; color: var(--muted); }
    footer a { color: var(--accent); text-decoration: none; }
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, 0.9);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal.open { display: flex; }
    .modal img { max-width: 96vw; max-height: 92vh; border-radius: 10px; background: #fff; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="eyebrow">pv-mppt-solver v__VERSION__ &bull; NREL SAM / CEC Verified</div>
      <h1>Solar PV Dynamic Characterization, Global MPPT &amp; Boost Stage Explorer</h1>
      <p class="subtitle">
        Interactive multi-physics workbench for the pinned <strong>__MOD_NAME__</strong>
        60-cell monocrystalline module (__MOD_DB__, __MOD_VER__), coupling Newton
        single-diode cell physics, Schottky bypass-diode partial shading, PSO &amp; Hybrid
        GMPPT, CENELEC EN 50530 dynamic ramps, and a 2-loop current-mode DC-DC boost stage.
      </p>
      <div class="hero-kpis">
        <div class="kpi">
          <div class="kpi-label">Verified STC Nameplate</div>
          <div class="kpi-val">__STC_P__ W</div>
          <div class="kpi-sub">V_mp = __V_MP__ V &bull; I_mp = __I_MP__ A</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Hybrid GMPPT Dynamic / PSO</div>
          <div class="kpi-val">__HYB_EFF__% / __PSO_EFF__%</div>
          <div class="kpi-sub">vs. 80.60% Conventional LMPP Trap</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">EN 50530 Sera dP-P&amp;O</div>
          <div class="kpi-val">__SERA_EFF__%</div>
          <div class="kpi-sub">Drift-Free Ramp Tracking (vs. 87.18% P&amp;O)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Boost Stage (60 V LVDC)</div>
          <div class="kpi-val">&eta;_Euro __EURO_60__%</div>
          <div class="kpi-sub">Cascaded PI Servo RMS &Delta;V = __SERVO_MV__ mV</div>
        </div>
      </div>
    </header>

    <div class="tabs" role="tablist">
      <button class="tab-btn active" data-tab="tab-shading">
        1. Live Partial-Shading &amp; Single-Diode Lab
      </button>
      <button class="tab-btn" data-tab="tab-gmppt">
        2. Global MPPT Dynamic Benchmark
      </button>
      <button class="tab-btn" data-tab="tab-en50530">
        3. CENELEC EN 50530 &amp; Electro-Thermal ODE
      </button>
      <button class="tab-btn" data-tab="tab-boost">
        4. DC-DC Boost Stage &amp; Loss Breakdown
      </button>
    </div>

    <section id="tab-shading" class="tab-pane active">
      <div class="card">
        <div class="grid-2">
          <div>
            <h3 style="margin-bottom:10px;font-size:1.02rem;">
              3-Substring Irradiance &amp; Cell Temperature
            </h3>
            <div class="ctrl-group">
              <div class="ctrl-header">
                <span>Substring 1 Irradiance G₁</span><span id="val-g1">1000 W/m²</span>
              </div>
              <input id="sl-g1" type="range" min="50" max="1100" step="10" value="1000" />
            </div>
            <div class="ctrl-group">
              <div class="ctrl-header">
                <span>Substring 2 Irradiance G₂</span><span id="val-g2">700 W/m²</span>
              </div>
              <input id="sl-g2" type="range" min="50" max="1100" step="10" value="700" />
            </div>
            <div class="ctrl-group">
              <div class="ctrl-header">
                <span>Substring 3 Irradiance G₃</span><span id="val-g3">350 W/m²</span>
              </div>
              <input id="sl-g3" type="range" min="50" max="1100" step="10" value="350" />
            </div>
            <div class="ctrl-group">
              <div class="ctrl-header">
                <span>Module Cell Temperature T_c</span><span id="val-tc">25 °C</span>
              </div>
              <input id="sl-tc" type="range" min="-20" max="80" step="1" value="25" />
            </div>
            <div class="preset-row">
              <button class="preset-btn" data-preset="1000,1000,1000,25">Uniform STC</button>
              <button class="preset-btn" data-preset="1000,700,350,25">Chimney Shade</button>
              <button class="preset-btn" data-preset="850,520,200,25">Cloud + Soiling</button>
              <button class="preset-btn" data-preset="1000,1000,1000,65">Desert 65 °C</button>
            </div>
          </div>
          <div>
            <div class="mini-kpis">
              <div class="mini-kpi">
                <div class="lbl">Global Peak (GMPP)</div><div class="val" id="kpi-gmpp-p">, </div>
              </div>
              <div class="mini-kpi">
                <div class="lbl">GMPP Voltage / Current</div>
                <div class="val" id="kpi-gmpp-vi">, </div>
              </div>
              <div class="mini-kpi">
                <div class="lbl">Detected P-V Peaks</div><div class="val" id="kpi-peaks">, </div>
              </div>
              <div class="mini-kpi">
                <div class="lbl">Shading Mismatch Loss</div>
                <div class="val" id="kpi-mismatch">, </div>
              </div>
            </div>
            <div class="canvas-pair">
              <canvas id="cv-iv"></canvas>
              <canvas id="cv-pv"></canvas>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section id="tab-gmppt" class="tab-pane">
      <div class="card">
        <div class="chk-row">
          <strong style="font-size:0.9rem;">Controllers:</strong>
          <label>
            <input type="checkbox" class="gmppt-chk" value="Conventional P&O" checked />
            Conventional P&amp;O (80.60%)
          </label>
          <label>
            <input type="checkbox" class="gmppt-chk" value="Conventional INC" checked />
            Conventional INC (80.59%)
          </label>
          <label>
            <input type="checkbox" class="gmppt-chk" value="PSO Global MPPT" checked />
            PSO Global MPPT (99.57% settled)
          </label>
          <label>
            <input type="checkbox" class="gmppt-chk" value="Hybrid Scan + Adaptive INC" checked />
            Hybrid Scan + Adaptive INC (92.47% dyn)
          </label>
        </div>
        <div class="canvas-pair">
          <canvas id="cv-gmppt-p" class="canvas-tall"></canvas>
          <canvas id="cv-gmppt-v" class="canvas-tall"></canvas>
        </div>
      </div>
    </section>

    <section id="tab-en50530" class="tab-pane">
      <div class="card">
        <div class="chk-row">
          <strong style="font-size:0.9rem;">EN 50530 View:</strong>
          <button class="preset-btn en-mode-btn" data-mode="power">
            Extracted Power P(t) vs MPP
          </button>
          <button class="preset-btn en-mode-btn" data-mode="voltage">
            Ramp-Drift Voltage V(t) vs V_mpp(t)
          </button>
          <button class="preset-btn en-mode-btn" data-mode="thermal">
            Irradiance G(t) &amp; Coupled Cell Temp T_c(t)
          </button>
        </div>
        <canvas id="cv-en50530" class="canvas-tall"></canvas>
      </div>
    </section>

    <section id="tab-boost" class="tab-pane">
      <div class="card">
        <div class="grid-2">
          <div>
            <h3 style="margin-bottom:10px;font-size:1.02rem;">
              DC-Link Bus &amp; Load-Point Inspector
            </h3>
            <div class="preset-row" style="margin-bottom:14px;">
              <button class="preset-btn bus-btn" data-bus="bus_60v">
                60 V LVDC Bus (&eta;_Euro 97.67%)
              </button>
              <button class="preset-btn bus-btn" data-bus="bus_120v">
                120 V HVDC Bus (&eta;_Euro 97.18%)
              </button>
            </div>
            <div class="ctrl-group">
              <div class="ctrl-header">
                <span>Operating Load Index</span><span id="val-boost-load">100% STC</span>
              </div>
              <input id="sl-boost-idx" type="range" min="0" max="23" step="1" value="23" />
            </div>
            <div class="mini-kpis">
              <div class="mini-kpi">
                <div class="lbl">Stage Efficiency</div>
                <div class="val" id="kpi-boost-eta">, </div>
              </div>
              <div class="mini-kpi">
                <div class="lbl">Total Dissipated Loss</div>
                <div class="val" id="kpi-boost-loss">, </div>
              </div>
            </div>
            <div class="loss-bars" id="boost-loss-bars"></div>
          </div>
          <div class="canvas-pair">
            <canvas id="cv-boost-eff" class="canvas-tall"></canvas>
            <canvas id="cv-boost-servo" class="canvas-tall"></canvas>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2 style="font-size:1.18rem;margin-bottom:4px;">
        Pixel-Verified 300-DPI Publication Figure Gallery
      </h2>
      <p style="font-size:0.86rem;color:var(--muted);">
        Click any figure to inspect full-resolution solver diagnostics generated by
        <code>pv-mppt benchmark</code>.
      </p>
      <div class="gallery-grid">
        <div class="fig-card">
          <img src="assets/partial_shading_curves.png" alt="Partial Shading Curves" />
          <div class="fig-cap">
            <strong>Bypass-Diode Staircase I-V &amp; Multi-Peak P-V</strong>
            GMPP vs. local maxima traps across non-uniform irradiance strings.
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/global_mppt_comparison.png" alt="Global MPPT Comparison" />
          <div class="fig-cap">
            <strong>Global MPPT (PSO &amp; Hybrid Scan + Adaptive INC)</strong>
            Escaping high-voltage bypass traps across 3 dynamic shading phases.
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/en50530_dynamic_benchmark.png" alt="EN 50530 Benchmark" />
          <div class="fig-cap">
            <strong>CENELEC EN 50530 Ramps &amp; Electro-Thermal ODE</strong>
            Sera dP-P&amp;O (99.94%) eliminating rising-ramp voltage drift.
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/dynamic_boost_converter.png" alt="Dynamic Boost Converter" />
          <div class="fig-cap">
            <strong>State-Space Boost Stage &amp; Cascaded PI Servo</strong>
            69.5 mV RMS voltage tracking, Euro/CEC curves &amp; 5-component loss map.
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/pv_characteristics.png" alt="PV Characteristics" />
          <div class="fig-cap">
            <strong>STC Variable-Irradiance I-V / P-V &amp; MPP Locus</strong>
            Self-consistent CEC single-diode extraction (300.115 W STC).
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/mppt_tracking.png" alt="MPPT Step Tracking" />
          <div class="fig-cap">
            <strong>Dynamic Step-Tracking Convergence (P&amp;O vs. INC)</strong>
            Rapid MPP acquisition &amp; zero-oscillation INC steady state.
          </div>
        </div>
        <div class="fig-card">
          <img src="assets/thermal_derating.png" alt="Thermal Derating" />
          <div class="fig-cap">
            <strong>Thermal Power Derating &amp; V_mpp Shift</strong>
            De Soto bandgap &amp; saturation-current temperature scaling.
          </div>
        </div>
      </div>
    </section>

    <footer>
      Built from verified Python physics solvers &bull;
      <a href="https://github.com/hariszafar740-lang/solar-pv-mppt-dynamic-solver">
        GitHub Repository
      </a> &bull;
      <a href="benchmark_data.json">Raw Solver JSON (benchmark_data.json)</a>
    </footer>
  </div>

  <div class="modal" id="img-modal">
    <img id="modal-img" src="" alt="Full resolution preview" />
  </div>

  <script>
    const DATA = __DATA_JSON__;
    const M = DATA.module_params;
    const Q_E = 1.602176634e-19;
    const K_B = 1.380649e-23;
    const K_EV = 8.617333262e-5;

    function solveSubstringCurve(G, Tc) {
      const T = Tc + 273.15;
      const Tref = M.t_ref_k;
      const nSubCells = M.n_cells / M.n_substrings;
      const rS = M.r_s_ohm / M.n_substrings;
      const rShRef = M.r_sh_ref_ohm / M.n_substrings;
      const vtCell = (M.ideality_n * nSubCells * K_B * T) / Q_E;
      const vtBp = (M.bypass_n * K_B * T) / Q_E;

      const egRef = 1.121;
      const eg = egRef * (1.0 - 0.0002677 * (T - Tref));
      const expTerm = Math.exp((egRef / (K_EV * Tref)) - (eg / (K_EV * T)));
      const io = M.i_o_ref_a * Math.pow(T / Tref, 3) * expTerm;
      const iphRaw = (M.i_ph_ref_a + M.k_i_a_per_k * (T - Tref)) * (G / M.g_ref_wm2);
      const iph = Math.max(0.0, iphRaw);
      const rSh = G > 1.0 ? rShRef * (M.g_ref_wm2 / G) : 1e8;

      function currentAtSubV(v) {
        if (v < 0.0) {
          const iBp = M.bypass_i_sat_a * (Math.exp(Math.min(45.0, -v / vtBp)) - 1.0);
          return iph - v / rSh + iBp;
        }
        let i = iph;
        for (let it = 0; it < 18; it++) {
          const arg = Math.min(60.0, (v + i * rS) / vtCell);
          const expVal = Math.exp(arg);
          const f = i - iph + io * (expVal - 1.0) + (v + i * rS) / rSh;
          const df = 1.0 + (io * rS / vtCell) * expVal + rS / rSh;
          const step = f / df;
          i -= step;
          if (Math.abs(step) < 1e-9) break;
        }
        return i;
      }

      function voltageAtCurrent(targetI) {
        let lo = -1.2;
        let hi = (M.v_oc_v / M.n_substrings) + 2.0;
        for (let it = 0; it < 34; it++) {
          const mid = 0.5 * (lo + hi);
          if (currentAtSubV(mid) > targetI) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
      }

      return { currentAtSubV, voltageAtCurrent, iscApprox: currentAtSubV(0.0) };
    }

    function solveString(gList, Tc) {
      const subs = gList.map(g => solveSubstringCurve(g, Tc));
      const iMax = Math.max(...subs.map(s => s.iscApprox));
      const nPts = 220;
      const pts = [];
      for (let k = nPts - 1; k >= 0; k--) {
        const iVal = (iMax * k) / (nPts - 1);
        let vSum = 0.0;
        for (const s of subs) vSum += s.voltageAtCurrent(iVal);
        if (vSum >= 0.0) pts.push({ v: vSum, i: iVal, p: vSum * iVal });
      }
      pts.sort((a, b) => a.v - b.v);
      if (pts.length === 0) pts.push({ v: 0, i: 0, p: 0 });

      const peaks = [];
      for (let k = 1; k < pts.length - 1; k++) {
        if (pts[k].p > pts[k - 1].p && pts[k].p >= pts[k + 1].p && pts[k].p > 1.0) {
          const prevPk = peaks[peaks.length - 1];
          if (peaks.length === 0 || Math.abs(pts[k].v - prevPk.v) > 1.8) {
            peaks.push({ ...pts[k] });
          } else if (pts[k].p > prevPk.p) {
            peaks[peaks.length - 1] = { ...pts[k] };
          }
        }
      }
      let gmpp = pts.reduce((best, cur) => (cur.p > best.p ? cur : best), pts[0]);
      if (peaks.length === 0) peaks.push(gmpp);
      gmpp = peaks.reduce((best, cur) => (cur.p > best.p ? cur : best), peaks[0]);

      let pUncoupled = 0.0;
      for (const g of gList) {
        const single = solveSubstringCurve(g, Tc);
        let pSubMax = 0.0;
        for (let m = 0; m <= 60; m++) {
          const vTest = (m / 60.0) * (M.v_oc_v / M.n_substrings + 1.0);
          const iTest = Math.max(0.0, single.currentAtSubV(vTest));
          pSubMax = Math.max(pSubMax, vTest * iTest);
        }
        pUncoupled += pSubMax;
      }
      const mismatch = pUncoupled > 0
        ? Math.max(0.0, 100.0 * (1.0 - gmpp.p / pUncoupled))
        : 0.0;
      return { pts, peaks, gmpp, mismatch };
    }

    function setupCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(300, rect.width);
      const h = Math.max(240, rect.height);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w, h };
    }

    function drawAxes(ctx, w, h, pad, xMax, yMin, yMax, title, xLbl, yLbl) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, w, h);
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px sans-serif";

      for (let k = 0; k <= 4; k++) {
        const y = pad.t + (ph * k) / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        const val = yMax - ((yMax - yMin) * k) / 4;
        ctx.textAlign = "right";
        ctx.fillText(val.toFixed(yMax - yMin < 12 ? 1 : 0), pad.l - 7, y + 4);
      }
      for (let k = 0; k <= 5; k++) {
        const x = pad.l + (pw * k) / 5;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
        const val = (xMax * k) / 5;
        ctx.textAlign = "center";
        ctx.fillText(val.toFixed(0), x, h - pad.b + 16);
      }

      ctx.strokeStyle = "#475569";
      ctx.strokeRect(pad.l, pad.t, pw, ph);

      ctx.fillStyle = "#f8fafc";
      ctx.font = "bold 12.5px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(title, pad.l, 20);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "11.5px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(xLbl, pad.l + pw / 2, h - 6);

      ctx.save();
      ctx.translate(14, pad.t + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLbl, 0, 0);
      ctx.restore();
      return { pw, ph };
    }

    function updateShadingLab() {
      const g1 = Number(document.getElementById("sl-g1").value);
      const g2 = Number(document.getElementById("sl-g2").value);
      const g3 = Number(document.getElementById("sl-g3").value);
      const tc = Number(document.getElementById("sl-tc").value);
      document.getElementById("val-g1").textContent = g1 + " W/m²";
      document.getElementById("val-g2").textContent = g2 + " W/m²";
      document.getElementById("val-g3").textContent = g3 + " W/m²";
      document.getElementById("val-tc").textContent = tc + " °C";

      const res = solveString([g1, g2, g3], tc);
      document.getElementById("kpi-gmpp-p").textContent = res.gmpp.p.toFixed(2) + " W";
      const viStr = res.gmpp.v.toFixed(1) + " V / " + res.gmpp.i.toFixed(2) + " A";
      document.getElementById("kpi-gmpp-vi").textContent = viStr;
      document.getElementById("kpi-peaks").textContent = String(res.peaks.length);
      document.getElementById("kpi-mismatch").textContent = res.mismatch.toFixed(1) + " %";

      const pad = { l: 48, r: 16, t: 32, b: 34 };
      const vMax = 43.0;
      const iMax = Math.max(11.0, res.pts[0].i * 1.1);
      const pMax = Math.max(320.0, res.gmpp.p * 1.15);

      const ivC = setupCanvas(document.getElementById("cv-iv"));
      const ivBox = drawAxes(
        ivC.ctx, ivC.w, ivC.h, pad, vMax, 0, iMax,
        "String Staircase I-V Curve", "String Voltage V (V)", "Current I (A)"
      );
      ivC.ctx.strokeStyle = "#38bdf8";
      ivC.ctx.lineWidth = 2.4;
      ivC.ctx.beginPath();
      res.pts.forEach((pt, idx) => {
        const x = pad.l + (pt.v / vMax) * ivBox.pw;
        const y = pad.t + (1.0 - pt.i / iMax) * ivBox.ph;
        if (idx === 0) ivC.ctx.moveTo(x, y); else ivC.ctx.lineTo(x, y);
      });
      ivC.ctx.stroke();

      const gxIv = pad.l + (res.gmpp.v / vMax) * ivBox.pw;
      const gyIv = pad.t + (1.0 - res.gmpp.i / iMax) * ivBox.ph;
      ivC.ctx.fillStyle = "#f59e0b";
      ivC.ctx.beginPath(); ivC.ctx.arc(gxIv, gyIv, 5.5, 0, Math.PI * 2); ivC.ctx.fill();

      const pvC = setupCanvas(document.getElementById("cv-pv"));
      const pvBox = drawAxes(
        pvC.ctx, pvC.w, pvC.h, pad, vMax, 0, pMax,
        "Multi-Peak P-V Curve (GMPP vs. LMPP Traps)", "String Voltage V (V)", "Power P (W)"
      );
      pvC.ctx.strokeStyle = "#22c55e";
      pvC.ctx.lineWidth = 2.4;
      pvC.ctx.beginPath();
      res.pts.forEach((pt, idx) => {
        const x = pad.l + (pt.v / vMax) * pvBox.pw;
        const y = pad.t + (1.0 - pt.p / pMax) * pvBox.ph;
        if (idx === 0) pvC.ctx.moveTo(x, y); else pvC.ctx.lineTo(x, y);
      });
      pvC.ctx.stroke();

      res.peaks.forEach(pk => {
        const x = pad.l + (pk.v / vMax) * pvBox.pw;
        const y = pad.t + (1.0 - pk.p / pMax) * pvBox.ph;
        const isG = Math.abs(pk.v - res.gmpp.v) < 0.2;
        pvC.ctx.fillStyle = isG ? "#f59e0b" : "#ef4444";
        pvC.ctx.beginPath();
        pvC.ctx.arc(x, y, isG ? 6.5 : 4.8, 0, Math.PI * 2);
        pvC.ctx.fill();
        pvC.ctx.fillStyle = "#f8fafc";
        pvC.ctx.font = "bold 10.5px sans-serif";
        pvC.ctx.textAlign = "center";
        const tag = (isG ? "GMPP " : "LMPP ") + pk.p.toFixed(0) + "W";
        pvC.ctx.fillText(tag, x, Math.max(pad.t + 12, y - 9));
      });
    }

    function drawGMPPT() {
      const G = DATA.gmppt;
      const checkedEls = document.querySelectorAll(".gmppt-chk:checked");
      const enabled = Array.from(checkedEls).map(el => el.value);
      const colors = {
        "Conventional P&O": "#ef4444",
        "Conventional INC": "#a855f7",
        "PSO Global MPPT": "#38bdf8",
        "Hybrid Scan + Adaptive INC": "#22c55e"
      };
      const pad = { l: 50, r: 16, t: 32, b: 36 };
      const xMax = G.steps.length - 1;

      const cP = setupCanvas(document.getElementById("cv-gmppt-p"));
      const bP = drawAxes(
        cP.ctx, cP.w, cP.h, pad, xMax, 0, 320,
        "3-Phase Dynamic Power Extraction P(k)", "Control Step k", "Power (W)"
      );
      cP.ctx.setLineDash([5, 4]);
      cP.ctx.strokeStyle = "#f8fafc";
      cP.ctx.lineWidth = 1.8;
      cP.ctx.beginPath();
      G.true_gmpp_power_w.forEach((p, idx) => {
        const x = pad.l + (idx / xMax) * bP.pw;
        const y = pad.t + (1 - p / 320) * bP.ph;
        if (idx === 0) cP.ctx.moveTo(x, y); else cP.ctx.lineTo(x, y);
      });
      cP.ctx.stroke();
      cP.ctx.setLineDash([]);

      enabled.forEach(name => {
        const series = G.controllers[name].powers_w;
        cP.ctx.strokeStyle = colors[name] || "#38bdf8";
        cP.ctx.lineWidth = 2.0;
        cP.ctx.beginPath();
        series.forEach((p, idx) => {
          const x = pad.l + (idx / xMax) * bP.pw;
          const y = pad.t + (1 - p / 320) * bP.ph;
          if (idx === 0) cP.ctx.moveTo(x, y); else cP.ctx.lineTo(x, y);
        });
        cP.ctx.stroke();
      });

      const cV = setupCanvas(document.getElementById("cv-gmppt-v"));
      const bV = drawAxes(
        cV.ctx, cV.w, cV.h, pad, xMax, 0, 42,
        "Operating Voltage Trajectory V(k) vs. GMPP Basin", "Control Step k", "Voltage (V)"
      );
      cV.ctx.setLineDash([5, 4]);
      cV.ctx.strokeStyle = "#f8fafc";
      cV.ctx.lineWidth = 1.8;
      cV.ctx.beginPath();
      G.true_gmpp_voltage_v.forEach((v, idx) => {
        const x = pad.l + (idx / xMax) * bV.pw;
        const y = pad.t + (1 - v / 42) * bV.ph;
        if (idx === 0) cV.ctx.moveTo(x, y); else cV.ctx.lineTo(x, y);
      });
      cV.ctx.stroke();
      cV.ctx.setLineDash([]);

      enabled.forEach(name => {
        const series = G.controllers[name].voltages_v;
        cV.ctx.strokeStyle = colors[name] || "#38bdf8";
        cV.ctx.lineWidth = 2.0;
        cV.ctx.beginPath();
        series.forEach((v, idx) => {
          const x = pad.l + (idx / xMax) * bV.pw;
          const y = pad.t + (1 - v / 42) * bV.ph;
          if (idx === 0) cV.ctx.moveTo(x, y); else cV.ctx.lineTo(x, y);
        });
        cV.ctx.stroke();
      });
    }

    let enMode = "power";
    function drawEN50530() {
      const E = DATA.en50530;
      const t = E.time_s;
      const tMax = t[t.length - 1];
      const pad = { l: 54, r: 20, t: 34, b: 36 };
      const c = setupCanvas(document.getElementById("cv-en50530"));

      if (enMode === "power") {
        const box = drawAxes(
          c.ctx, c.w, c.h, pad, tMax, 0, 320,
          "CENELEC EN 50530 Dynamic Power Tracking (Sera dP-P&O 99.94% vs. P&O 87.18%)",
          "Time t (s)", "Power (W)"
        );
        const plotLine = (arr, col, dash) => {
          c.ctx.setLineDash(dash);
          c.ctx.strokeStyle = col;
          c.ctx.lineWidth = 2.1;
          c.ctx.beginPath();
          arr.forEach((v, i) => {
            const x = pad.l + (t[i] / tMax) * box.pw;
            const y = pad.t + (1 - v / 320) * box.ph;
            if (i === 0) c.ctx.moveTo(x, y); else c.ctx.lineTo(x, y);
          });
          c.ctx.stroke();
          c.ctx.setLineDash([]);
        };
        plotLine(E.ideal_mpp_power_w, "#f8fafc", [5, 4]);
        plotLine(E.controllers["Standard P&O"].powers_w, "#ef4444", []);
        plotLine(E.controllers["Incremental Conductance"].powers_w, "#a855f7", []);
        plotLine(E.controllers["Sera dP-P&O (Drift-Free)"].powers_w, "#22c55e", []);
      } else if (enMode === "voltage") {
        const box = drawAxes(
          c.ctx, c.w, c.h, pad, tMax, 15, 38,
          "Ramp-Induced Voltage Drift V(t) vs. Thermally Shifting V_mpp(t)",
          "Time t (s)", "Voltage (V)"
        );
        const plotLine = (arr, col, dash) => {
          c.ctx.setLineDash(dash);
          c.ctx.strokeStyle = col;
          c.ctx.lineWidth = 2.1;
          c.ctx.beginPath();
          arr.forEach((v, i) => {
            const x = pad.l + (t[i] / tMax) * box.pw;
            const y = pad.t + (1 - (v - 15) / 23) * box.ph;
            if (i === 0) c.ctx.moveTo(x, y); else c.ctx.lineTo(x, y);
          });
          c.ctx.stroke();
          c.ctx.setLineDash([]);
        };
        plotLine(E.ideal_mpp_voltage_v, "#f8fafc", [5, 4]);
        plotLine(E.controllers["Standard P&O"].voltages_v, "#ef4444", []);
        plotLine(E.controllers["Incremental Conductance"].voltages_v, "#a855f7", []);
        plotLine(E.controllers["Sera dP-P&O (Drift-Free)"].voltages_v, "#22c55e", []);
      } else {
        const box = drawAxes(
          c.ctx, c.w, c.h, pad, tMax, 24, 41,
          "Coupled Lumped Electro-Thermal Cell Temperature ODE (MPPT-Cooled vs. Open-Circuit)",
          "Time t (s)", "Cell Temp (°C)"
        );
        const plotLine = (arr, col, dash) => {
          c.ctx.setLineDash(dash);
          c.ctx.strokeStyle = col;
          c.ctx.lineWidth = 2.3;
          c.ctx.beginPath();
          arr.forEach((v, i) => {
            const x = pad.l + (t[i] / tMax) * box.pw;
            const y = pad.t + (1 - (v - 24) / 17) * box.ph;
            if (i === 0) c.ctx.moveTo(x, y); else c.ctx.lineTo(x, y);
          });
          c.ctx.stroke();
          c.ctx.setLineDash([]);
        };
        plotLine(E.open_circuit_cell_temp_c, "#ef4444", [6, 4]);
        plotLine(E.ideal_cell_temp_c, "#38bdf8", []);
      }
    }

    let activeBusKey = "bus_60v";
    function drawBoost() {
      const B = DATA.boost;
      const curve = B[activeBusKey];
      const rawIdx = Number(document.getElementById("sl-boost-idx").value);
      const idx = Math.min(curve.load_pct.length - 1, rawIdx);
      const loadLabel =
        curve.load_pct[idx].toFixed(0) + "% STC (" + curve.p_in_w[idx].toFixed(0) + " W)";
      document.getElementById("val-boost-load").textContent = loadLabel;
      const etaLabel = curve.efficiency_pct[idx].toFixed(2) + " %";
      document.getElementById("kpi-boost-eta").textContent = etaLabel;

      const parts = [
        ["Inductor Copper (I_rms² R_L)", curve.losses_w.inductor_copper[idx], "#38bdf8"],
        ["Freewheeling Diode", curve.losses_w.diode_conduction[idx], "#f59e0b"],
        ["MOSFET Conduction", curve.losses_w.mosfet_conduction[idx], "#22c55e"],
        ["MOSFET Sw + C_oss + Q_rr", curve.losses_w.mosfet_switching[idx], "#ef4444"],
        ["Magnetic Core Ripple", curve.losses_w.magnetic_core[idx], "#a855f7"]
      ];
      const totLoss = parts.reduce((acc, p) => acc + p[1], 0);
      document.getElementById("kpi-boost-loss").textContent = totLoss.toFixed(2) + " W";

      const barsEl = document.getElementById("boost-loss-bars");
      barsEl.innerHTML = parts.map(([lbl, val, col]) => {
        const pct = totLoss > 0 ? Math.min(100, (100 * val) / totLoss) : 0;
        return (
          '<div class="loss-row"><span>' + lbl + '</span>' +
          '<div class="bar-track"><div class="bar-fill" style="width:' +
          pct.toFixed(1) + '%;background:' + col + '"></div></div>' +
          '<span style="text-align:right;font-family:monospace">' +
          val.toFixed(2) + ' W</span></div>'
        );
      }).join("");

      const pad = { l: 50, r: 16, t: 32, b: 36 };
      const cEff = setupCanvas(document.getElementById("cv-boost-eff"));
      const bEff = drawAxes(
        cEff.ctx, cEff.w, cEff.h, pad, 100, 91.5, 99.0,
        "Stage Efficiency Curve (60 V LVDC vs. 120 V HVDC)", "Load (% STC)", "Efficiency (%)"
      );
      [["bus_60v", "#38bdf8"], ["bus_120v", "#f59e0b"]].forEach(([key, col]) => {
        const cv = B[key];
        cEff.ctx.strokeStyle = col;
        cEff.ctx.lineWidth = key === activeBusKey ? 2.6 : 1.7;
        cEff.ctx.beginPath();
        cv.load_pct.forEach((lp, i) => {
          const x = pad.l + (lp / 100) * bEff.pw;
          const y = pad.t + (1 - (cv.efficiency_pct[i] - 91.5) / 7.5) * bEff.ph;
          if (i === 0) cEff.ctx.moveTo(x, y); else cEff.ctx.lineTo(x, y);
        });
        cEff.ctx.stroke();
      });
      const selX = pad.l + (curve.load_pct[idx] / 100) * bEff.pw;
      const selY = pad.t + (1 - (curve.efficiency_pct[idx] - 91.5) / 7.5) * bEff.ph;
      cEff.ctx.fillStyle = "#ffffff";
      cEff.ctx.beginPath(); cEff.ctx.arc(selX, selY, 5.5, 0, Math.PI * 2); cEff.ctx.fill();

      const T = B.transient;
      const tMax = T.time_ms[T.time_ms.length - 1];
      const cTr = setupCanvas(document.getElementById("cv-boost-servo"));
      const bTr = drawAxes(
        cTr.ctx, cTr.w, cTr.h, pad, tMax, 28.5, 34.5,
        "Closed-Loop Cascaded PI Servo (v_pv vs. v_ref vs. V_mpp)",
        "Time (ms)", "Voltage (V)"
      );
      const drawTr = (arr, col, dash) => {
        cTr.ctx.setLineDash(dash);
        cTr.ctx.strokeStyle = col;
        cTr.ctx.lineWidth = 2.0;
        cTr.ctx.beginPath();
        arr.forEach((v, i) => {
          const x = pad.l + (T.time_ms[i] / tMax) * bTr.pw;
          const y = pad.t + (1 - (v - 28.5) / 6.0) * bTr.ph;
          if (i === 0) cTr.ctx.moveTo(x, y); else cTr.ctx.lineTo(x, y);
        });
        cTr.ctx.stroke();
        cTr.ctx.setLineDash([]);
      };
      drawTr(T.v_mpp_true_v, "#22c55e", [4, 4]);
      drawTr(T.v_ref_v, "#f59e0b", [6, 3]);
      drawTr(T.v_pv_v, "#38bdf8", []);
    }

    function redrawActive() {
      const activeId = document.querySelector(".tab-pane.active").id;
      if (activeId === "tab-shading") updateShadingLab();
      else if (activeId === "tab-gmppt") drawGMPPT();
      else if (activeId === "tab-en50530") drawEN50530();
      else if (activeId === "tab-boost") drawBoost();
    }

    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
        redrawActive();
      });
    });

    ["sl-g1", "sl-g2", "sl-g3", "sl-tc"].forEach(id => {
      document.getElementById(id).addEventListener("input", updateShadingLab);
    });
    document.querySelectorAll("[data-preset]").forEach(btn => {
      btn.addEventListener("click", () => {
        const [g1, g2, g3, tc] = btn.dataset.preset.split(",");
        document.getElementById("sl-g1").value = g1;
        document.getElementById("sl-g2").value = g2;
        document.getElementById("sl-g3").value = g3;
        document.getElementById("sl-tc").value = tc;
        updateShadingLab();
      });
    });
    document.querySelectorAll(".gmppt-chk").forEach(chk => {
      chk.addEventListener("change", drawGMPPT);
    });
    document.querySelectorAll(".en-mode-btn").forEach(btn => {
      btn.addEventListener("click", () => { enMode = btn.dataset.mode; drawEN50530(); });
    });
    document.querySelectorAll(".bus-btn").forEach(btn => {
      btn.addEventListener("click", () => { activeBusKey = btn.dataset.bus; drawBoost(); });
    });
    document.getElementById("sl-boost-idx").addEventListener("input", drawBoost);

    const modal = document.getElementById("img-modal");
    const modalImg = document.getElementById("modal-img");
    document.querySelectorAll(".fig-card img").forEach(img => {
      img.addEventListener("click", () => {
        modalImg.src = img.src;
        modal.classList.add("open");
      });
    });
    modal.addEventListener("click", () => modal.classList.remove("open"));

    window.addEventListener("resize", redrawActive);
    redrawActive();
  </script>
</body>
</html>
"""


def _render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    mod = payload["module_params"]
    sera_eff = payload["en50530"]["controllers"]["Sera dP-P&O (Drift-Free)"][
        "en50530_eff_pct"
    ]
    hyb_eff = payload["gmppt"]["controllers"]["Hybrid Scan + Adaptive INC"][
        "dynamic_eff_pct"
    ]
    pso_eff = payload["gmppt"]["controllers"]["PSO Global MPPT"]["settled_eff_pct"]
    euro_60 = payload["boost"]["bus_60v"]["euro_efficiency_pct"]
    servo_mv = payload["boost"]["transient"]["rms_servo_mv"]

    replacements = {
        "__VERSION__": str(payload["version"]),
        "__MOD_NAME__": str(mod["name"]),
        "__MOD_DB__": str(mod["database"]),
        "__MOD_VER__": str(mod["record_version"]),
        "__STC_P__": f"{mod['stc_p_max_w']:.2f}",
        "__V_MP__": f"{mod['v_mp_v']:.2f}",
        "__I_MP__": f"{mod['i_mp_a']:.2f}",
        "__HYB_EFF__": f"{hyb_eff:.2f}",
        "__PSO_EFF__": f"{pso_eff:.2f}",
        "__SERA_EFF__": f"{sera_eff:.2f}",
        "__EURO_60__": f"{euro_60:.2f}",
        "__SERVO_MV__": f"{servo_mv:.1f}",
        "__DATA_JSON__": data_json,
    }
    html = _HTML_TEMPLATE
    for token, val in replacements.items():
        html = html.replace(token, val)
    return html


def build_static_site(
    output_dir: str | Path = "site",
    *,
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Generate the GitHub Pages static site bundle in ``output_dir``."""
    out_path = Path(output_dir)
    res_path = Path(results_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = export_benchmark_payload()
    copied_assets = _ensure_figures(res_path, out_path / "assets")
    payload["assets"] = copied_assets

    json_path = out_path / "benchmark_data.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_path = out_path / "index.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")

    nojekyll_path = out_path / ".nojekyll"
    nojekyll_path.write_text("", encoding="utf-8")

    return {
        "output_dir": str(out_path),
        "index_html": str(html_path),
        "benchmark_json": str(json_path),
        "assets": copied_assets,
    }
