"""Command-line interface (`pv-mppt`) for Solar PV, GMPPT, EN 50530 & Boost Analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src import __version__
from src.dynamic_boost import (
    BoostPowerStageParams,
    evaluate_converter_efficiency_curve,
    run_closed_loop_boost_transient,
)
from src.en50530_benchmark import LumpedThermalModel, run_en50530_dynamic_benchmark
from src.global_mppt import run_dynamic_gmppt_benchmark
from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.partial_shading import ShadedPVString, get_benchmark_shading_profiles
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
from src.web_dashboard import build_static_site

__all__ = ["build_parser", "main"]


def _parse_stage_spec(spec: str) -> tuple[float, float]:
    """Parse ``G_WM2:DURATION_MS`` (e.g. ``1000:45`` -> ``(1000.0, 0.045)``)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid stage '{spec}'. Expected G_WM2:DURATION_MS (e.g. 1000:45)."
        )
    g_raw, ms_raw = spec.split(":", 1)
    try:
        g_val = float(g_raw)
        dur_ms = float(ms_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Non-numeric stage '{spec}'. Expected G_WM2:DURATION_MS."
        ) from exc
    if not np.isfinite(g_val) or not np.isfinite(dur_ms) or g_val <= 0.0 or dur_ms <= 0.0:
        raise argparse.ArgumentTypeError(
            f"Stage irradiance and duration_ms must be positive finite numbers: '{spec}'."
        )
    return g_val, dur_ms * 1e-3


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``pv-mppt`` argument parser with engineering subcommands."""
    parser = argparse.ArgumentParser(
        prog="pv-mppt",
        description=(
            "Solar PV Dynamic Characterization, Bypass-Diode Partial Shading, "
            "Global MPPT, CENELEC EN 50530 & DC-DC Boost Converter CLI."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # 1. info
    p_info = sub.add_parser(
        "info",
        help="Show pinned CEC module provenance and extracted single-diode parameters.",
    )
    p_info.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 2. iv
    p_iv = sub.add_parser(
        "iv",
        help="Solve single-diode I-V / P-V curve at arbitrary (G, T_c).",
    )
    p_iv.add_argument(
        "-g",
        "--irradiance",
        type=float,
        default=1000.0,
        help="Plane-of-array irradiance G in W/m^2 (default: 1000.0).",
    )
    p_iv.add_argument(
        "-t",
        "--temp",
        type=float,
        default=25.0,
        help="Cell temperature T_c in degC within [-40, 85] (default: 25.0).",
    )
    p_iv.add_argument(
        "--points",
        type=int,
        default=400,
        help="Voltage grid resolution (default: 400).",
    )
    p_iv.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PNG path to render multi-irradiance PV curves.",
    )
    p_iv.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 3. shading
    p_sh = sub.add_parser(
        "shading",
        help="Evaluate bypass-diode partial-shading multi-peak P-V curve & optional GMPPT.",
    )
    p_sh.add_argument(
        "--irradiances",
        type=float,
        nargs="+",
        default=[1000.0, 700.0, 350.0],
        help="Per-substring irradiances in W/m^2 (default: 1000 700 350).",
    )
    p_sh.add_argument(
        "-t",
        "--temp",
        type=float,
        default=25.0,
        help="Uniform substring cell temperature in degC (default: 25.0).",
    )
    p_sh.add_argument(
        "--modules",
        type=int,
        default=1,
        help="Number of series-connected modules (default: 1).",
    )
    p_sh.add_argument(
        "--bypass",
        type=int,
        default=3,
        help="Bypass diodes per module (default: 3).",
    )
    p_sh.add_argument(
        "--run-gmppt",
        action="store_true",
        help="Also run the 3-phase dynamic GMPPT benchmark (P&O, INC, PSO, Hybrid).",
    )
    p_sh.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PNG path for partial-shading P-V curve figure.",
    )
    p_sh.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 4. en50530
    p_en = sub.add_parser(
        "en50530",
        help="Run CENELEC EN 50530 trapezoidal irradiance ramp & electro-thermal benchmark.",
    )
    p_en.add_argument(
        "--step-size",
        type=float,
        default=0.22,
        help="Controller voltage perturbation step in Volts (default: 0.22).",
    )
    p_en.add_argument(
        "--dt",
        type=float,
        default=0.2,
        help="MPPT sampling interval in seconds (default: 0.2).",
    )
    p_en.add_argument(
        "--t-amb",
        type=float,
        default=25.0,
        help="Ambient temperature in degC (default: 25.0).",
    )
    p_en.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PNG path for the 3-panel EN 50530 figure.",
    )
    p_en.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 5. boost
    p_bo = sub.add_parser(
        "boost",
        help="Analyze state-space boost stage, cascaded PI servo & semiconductor/magnetic losses.",
    )
    p_bo.add_argument(
        "--v-bus",
        type=float,
        default=60.0,
        help="DC-link bus voltage V_bus in Volts (default: 60.0).",
    )
    p_bo.add_argument(
        "--f-sw-khz",
        type=float,
        default=50.0,
        help="PWM switching frequency in kHz (default: 50.0).",
    )
    p_bo.add_argument(
        "--stages",
        type=_parse_stage_spec,
        nargs="+",
        default=None,
        help="Irradiance step stages as G_WM2:DURATION_MS (default: 1000:45 550:45 900:45).",
    )
    p_bo.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PNG path for the 2x2 boost converter dashboard.",
    )
    p_bo.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 6. benchmark
    p_bm = sub.add_parser(
        "benchmark",
        help="Execute full multi-physics benchmark suite and export all 7 figures.",
    )
    p_bm.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="results",
        help="Directory for generated PNG figures (default: results).",
    )
    p_bm.add_argument("--json", action="store_true", help="Emit JSON summary.")

    # 7. web
    p_web = sub.add_parser(
        "web",
        help="Build static interactive GitHub Pages explorer bundle (site/).",
    )
    p_web.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="site",
        help="Target directory for index.html, benchmark_data.json & assets/ (default: site).",
    )
    p_web.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Source directory of 300-DPI PNG figures (default: results).",
    )
    p_web.add_argument("--json", action="store_true", help="Emit JSON manifest.")

    return parser


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _cmd_info(args: argparse.Namespace) -> int:
    pv = SolarPVModel()
    ref = pv.REFERENCE_MODULE
    _, _, p_stc = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=500)
    stc_pmax = float(np.max(p_stc))

    payload: dict[str, Any] = {
        "package": "pv-mppt-solver",
        "version": __version__,
        "reference_module": {
            "name": ref["name"],
            "database": ref["database"],
            "record_version": ref["record_version"],
            "n_cells": pv.N_s,
            "i_sc_a": pv.I_sc,
            "v_oc_v": pv.V_oc,
            "i_mp_a": pv.I_mp,
            "v_mp_v": pv.V_mp,
            "r_s_ohm": pv.R_s,
            "r_sh_ref_ohm": pv.R_sh,
            "dataset_sha256": ref["dataset_sha256"],
            "stc_p_max_w": round(stc_pmax, 4),
        },
        "extracted_single_diode": {
            "ideality_factor_n": round(float(pv.n), 6),
            "modified_ideality_a_ref_v": round(float(pv.a_ref), 6),
            "i_ph_ref_a": round(float(pv.I_ph_ref), 6),
            "i_o_ref_a": float(f"{pv.I_o_ref:.6e}"),
        },
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    print(f"pv-mppt-solver v{__version__}, Reference PV Module & Diode Extraction")
    print("=" * 72)
    print(f"Module:           {ref['name']} ({pv.N_s} cells)")
    print(f"Database:         {ref['database']} ({ref['record_version']})")
    print(
        f"Datasheet STC:    V_oc = {pv.V_oc:.2f} V | I_sc = {pv.I_sc:.2f} A | "
        f"V_mp = {pv.V_mp:.2f} V | I_mp = {pv.I_mp:.2f} A"
    )
    print(
        f"Extracted SDM:    n = {pv.n:.6f} | I_ph,ref = {pv.I_ph_ref:.6f} A | "
        f"I_0,ref = {pv.I_o_ref:.6e} A"
    )
    print(f"Verified STC MPP: P_max = {stc_pmax:.4f} W")
    print("=" * 72)
    return 0


def _cmd_iv(args: argparse.Namespace) -> int:
    pv = SolarPVModel()
    v_arr, i_arr, p_arr = pv.calculate_iv_curve(
        G=args.irradiance, T_c_celsius=args.temp, num_points=args.points
    )
    mpp_idx = int(np.argmax(p_arr))
    i_sc = float(i_arr[0])
    nz_idx = np.where(i_arr > 1e-6)[0]
    v_oc = float(v_arr[nz_idx[-1] + 1]) if (
        len(nz_idx) > 0 and nz_idx[-1] + 1 < len(v_arr)
    ) else (float(v_arr[nz_idx[-1]]) if len(nz_idx) > 0 else 0.0)
    v_mpp = float(v_arr[mpp_idx])
    i_mpp = float(i_arr[mpp_idx])
    p_mpp = float(p_arr[mpp_idx])
    denom = v_oc * i_sc
    ff_pct = (100.0 * p_mpp / denom) if denom > 0.0 else 0.0
    stc_p = pv.V_mp * pv.I_mp
    rel_stc_pct = 100.0 * (p_mpp - stc_p) / stc_p

    if args.plot:
        plot_pv_characteristics(pv, save_path=args.plot)

    payload: dict[str, Any] = {
        "irradiance_wm2": args.irradiance,
        "temp_c": args.temp,
        "i_sc_a": round(i_sc, 4),
        "v_oc_v": round(v_oc, 4),
        "v_mpp_v": round(v_mpp, 4),
        "i_mpp_a": round(i_mpp, 4),
        "p_mpp_w": round(p_mpp, 4),
        "fill_factor_pct": round(ff_pct, 2),
        "delta_vs_stc_pct": round(rel_stc_pct, 2),
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    print(
        f"Single-Diode Operating Point  [G = {args.irradiance:.1f} W/m^2, "
        f"T_c = {args.temp:.1f} degC]"
    )
    print("-" * 72)
    print(f"  Short-Circuit Current (I_sc):   {i_sc:8.4f} A")
    print(f"  Open-Circuit Voltage  (V_oc):   {v_oc:8.4f} V")
    print(f"  MPP Voltage           (V_mpp):  {v_mpp:8.4f} V")
    print(f"  MPP Current           (I_mpp):  {i_mpp:8.4f} A")
    print(f"  Maximum Power         (P_mpp):  {p_mpp:8.4f} W ({rel_stc_pct:+.2f}% vs STC)")
    print(f"  Fill Factor           (FF):     {ff_pct:8.2f} %")
    if args.plot:
        print(f"  Saved Plot:                     {args.plot}")
    print("=" * 72)
    return 0


def _cmd_shading(args: argparse.Namespace) -> int:
    pv = SolarPVModel()
    total_substrings = args.modules * args.bypass
    if len(args.irradiances) != total_substrings:
        raise ValueError(
            f"Expected {total_substrings} substring irradiances "
            f"(modules={args.modules} * bypass={args.bypass}), "
            f"got {len(args.irradiances)}."
        )
    string = ShadedPVString(
        pv, n_modules=args.modules, n_bypass_per_module=args.bypass
    )
    res = string.solve_curve(
        args.irradiances, temperatures_c=args.temp, num_points=600
    )

    if args.plot:
        label = "Custom Shading (" + " / ".join(f"{g:.0f}" for g in args.irradiances) + " W/m²)"
        plot_partial_shading_curves({label: res}, save_path=args.plot)

    gmppt_summary: dict[str, Any] | None = None
    if args.run_gmppt:
        bench = run_dynamic_gmppt_benchmark(string)
        gmppt_summary = {
            name: {
                "dynamic_efficiency_pct": round(t.tracking_efficiency_pct, 2),
                "settled_gmpp_efficiency_pct": round(t.settled_efficiency_pct, 2),
            }
            for name, t in bench.trajectories.items()
        }

    payload: dict[str, Any] = {
        "irradiances_wm2": [float(g) for g in args.irradiances],
        "temp_c": args.temp,
        "num_peaks": res.num_peaks,
        "gmpp": {
            "voltage_v": round(res.gmpp.voltage_v, 4),
            "current_a": round(res.gmpp.current_a, 4),
            "power_w": round(res.gmpp.power_w, 4),
        },
        "mismatch_loss_pct": round(res.mismatch_loss_pct, 2),
        "peaks": [
            {
                "voltage_v": round(pk.voltage_v, 4),
                "current_a": round(pk.current_a, 4),
                "power_w": round(pk.power_w, 4),
                "is_global": pk.is_global,
            }
            for pk in res.peaks
        ],
        "gmppt_benchmark": gmppt_summary,
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    irr_str = " / ".join(f"{g:.0f}" for g in args.irradiances)
    print(f"Series-String Bypass-Diode Shading  [{irr_str} W/m^2 @ {args.temp:.1f} degC]")
    print("-" * 72)
    print(
        f"  Detected Peaks: {res.num_peaks} | GMPP: {res.gmpp.power_w:.2f} W @ "
        f"{res.gmpp.voltage_v:.2f} V ({res.gmpp.current_a:.2f} A) | "
        f"Mismatch Loss: {res.mismatch_loss_pct:.2f}%"
    )
    for idx, pk in enumerate(res.peaks, start=1):
        tag = "GMPP" if pk.is_global else "LMPP"
        print(
            f"    Peak #{idx} [{tag}]: P = {pk.power_w:6.2f} W | "
            f"V = {pk.voltage_v:6.2f} V | I = {pk.current_a:5.2f} A"
        )
    if gmppt_summary is not None:
        print("-" * 72)
        print("  3-Phase Dynamic GMPPT Benchmark:")
        for name, metrics in gmppt_summary.items():
            print(
                f"    {name:<28} -> Dyn = {metrics['dynamic_efficiency_pct']:5.2f}% | "
                f"Settled = {metrics['settled_gmpp_efficiency_pct']:5.2f}%"
            )
    if args.plot:
        print(f"  Saved Plot: {args.plot}")
    print("=" * 72)
    return 0


def _cmd_en50530(args: argparse.Namespace) -> int:
    pv = SolarPVModel()
    thermal = LumpedThermalModel(t_ambient_c=args.t_amb)
    res = run_en50530_dynamic_benchmark(
        pv,
        thermal,
        dt_s=args.dt,
        step_size_v=args.step_size,
    )
    if args.plot:
        plot_en50530_benchmark(res, save_path=args.plot)

    payload: dict[str, Any] = {
        "dt_s": args.dt,
        "step_size_v": args.step_size,
        "t_amb_c": args.t_amb,
        "duration_s": round(float(res.time_s[-1]), 2),
        "peak_mpp_cell_temp_c": round(float(np.max(res.ideal_cell_temp_c)), 2),
        "peak_open_circuit_cell_temp_c": round(
            float(np.max(res.open_circuit_cell_temp_c)), 2
        ),
        "controllers": {
            name: {
                "en50530_efficiency_pct": round(c.en50530_efficiency_pct, 3),
                "rms_voltage_error_v": round(c.rms_voltage_error_v, 3),
                "peak_voltage_excursion_v": round(c.peak_voltage_excursion_v, 3),
            }
            for name, c in res.controllers.items()
        },
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    print(
        f"CENELEC EN 50530 Dynamic Ramp Benchmark  "
        f"[dt = {args.dt:.3f} s, dV = {args.step_size:.2f} V, "
        f"T_cell,max = {np.max(res.ideal_cell_temp_c):.2f} degC]"
    )
    print("-" * 72)
    for name, c in res.controllers.items():
        print(
            f"  {name:<28} -> eta_EN50530 = {c.en50530_efficiency_pct:6.2f}% | "
            f"RMS dV = {c.rms_voltage_error_v:5.2f} V | "
            f"Peak |dV| = {c.peak_voltage_excursion_v:5.2f} V"
        )
    if args.plot:
        print(f"  Saved Plot: {args.plot}")
    print("=" * 72)
    return 0


def _cmd_boost(args: argparse.Namespace) -> int:
    if args.f_sw_khz <= 0.0 or not np.isfinite(args.f_sw_khz):
        raise ValueError(f"f_sw_khz must be positive and finite, got {args.f_sw_khz}")

    pv = SolarPVModel()
    params = BoostPowerStageParams(
        v_bus_v=args.v_bus,
        f_sw_hz=args.f_sw_khz * 1e3,
    )
    eff_curve = evaluate_converter_efficiency_curve(pv, params)
    tr = run_closed_loop_boost_transient(pv, params, stages=args.stages)

    if args.plot:
        comp_v = 120.0 if abs(args.v_bus - 120.0) > 1e-3 else 60.0
        comp_curve = evaluate_converter_efficiency_curve(
            pv, BoostPowerStageParams(v_bus_v=comp_v, f_sw_hz=args.f_sw_khz * 1e3)
        )
        c_low, c_high = (
            (eff_curve, comp_curve) if args.v_bus <= comp_v else (comp_curve, eff_curve)
        )
        plot_dynamic_boost_converter(tr, c_low, c_high, save_path=args.plot)

    bd_stc = eff_curve.breakdowns[-1]
    payload: dict[str, Any] = {
        "v_bus_v": args.v_bus,
        "f_sw_khz": args.f_sw_khz,
        "peak_efficiency_pct": round(eff_curve.peak_efficiency_pct, 2),
        "euro_efficiency_pct": round(eff_curve.euro_efficiency_pct, 2),
        "cec_efficiency_pct": round(eff_curve.cec_efficiency_pct, 2),
        "stc_loss_breakdown_w": {
            "p_in_w": round(bd_stc.p_in_w, 2),
            "p_out_w": round(bd_stc.p_out_w, 2),
            "total_loss_w": round(bd_stc.total_loss_w, 3),
            "inductor_copper_w": round(bd_stc.p_inductor_copper_w, 3),
            "diode_conduction_w": round(bd_stc.p_diode_cond_w, 3),
            "mosfet_conduction_w": round(bd_stc.p_mosfet_cond_w, 3),
            "mosfet_switching_w": round(bd_stc.p_mosfet_sw_w, 3),
            "magnetic_core_w": round(bd_stc.p_core_w, 3),
        },
        "closed_loop_transient": {
            "avg_converter_efficiency_pct": round(tr.avg_converter_efficiency_pct, 2),
            "rms_servo_tracking_error_mv": round(tr.rms_servo_tracking_error_v * 1e3, 2),
            "v_pv_final_v": round(float(tr.v_pv_v[-1]), 3),
            "v_mpp_final_v": round(float(tr.v_mpp_true_v[-1]), 3),
        },
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    print(
        f"State-Space DC-DC Boost Stage & Cascaded PI Servo  "
        f"[V_bus = {args.v_bus:.1f} V, f_sw = {args.f_sw_khz:.1f} kHz]"
    )
    print("-" * 72)
    print(
        f"  Weighted Ratings:  Peak = {eff_curve.peak_efficiency_pct:.2f}% | "
        f"eta_Euro = {eff_curve.euro_efficiency_pct:.2f}% | "
        f"eta_CEC = {eff_curve.cec_efficiency_pct:.2f}%"
    )
    print(
        f"  STC Loss Map:      Total = {bd_stc.total_loss_w:.2f} W "
        f"(Cu {bd_stc.p_inductor_copper_w:.2f} W | Diode {bd_stc.p_diode_cond_w:.2f} W | "
        f"MOS_cond {bd_stc.p_mosfet_cond_w:.2f} W | MOS_sw {bd_stc.p_mosfet_sw_w:.2f} W | "
        f"Core {bd_stc.p_core_w:.2f} W)"
    )
    print(
        f"  Closed-Loop Servo: Avg Eff = {tr.avg_converter_efficiency_pct:.2f}% | "
        f"RMS Servo Error = {tr.rms_servo_tracking_error_v * 1e3:.1f} mV | "
        f"Final V_pv = {tr.v_pv_v[-1]:.2f} V (V_mpp = {tr.v_mpp_true_v[-1]:.2f} V)"
    )
    if args.plot:
        print(f"  Saved Plot:        {args.plot}")
    print("=" * 72)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv = SolarPVModel()
    shaded = ShadedPVString(pv, n_modules=1, n_bypass_per_module=3)

    p1 = str(out_dir / "pv_characteristics.png")
    p2 = str(out_dir / "thermal_derating.png")
    p3 = str(out_dir / "mppt_tracking.png")
    p4 = str(out_dir / "partial_shading_curves.png")
    p5 = str(out_dir / "global_mppt_comparison.png")
    p6 = str(out_dir / "en50530_dynamic_benchmark.png")
    p7 = str(out_dir / "dynamic_boost_converter.png")

    plot_pv_characteristics(pv, save_path=p1)
    plot_thermal_derating(pv, save_path=p2)

    irr_prof = [1000.0] * 50 + [600.0] * 50 + [900.0] * 50
    t_steps = np.arange(len(irr_prof))
    p_true, p_po, p_inc = [], [], []
    po = PerturbAndObserve(step_size=0.45, v_init=22.0, v_min=0.0, v_max=pv.V_oc)
    inc = IncrementalConductance(
        step_size=0.45, v_init=22.0, v_min=0.0, v_max=pv.V_oc, conductance_tol=2e-2
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
    plot_mppt_tracking(t_steps, irr_prof, p_true, p_po, p_inc, save_path=p3)

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
    plot_partial_shading_curves(shading_cases, save_path=p4)

    gmppt_res = run_dynamic_gmppt_benchmark(shaded)
    plot_global_mppt_comparison(gmppt_res, save_path=p5)

    en_res = run_en50530_dynamic_benchmark(pv)
    plot_en50530_benchmark(en_res, save_path=p6)

    boost_tr = run_closed_loop_boost_transient(pv)
    eff_60 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=60.0))
    eff_120 = evaluate_converter_efficiency_curve(pv, BoostPowerStageParams(v_bus_v=120.0))
    plot_dynamic_boost_converter(boost_tr, eff_60, eff_120, save_path=p7)

    figures = [p1, p2, p3, p4, p5, p6, p7]
    payload: dict[str, Any] = {
        "output_dir": str(out_dir),
        "figures": figures,
        "gmppt_hybrid_dynamic_efficiency_pct": round(
            gmppt_res.trajectories["Hybrid Scan + Adaptive INC"].tracking_efficiency_pct, 2
        ),
        "en50530_sera_dp_po_efficiency_pct": round(
            en_res.controllers["Sera dP-P&O (Drift-Free)"].en50530_efficiency_pct, 2
        ),
        "boost_60v_euro_efficiency_pct": round(eff_60.euro_efficiency_pct, 2),
        "boost_rms_servo_error_mv": round(boost_tr.rms_servo_tracking_error_v * 1e3, 2),
    }
    if args.json:
        _emit_json(payload)
        return 0

    print("=" * 72)
    print(f"Full Multi-Physics Suite Complete, Generated {len(figures)} Figures in '{out_dir}'")
    print("-" * 72)
    print(
        f"  Hybrid GMPPT Dynamic Efficiency:  "
        f"{payload['gmppt_hybrid_dynamic_efficiency_pct']:.2f}%"
    )
    print(
        f"  Sera dP-P&O EN 50530 Efficiency:  "
        f"{payload['en50530_sera_dp_po_efficiency_pct']:.2f}%"
    )
    print(
        f"  Boost Stage (60 V) Euro Rating:   "
        f"{payload['boost_60v_euro_efficiency_pct']:.2f}% "
        f"(Servo RMS = {payload['boost_rms_servo_error_mv']:.1f} mV)"
    )
    print("=" * 72)
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    manifest = build_static_site(args.output_dir, results_dir=args.results_dir)
    if args.json:
        _emit_json(manifest)
        return 0

    print("=" * 72)
    print(f"Static Interactive Explorer Built -> {manifest['output_dir']}")
    print("-" * 72)
    print(f"  Entry HTML:      {manifest['index_html']}")
    print(f"  Solver Dataset:  {manifest['benchmark_json']}")
    print(f"  Figure Assets:   {len(manifest['assets'])} PNG files copied")
    print("=" * 72)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code (0 on success, 2 on domain error)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    handlers = {
        "info": _cmd_info,
        "iv": _cmd_iv,
        "shading": _cmd_shading,
        "en50530": _cmd_en50530,
        "boost": _cmd_boost,
        "benchmark": _cmd_benchmark,
        "web": _cmd_web,
    }
    try:
        return handlers[args.command](args)
    except ValueError as exc:
        print(f"pv-mppt error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
