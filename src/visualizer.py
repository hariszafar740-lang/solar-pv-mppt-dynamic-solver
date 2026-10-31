from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_pv_characteristics(pv_model, save_path='results/pv_characteristics.png'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    irradiances = [1000, 800, 600, 400]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6))
    mpp_v_list, mpp_p_list = [], []
    for G, col in zip(irradiances, colors, strict=True):
        v, i, p = pv_model.calculate_iv_curve(G=G, T_c_celsius=25.0, num_points=400)
        idx = int(np.argmax(p))
        v_mp, i_mp, p_mp = float(v[idx]), float(i[idx]), float(p[idx])
        mpp_v_list.append(v_mp)
        mpp_p_list.append(p_mp)

        ax1.plot(v, i, label=f'G = {G} W/m² (I_sc = {i[0]:.2f} A)', color=col, linewidth=2.1)
        ax1.scatter([v_mp], [i_mp], color=col, s=46, zorder=5, edgecolors='white', linewidths=1.1)

        ax2.plot(
            v,
            p,
            label=f'G = {G} W/m² (P_mpp = {p_mp:.1f} W @ {v_mp:.1f} V)',
            color=col,
            linewidth=2.1,
        )
        ax2.scatter([v_mp], [p_mp], color=col, s=50, zorder=5, edgecolors='white', linewidths=1.1)

    ax2.plot(
        mpp_v_list,
        mpp_p_list,
        color='#334155',
        linestyle=':',
        linewidth=1.5,
        label='MPP Locus V_mpp(G)',
        zorder=4,
    )

    ax1.set_title('I-V Characteristics under Variable Irradiance (T_c = 25°C)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Module Voltage V (Volts)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Module Current I (Amperes)', fontsize=10, fontweight='bold')
    ax1.grid(True, ls='--', alpha=0.45)
    ax1.legend(loc='lower left', fontsize=8.8, framealpha=0.94)

    ax2.set_ylim(0.0, max(mpp_p_list) * 1.12)
    ax2.set_title('P-V Characteristics & Maximum Power Point Locus',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Module Voltage V (Volts)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Output Power P (Watts)', fontsize=10, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.45)
    ax2.legend(loc='upper left', fontsize=8.6, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mppt_tracking(time_steps, irradiance, p_true, p_po, p_inc,
                       save_path='results/mppt_tracking.png'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    t_arr = np.asarray(time_steps, dtype=float)
    g_arr = np.asarray(irradiance, dtype=float)
    pt_arr = np.asarray(p_true, dtype=float)
    po_arr = np.asarray(p_po, dtype=float)
    inc_arr = np.asarray(p_inc, dtype=float)

    p_sum = float(np.sum(pt_arr))
    eta_po = (100.0 * float(np.sum(po_arr)) / p_sum) if p_sum > 0.0 else 0.0
    eta_inc = (100.0 * float(np.sum(inc_arr)) / p_sum) if p_sum > 0.0 else 0.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={'height_ratios': [1.0, 1.55]}
    )

    ax1.plot(t_arr, g_arr, color='#334155', linewidth=2.3, label='Irradiance G(k) (W/m²)')
    ax1.fill_between(t_arr, g_arr, float(np.min(g_arr)) - 40.0, color='#f59e0b', alpha=0.14)
    ax1.set_ylim(max(0.0, float(np.min(g_arr)) - 60.0), float(np.max(g_arr)) + 75.0)
    ax1.set_ylabel('Irradiance (W/m²)', fontsize=10, fontweight='bold')
    ax1.set_title('Dynamic Solar Irradiance Step Profile (1000 → 600 → 900 W/m²)',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, ls='--', alpha=0.45)
    ax1.legend(loc='lower right', fontsize=9.0, framealpha=0.94)

    ax2.plot(
        t_arr,
        pt_arr,
        color='#111827',
        linestyle=':',
        linewidth=2.1,
        alpha=0.85,
        label='Theoretical Instantaneous P_mpp(k)',
        zorder=3,
    )
    ax2.plot(
        t_arr,
        po_arr,
        color='#1f77b4',
        linewidth=2.6,
        alpha=0.82,
        label=f'Perturb & Observe, P&O (η = {eta_po:.2f}%)',
        zorder=4,
    )
    ax2.plot(
        t_arr,
        inc_arr,
        color='#16a34a',
        linestyle='--',
        linewidth=2.0,
        label=f'Incremental Conductance, INC (η = {eta_inc:.2f}%)',
        zorder=5,
    )
    ax2.set_ylim(
        max(0.0, float(min(po_arr.min(), inc_arr.min())) - 18.0),
        float(pt_arr.max()) + 22.0,
    )
    ax2.set_xlabel('Simulation Control Step k (-)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Extracted Power P (Watts)', fontsize=10, fontweight='bold')
    ax2.set_title('MPPT Control Engine Dynamic Step-Tracking & Steady-State Convergence',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.45)
    ax2.legend(loc='lower right', fontsize=9.0, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_thermal_derating(pv_model, save_path='results/thermal_derating.png'):
    """Renders thermal power derating curves across operating temperatures."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    temperatures = [25, 45, 65]
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    v_peaks, p_peaks = [], []
    for T_c, col in zip(temperatures, colors, strict=True):
        v, _, p = pv_model.calculate_iv_curve(G=1000.0, T_c_celsius=T_c, num_points=400)
        idx = int(np.argmax(p))
        vp, pp = float(v[idx]), float(p[idx])
        v_peaks.append(vp)
        p_peaks.append(pp)
        ax.plot(
            v,
            p,
            label=f'T_cell = {T_c}°C (P_max = {pp:.1f} W @ V_mpp = {vp:.1f} V)',
            color=col,
            linewidth=2.2,
        )
        ax.scatter([vp], [pp], color=col, s=54, zorder=5, edgecolors='white', linewidths=1.2)

    ax.plot(
        v_peaks,
        p_peaks,
        color='#334155',
        linestyle='--',
        linewidth=1.5,
        label='Thermal V_mpp Drift Locus (γ_P ≈ -0.39%/K)',
        zorder=4,
    )
    ax.set_ylim(0.0, max(p_peaks) * 1.13)
    ax.set_title('PV Module Thermal Power Derating & MPP Voltage Shift (G = 1000 W/m²)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Module Voltage V (Volts)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Power Output P (Watts)', fontsize=10, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.45)
    ax.legend(loc='upper left', fontsize=8.8, framealpha=0.94)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_partial_shading_curves(
    shading_results,
    save_path='results/partial_shading_curves.png',
):
    """Render bypass-diode staircase I-V and multi-peak P-V curves with GMPP/LMPP markers."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(shading_results, dict):
        items = [("Partial Shading", shading_results)]
    else:
        items = list(shading_results.items())

    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6))

    max_p = 1.0
    line_handles = []
    has_local_peaks = False

    for idx, (label, res) in enumerate(items):
        col = palette[idx % len(palette)]
        ax1.plot(res.voltages, res.currents, color=col, linewidth=2.1, label=label)
        (line_h,) = ax2.plot(res.voltages, res.powers, color=col, linewidth=2.1, label=label)
        line_handles.append(line_h)
        max_p = max(max_p, float(res.gmpp.power_w))

        ax2.scatter(
            [res.gmpp.voltage_v],
            [res.gmpp.power_w],
            color=col,
            edgecolors='#111111',
            linewidths=0.9,
            marker='*',
            s=145,
            zorder=6,
        )
        local_peaks = [pk for pk in res.peaks if not pk.is_global]
        if local_peaks:
            has_local_peaks = True
            ax2.scatter(
                [pk.voltage_v for pk in local_peaks],
                [pk.power_w for pk in local_peaks],
                facecolors='white',
                edgecolors=col,
                linewidths=1.8,
                marker='o',
                s=52,
                zorder=5,
            )

    gmpp_proxy = Line2D(
        [0],
        [0],
        marker='*',
        color='none',
        markerfacecolor='#f59e0b',
        markeredgecolor='#111111',
        markersize=11,
        label='GMPP (Global Peak)',
    )
    legend_handles = [*line_handles, gmpp_proxy]
    if has_local_peaks:
        lmpp_proxy = Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor='white',
            markeredgecolor='#374151',
            markeredgewidth=1.8,
            markersize=7,
            label='LMPP (Local Trap)',
        )
        legend_handles.append(lmpp_proxy)

    ax1.set_title(
        'Bypass-Diode Staircase I-V Characteristics',
        fontsize=11.5,
        fontweight='bold',
    )
    ax1.set_xlabel('String Terminal Voltage V (Volts)', fontsize=10)
    ax1.set_ylabel('String Current I (Amperes)', fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=8.8)

    ax2.set_ylim(0.0, max_p * 1.13)
    ax2.set_title(
        'Multi-Peak P-V Characteristics (GMPP vs. Local Traps)',
        fontsize=11.5,
        fontweight='bold',
    )
    ax2.set_xlabel('String Terminal Voltage V (Volts)', fontsize=10)
    ax2.set_ylabel('String Output Power P (Watts)', fontsize=10)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend(handles=legend_handles, loc='upper left', fontsize=8.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_global_mppt_comparison(
    benchmark,
    save_path='results/global_mppt_comparison.png',
):
    """Render 2-panel dynamic Global MPPT vs. Conventional MPPT benchmark under shading."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.8, 8.3), sharex=True)

    t = benchmark.time_steps
    max_p = float(max(benchmark.gmpp_powers)) if len(benchmark.gmpp_powers) else 100.0

    styles = {
        "Conventional P&O": ("#d62728", "-", 1.75, 0.88, 3),
        "Conventional INC": ("#9467bd", "--", 1.65, 0.88, 4),
        "PSO Global MPPT": ("#1f77b4", "-", 1.90, 0.92, 5),
        "Hybrid Scan + Adaptive INC": ("#2ca02c", "-", 2.15, 0.96, 6),
    }

    ax1.plot(
        t,
        benchmark.gmpp_powers,
        color="#111111",
        linestyle=":",
        linewidth=2.3,
        label="True Dynamic GMPP",
        zorder=7,
    )
    ax2.plot(
        t,
        benchmark.gmpp_voltages,
        color="#111111",
        linestyle=":",
        linewidth=2.3,
        label="True GMPP Voltage",
        zorder=7,
    )

    for name, traj in benchmark.trajectories.items():
        col, ls, lw, alpha, zo = styles.get(name, ("#ff7f0e", "-", 1.8, 0.9, 4))
        lbl = (
            f"{name} (Dyn {traj.tracking_efficiency_pct:.1f}% / "
            f"Settled {traj.settled_efficiency_pct:.1f}%)"
        )
        ax1.plot(
            t,
            traj.powers,
            color=col,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            label=lbl,
            zorder=zo,
        )
        ax2.plot(
            t,
            traj.voltages,
            color=col,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            label=name,
            zorder=zo,
        )

    boundaries = list(benchmark.stage_boundaries) + [len(t)]
    for b_idx, start in enumerate(benchmark.stage_boundaries):
        if start > 0:
            ax1.axvline(start, color="#6b7280", linestyle="--", linewidth=1.1, alpha=0.7)
            ax2.axvline(start, color="#6b7280", linestyle="--", linewidth=1.1, alpha=0.7)
        end = boundaries[b_idx + 1]
        mid = 0.5 * (start + end)
        short_lbl = benchmark.stage_labels[b_idx]
        ax1.text(
            mid,
            max_p * 1.11,
            short_lbl,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#1f2937",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#f3f4f6",
                "edgecolor": "#d1d5db",
                "alpha": 0.95,
            },
        )

    ax1.set_ylim(0.0, max_p * 1.22)
    ax1.set_ylabel("String Output Power P (Watts)", fontsize=10, fontweight="bold")
    ax1.set_title(
        "Global MPPT (PSO & Hybrid Scan + Adaptive INC) vs. Local-Trap Conventional MPPT",
        fontsize=11.5,
        fontweight="bold",
        pad=38,
    )
    ax1.grid(True, ls="--", alpha=0.45)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=8.3,
        framealpha=0.95,
    )

    ax2.set_ylim(0.0, 43.0)
    ax2.set_xlabel("Control Time Steps (k)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Operating Voltage V (Volts)", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Operating Voltage Trajectory Across Multi-Peak Bypass-Diode Basins",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.grid(True, ls="--", alpha=0.45)
    ax2.legend(loc="lower right", ncol=3, fontsize=8.4, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_en50530_benchmark(
    result,
    save_path='results/en50530_dynamic_benchmark.png',
):
    """Render 3-panel CENELEC EN 50530 ramp & coupled electro-thermal transient benchmark."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13.8, 10.2), sharex=True)

    t = result.time_s
    max_g = float(max(result.irradiance_wm2)) if len(result.irradiance_wm2) else 1000.0
    max_p = float(max(result.ideal_mpp_power_w)) if len(result.ideal_mpp_power_w) else 300.0

    (l_irr,) = ax1.plot(
        t,
        result.irradiance_wm2,
        color="#d97706",
        linewidth=2.3,
        label="EN 50530 Irradiance G(t)",
    )
    ax1.fill_between(t, 0.0, result.irradiance_wm2, color="#fef3c7", alpha=0.45)
    ax1.set_ylim(0.0, max_g * 1.24)
    ax1.set_ylabel("Irradiance G (W/m²)", fontsize=10, fontweight="bold", color="#b45309")
    ax1.grid(True, ls="--", alpha=0.45)

    ax1_t = ax1.twinx()
    (l_toc,) = ax1_t.plot(
        t,
        result.open_circuit_cell_temp_c,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label="Open-Circuit Cell Temp T_c,oc(t)",
    )
    (l_tmpp,) = ax1_t.plot(
        t,
        result.ideal_cell_temp_c,
        color="#0f766e",
        linestyle="-.",
        linewidth=2.0,
        label="MPPT-Cooled Cell Temp T_c,mpp(t)",
    )
    t_min_plot = min(float(min(result.ideal_cell_temp_c)), 24.0) - 1.5
    t_max_plot = float(max(result.open_circuit_cell_temp_c)) + 4.5
    ax1_t.set_ylim(t_min_plot, t_max_plot)
    ax1_t.set_ylabel("Cell Temp T_c (°C)", fontsize=10, fontweight="bold", color="#0f766e")

    boundaries = list(result.segment_boundaries_s) + [float(t[-1])]
    for idx, start_s in enumerate(result.segment_boundaries_s):
        if start_s > 0.0:
            for ax in (ax1, ax2, ax3):
                ax.axvline(start_s, color="#6b7280", linestyle="--", linewidth=1.1, alpha=0.7)
        mid_s = 0.5 * (start_s + boundaries[idx + 1])
        ax1.text(
            mid_s,
            max_g * 1.12,
            result.segment_labels[idx],
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#1f2937",
            bbox={
                "boxstyle": "round,pad=0.26",
                "facecolor": "#f9fafb",
                "edgecolor": "#d1d5db",
                "alpha": 0.95,
            },
        )

    ax1.set_title(
        "CENELEC EN 50530 Dynamic Irradiance Ramps & Coupled Electro-Thermal Cell Transient",
        fontsize=11.5,
        fontweight="bold",
        pad=32,
    )
    ax1.legend(
        handles=[l_irr, l_toc, l_tmpp],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=8.5,
        framealpha=0.95,
    )

    styles = {
        "Standard P&O": ("#d62728", "-", 1.75, 0.88, 3),
        "Incremental Conductance": ("#9467bd", "--", 1.70, 0.90, 4),
        "Sera dP-P&O (Drift-Free)": ("#16a34a", "-", 2.15, 0.96, 6),
    }

    ax2.plot(
        t,
        result.ideal_mpp_power_w,
        color="#111111",
        linestyle=":",
        linewidth=2.3,
        label="True Electro-Thermal MPP Power",
        zorder=7,
    )
    ax3.plot(
        t,
        result.ideal_mpp_voltage_v,
        color="#111111",
        linestyle=":",
        linewidth=2.3,
        label="True Electro-Thermal V_mpp(t)",
        zorder=7,
    )

    for name, summary in result.controllers.items():
        col, ls, lw, alpha, zo = styles.get(name, ("#1f77b4", "-", 1.8, 0.9, 5))
        p_lbl = (
            f"{name} (EN 50530 η = {summary.en50530_efficiency_pct:.2f}%, "
            f"RMS ΔV = {summary.rms_voltage_error_v:.2f} V)"
        )
        ax2.plot(
            t,
            summary.powers,
            color=col,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            label=p_lbl,
            zorder=zo,
        )
        ax3.plot(
            t,
            summary.voltages,
            color=col,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            label=name,
            zorder=zo,
        )

    ax2.set_ylim(0.0, max_p * 1.15)
    ax2.set_ylabel("Extracted Power P (Watts)", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Dynamic Power Extraction & EN 50530 Integral Tracking Efficiency",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.grid(True, ls="--", alpha=0.45)
    ax2.legend(loc="upper left", fontsize=8.4, framealpha=0.94)

    v_all = [result.ideal_mpp_voltage_v] + [c.voltages for c in result.controllers.values()]
    v_lo = max(0.0, min(float(v.min()) for v in v_all) - 2.5)
    v_hi = max(float(v.max()) for v in v_all) + 2.5
    ax3.set_ylim(v_lo, v_hi)
    ax3.set_xlabel("Simulation Time t (Seconds)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Operating Voltage V (Volts)", fontsize=10, fontweight="bold")
    ax3.set_title(
        "Ramp-Drift Voltage Excursion vs. Thermally Shifting V_mpp(t)",
        fontsize=11.0,
        fontweight="bold",
    )
    ax3.grid(True, ls="--", alpha=0.45)
    ax3.legend(loc="lower right", ncol=2, fontsize=8.5, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dynamic_boost_converter(
    transient,
    eff_curve_60v,
    eff_curve_120v,
    save_path="results/dynamic_boost_converter.png",
):
    """Render a 2x2 power-electronics & cascaded current-mode boost dashboard."""
    import numpy as np

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.5, 9.6))

    t_ms = transient.time_ms
    g_arr = transient.irradiance_wm2
    step_indices = np.where(np.abs(np.diff(g_arr)) > 1e-3)[0] + 1
    step_times_ms = [float(t_ms[idx]) for idx in step_indices]

    # --- Panel (a): Cascaded PV Voltage Servo ---
    ax1.plot(
        t_ms,
        transient.v_mpp_true_v,
        color="#2ca02c",
        linestyle=":",
        linewidth=2.0,
        label="True Instantaneous V_mpp(G)",
        zorder=3,
    )
    ax1.plot(
        t_ms,
        transient.v_ref_v,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.9,
        label="Outer MPPT Command v_ref(t)",
        zorder=4,
    )
    ax1.plot(
        t_ms,
        transient.v_pv_v,
        color="#1f77b4",
        linewidth=2.1,
        label=(
            f"Capacitor Voltage v_pv(t) "
            f"(RMS Servo ΔV = {transient.rms_servo_tracking_error_v * 1e3:.1f} mV)"
        ),
        zorder=5,
    )
    for st in step_times_ms:
        ax1.axvline(st, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.6)

    v_min_plot = min(
        float(transient.v_pv_v.min()),
        float(transient.v_ref_v.min()),
        float(transient.v_mpp_true_v.min()),
    )
    v_max_plot = max(
        float(transient.v_pv_v.max()),
        float(transient.v_ref_v.max()),
        float(transient.v_mpp_true_v.max()),
    )
    ax1.set_ylim(v_min_plot - 0.8, v_max_plot + 1.45)

    bounds = [0.0] + step_times_ms + [float(t_ms[-1])]
    for i in range(len(bounds) - 1):
        mid_t = 0.5 * (bounds[i] + bounds[i + 1])
        mid_idx = int(np.searchsorted(t_ms, mid_t))
        g_stage = g_arr[min(mid_idx, len(g_arr) - 1)]
        ax1.text(
            mid_t,
            v_max_plot + 0.95,
            f"G = {g_stage:.0f} W/m²",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#f4f6f8",
                "edgecolor": "#cbd5e1",
                "alpha": 0.92,
            },
        )

    ax1.set_title(
        "(a) Closed-Loop Cascaded PI PV-Voltage Servo Response",
        fontsize=11.0,
        fontweight="bold",
    )
    ax1.set_xlabel("Time t (Milliseconds)", fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("PV Terminal Voltage (Volts)", fontsize=9.8, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=8.3, framealpha=0.94)

    # --- Panel (b): Inductor Current & PWM Duty Cycle ---
    ax2_r = ax2.twinx()
    l_ipv = ax2.plot(
        t_ms,
        transient.i_pv_a,
        color="#475569",
        linestyle="--",
        linewidth=1.7,
        label="PV Array Current i_pv(t)",
        zorder=3,
    )
    l_il = ax2.plot(
        t_ms,
        transient.i_inductor_a,
        color="#d62728",
        linewidth=2.0,
        label="Inductor Current i_L(t)",
        zorder=4,
    )
    l_duty = ax2_r.plot(
        t_ms,
        transient.duty_cycle,
        color="#7e22ce",
        linewidth=1.55,
        alpha=0.88,
        label="PWM Duty Ratio D(t)",
        zorder=2,
    )
    for st in step_times_ms:
        ax2.axvline(st, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.6)

    ax2.set_ylim(
        max(0.0, float(min(transient.i_inductor_a.min(), transient.i_pv_a.min())) - 1.2),
        float(max(transient.i_inductor_a.max(), transient.i_pv_a.max())) + 2.2,
    )
    ax2_r.set_ylim(0.0, 1.02)
    ax2.set_title(
        "(b) Average-Current-Mode Inductor Slew & PWM Duty Command",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.set_xlabel("Time t (Milliseconds)", fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Current (Amperes)", fontsize=9.8, fontweight="bold", color="#b91c1c")
    ax2_r.set_ylabel("PWM Duty Ratio D (-)", fontsize=9.8, fontweight="bold", color="#6b21a8")
    ax2.grid(True, ls="--", alpha=0.45)

    lines_b = l_il + l_ipv + l_duty
    labels_b = [ln.get_label() for ln in lines_b]
    ax2.legend(lines_b, labels_b, loc="upper right", fontsize=8.3, framealpha=0.94)

    # --- Panel (c): Weighted Converter Efficiency Curves (60 V vs 120 V Bus) ---
    pct_load_60 = eff_curve_60v.load_fractions * 100.0
    pct_load_120 = eff_curve_120v.load_fractions * 100.0

    lbl_60 = (
        f"V_bus = {eff_curve_60v.v_bus_v:.0f} V LVDC "
        f"(Peak {eff_curve_60v.peak_efficiency_pct:.2f}% | "
        f"η_Euro {eff_curve_60v.euro_efficiency_pct:.2f}% | "
        f"η_CEC {eff_curve_60v.cec_efficiency_pct:.2f}%)"
    )
    lbl_120 = (
        f"V_bus = {eff_curve_120v.v_bus_v:.0f} V HVDC "
        f"(Peak {eff_curve_120v.peak_efficiency_pct:.2f}% | "
        f"η_Euro {eff_curve_120v.euro_efficiency_pct:.2f}% | "
        f"η_CEC {eff_curve_120v.cec_efficiency_pct:.2f}%)"
    )

    ax3.plot(
        pct_load_60,
        eff_curve_60v.efficiencies_pct,
        color="#0284c7",
        marker="o",
        markersize=4.0,
        linewidth=2.1,
        label=lbl_60,
    )
    ax3.plot(
        pct_load_120,
        eff_curve_120v.efficiencies_pct,
        color="#ea580c",
        marker="s",
        markersize=4.0,
        linestyle="--",
        linewidth=2.0,
        label=lbl_120,
    )

    pk60_idx = int(np.argmax(eff_curve_60v.efficiencies_pct))
    ax3.scatter(
        [pct_load_60[pk60_idx]],
        [eff_curve_60v.efficiencies_pct[pk60_idx]],
        color="#0284c7",
        s=65,
        zorder=6,
        edgecolors="white",
        linewidths=1.2,
    )

    eta_min = min(
        float(eff_curve_60v.efficiencies_pct.min()),
        float(eff_curve_120v.efficiencies_pct.min()),
    )
    eta_max = max(
        float(eff_curve_60v.efficiencies_pct.max()),
        float(eff_curve_120v.efficiencies_pct.max()),
    )
    ax3.set_ylim(eta_min - 0.65, min(100.0, eta_max + 0.55))
    ax3.set_xlim(0.0, 104.0)
    ax3.set_title(
        "(c) Converter Efficiency Curve & European / CEC Weighted Ratings",
        fontsize=11.0,
        fontweight="bold",
    )
    ax3.set_xlabel(
        "Normalized PV Input Load (% of STC Rated Power)",
        fontsize=9.8,
        fontweight="bold",
    )
    ax3.set_ylabel("Stage Conversion Efficiency η (%)", fontsize=9.8, fontweight="bold")
    ax3.grid(True, ls="--", alpha=0.45)
    ax3.legend(loc="lower right", fontsize=8.0, framealpha=0.94)

    # --- Panel (d): Stacked Component Loss Breakdown (60 V Bus) ---
    p_in = eff_curve_60v.p_in_w
    bds = eff_curve_60v.breakdowns
    loss_cu = np.array([b.p_inductor_copper_w for b in bds])
    loss_diode = np.array([b.p_diode_cond_w for b in bds])
    loss_mos_cond = np.array([b.p_mosfet_cond_w for b in bds])
    loss_mos_sw = np.array([b.p_mosfet_sw_w for b in bds])
    loss_core = np.array([b.p_core_w for b in bds])

    ax4.stackplot(
        p_in,
        loss_cu,
        loss_diode,
        loss_mos_cond,
        loss_mos_sw,
        loss_core,
        labels=[
            "Inductor Winding Copper (I_rms² R_L)",
            "Freewheeling Diode Conduction",
            "MOSFET On-State Conduction (D I_rms² R_dson)",
            "MOSFET Switching + C_oss + Q_rr",
            "Magnetic Core Ripple Loss",
        ],
        colors=["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"],
        alpha=0.86,
    )
    tot_loss = loss_cu + loss_diode + loss_mos_cond + loss_mos_sw + loss_core
    ax4.plot(p_in, tot_loss, color="#111827", linewidth=1.6, label="Total Stage Loss P_loss")

    ax4.set_xlim(float(p_in[0]), float(p_in[-1]))
    ax4.set_ylim(0.0, float(tot_loss.max()) * 1.18)
    ax4.set_title(
        f"(d) Semiconductor & Magnetic Loss Breakdown (V_bus = {eff_curve_60v.v_bus_v:.0f} V)",
        fontsize=11.0,
        fontweight="bold",
    )
    ax4.set_xlabel("PV Input Power P_in (Watts)", fontsize=9.8, fontweight="bold")
    ax4.set_ylabel("Dissipated Power Loss (Watts)", fontsize=9.8, fontweight="bold")
    ax4.grid(True, ls="--", alpha=0.45)
    ax4.legend(loc="upper left", fontsize=8.0, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
