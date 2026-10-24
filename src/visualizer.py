from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_pv_characteristics(pv_model, save_path='results/pv_characteristics.png'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    irradiances = [1000, 800, 600, 400]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for G, col in zip(irradiances, colors, strict=True):
        v, i, p = pv_model.calculate_iv_curve(G=G, T_c_celsius=25.0)
        ax1.plot(v, i, label=f'G = {G} W/m²', color=col, linewidth=2)
        ax2.plot(v, p, label=f'G = {G} W/m²', color=col, linewidth=2)

    ax1.set_title('I-V Characteristics under Variable Irradiance',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Voltage V (Volts)', fontsize=10)
    ax1.set_ylabel('Current I (Amperes)', fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()

    ax2.set_title('P-V Characteristics under Variable Irradiance',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Voltage V (Volts)', fontsize=10)
    ax2.set_ylabel('Power P (Watts)', fontsize=10)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mppt_tracking(time_steps, irradiance, p_true, p_po, p_inc,
                       save_path='results/mppt_tracking.png'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(time_steps, irradiance, color='#7f7f7f', linewidth=2.5,
             label='Irradiance (W/m²)')
    ax1.set_ylabel('Irradiance (W/m²)', fontsize=10, fontweight='bold')
    ax1.set_title('Dynamic Solar Irradiance Step Profile',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(loc='upper right')

    ax2.plot(time_steps, p_true, 'k--', label='Theoretical MPP Power',
             linewidth=2, alpha=0.7)
    ax2.plot(time_steps, p_po, color='#1f77b4', label='Perturb & Observe (P&O)',
             linewidth=1.8)
    ax2.plot(time_steps, p_inc, color='#2ca02c', label='Incremental Conductance (INC)',
             linewidth=1.8)
    ax2.set_xlabel('Simulation Time Steps (k)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Extracted Power (Watts)', fontsize=10, fontweight='bold')
    ax2.set_title('MPPT Control Engine Dynamic Tracking Performance',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_thermal_derating(pv_model, save_path='results/thermal_derating.png'):
    """Renders thermal power derating curves across operating temperatures."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    temperatures = [25, 45, 65]
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    plt.figure(figsize=(8, 5))
    for T_c, col in zip(temperatures, colors, strict=True):
        v, _, p = pv_model.calculate_iv_curve(G=1000.0, T_c_celsius=T_c)
        plt.plot(v, p, label=f'T = {T_c}°C', color=col, linewidth=2)

    plt.title('PV Module Thermal Power Derating Curve (G = 1000 W/m²)',
              fontsize=11, fontweight='bold')
    plt.xlabel('Module Voltage V (Volts)', fontsize=10)
    plt.ylabel('Power Output P (Watts)', fontsize=10)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
