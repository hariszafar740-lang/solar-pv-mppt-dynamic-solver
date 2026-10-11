import matplotlib.pyplot as plt


def plot_pv_characteristics(pv_model, save_path='results/pv_characteristics.png'):
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
