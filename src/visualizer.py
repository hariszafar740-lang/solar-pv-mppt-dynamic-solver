import matplotlib.pyplot as plt

def plot_pv_characteristics(pv_model, save_path='results/pv_characteristics.png'):
    """Generates combined I-V and P-V characteristic curves under variable irradiance."""
    irradiances = [1000, 800, 600, 400]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for G, col in zip(irradiances, colors):
        v, i, p = pv_model.calculate_iv_curve(G=G, T_c_celsius=25.0)
        ax1.plot(v, i, label=f'G = {G} W/m²', color=col, linewidth=2)
        ax2.plot(v, p, label=f'G = {G} W/m²', color=col, linewidth=2)
        
    ax1.set_title('I-V Characteristics under Variable Irradiance', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Voltage V (Volts)', fontsize=10)
    ax1.set_ylabel('Current I (Amperes)', fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()
    
    ax2.set_title('P-V Characteristics under Variable Irradiance', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Voltage V (Volts)', fontsize=10)
    ax2.set_ylabel('Power P (Watts)', fontsize=10)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
