import numpy as np
from src.pv_model import SolarPVModel
from src.visualizer import plot_pv_characteristics

def main():
    pv = SolarPVModel()
    plot_pv_characteristics(pv)
    
    print("=" * 65)
    print("DAY 8: SOLAR PV SINGLE-DIODE DYNAMIC MODEL RESULTS")
    print("=" * 65)
    
    for G in [1000, 800, 600, 400]:
        v, i, p = pv.calculate_iv_curve(G=G)
        mpp_idx = np.argmax(p)
        print(f"Irradiance G = {G:4d} W/m² | P_max = {p[mpp_idx]:6.2f} W | V_mpp = {v[mpp_idx]:5.2f} V | I_mpp = {i[mpp_idx]:4.2f} A")
        
    print("-" * 65)
    print("Visual artifact generated and saved to: results/pv_characteristics.png")
    print("=" * 65)

if __name__ == "__main__":
    main()
