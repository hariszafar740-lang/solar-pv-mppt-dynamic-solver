import numpy as np
from src.pv_model import SolarPVModel
from src.mppt_algorithms import PerturbAndObserve, IncrementalConductance
from src.visualizer import plot_pv_characteristics, plot_mppt_tracking

def main():
    pv = SolarPVModel()
    plot_pv_characteristics(pv)
    
    # Construct Dynamic Irradiance Step Profile (150 Time Steps)
    irradiance_profile = [1000.0]*50 + [600.0]*50 + [900.0]*50
    time_steps = np.arange(len(irradiance_profile))
    
    p_true = []
    p_po = []
    p_inc = []
    
    po = PerturbAndObserve(step_size=0.15, v_init=20.0)
    inc = IncrementalConductance(step_size=0.15, v_init=20.0)
    
    v_po = 20.0
    v_inc = 20.0
    
    for G in irradiance_profile:
        v_arr, i_arr, p_arr = pv.calculate_iv_curve(G=G)
        p_true.append(np.max(p_arr))
        
        # P&O Iteration
        p_curr_po = np.interp(v_po, v_arr, p_arr)
        p_po.append(p_curr_po)
        v_po = po.step(v_po, p_curr_po)
        
        # INC Iteration
        p_curr_inc = np.interp(v_inc, v_arr, p_arr)
        i_curr_inc = np.interp(v_inc, v_arr, i_arr)
        p_inc.append(p_curr_inc)
        v_inc = inc.step(v_inc, i_curr_inc)
        
    plot_mppt_tracking(time_steps, irradiance_profile, p_true, p_po, p_inc)
    
    eff_po = (np.mean(p_po) / np.mean(p_true)) * 100.0
    eff_inc = (np.mean(p_inc) / np.mean(p_true)) * 100.0
    
    print("=" * 70)
    print("DAY 9: DYNAMIC MPPT CONTROL ENGINE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Dynamic P&O Tracking Efficiency: {eff_po:.2f}%")
    print(f"Overall Dynamic INC Tracking Efficiency: {eff_inc:.2f}%")
    print("-" * 70)
    print("Visual artifacts updated:")
    print("  - /results/pv_characteristics.png")
    print("  - /results/mppt_tracking.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
