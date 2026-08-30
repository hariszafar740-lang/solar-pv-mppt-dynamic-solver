import numpy as np
from src.pv_model import SolarPVModel
from src.mppt_algorithms import PerturbAndObserve, IncrementalConductance
from src.boost_converter import BoostConverterInterface
from src.visualizer import plot_pv_characteristics, plot_mppt_tracking, plot_thermal_derating

def main():
    pv = SolarPVModel()
    converter = BoostConverterInterface(v_out=60.0, efficiency=0.95)
    
    # Generate Plots
    plot_pv_characteristics(pv)
    plot_thermal_derating(pv)
    
    # Run Dynamic MPPT Profile
    irradiance_profile = [1000.0]*50 + [600.0]*50 + [900.0]*50
    time_steps = np.arange(len(irradiance_profile))
    
    p_true, p_po, p_inc = [], [], []
    po = PerturbAndObserve(step_size=0.15, v_init=20.0)
    inc = IncrementalConductance(step_size=0.15, v_init=20.0)
    v_po, v_inc = 20.0, 20.0
    
    for G in irradiance_profile:
        v_arr, i_arr, p_arr = pv.calculate_iv_curve(G=G)
        p_true.append(np.max(p_arr))
        
        # P&O Step
        p_curr_po = np.interp(v_po, v_arr, p_arr)
        p_po.append(p_curr_po)
        v_po = po.step(v_po, p_curr_po)
        
        # INC Step
        p_curr_inc = np.interp(v_inc, v_arr, p_arr)
        i_curr_inc = np.interp(v_inc, v_arr, i_arr)
        p_inc.append(p_curr_inc)
        v_inc = inc.step(v_inc, i_curr_inc)
        
    plot_mppt_tracking(time_steps, irradiance_profile, p_true, p_po, p_inc)
    
    # Boost Converter Duty Cycle Translation
    avg_v_mpp = np.mean([v_po, v_inc])
    duty_cycle = converter.compute_duty_cycle(avg_v_mpp)
    net_p_out = converter.compute_output_power(np.mean(p_inc))
    
    print("=" * 70)
    print("DAY 10: COMPLETE PROJECT 2 BENCHMARK & CONVERTER EVALUATION")
    print("=" * 70)
    print(f"Tracking Voltage V_mpp:                 {avg_v_mpp:.2f} V")
    print(f"Boost Converter Duty Ratio (D):         {duty_cycle:.4f}")
    print(f"DC Bus Link Power Delivered (P_out):    {net_p_out:.2f} W (eta = 95%)")
    print("-" * 70)
    print("Thermal Derating Metrics (1000 W/m²):")
    for T in [25, 45, 65]:
        _, _, p_t = pv.calculate_iv_curve(G=1000.0, T_c_celsius=T)
        print(f"  T = {T}°C -> Peak Power P_max = {max(p_t):.2f} W")
    print("=" * 70)

if __name__ == "__main__":
    main()
