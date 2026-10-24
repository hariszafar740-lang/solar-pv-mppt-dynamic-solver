import numpy as np

from src.boost_converter import BoostConverterInterface
from src.en50530_benchmark import run_en50530_dynamic_benchmark
from src.global_mppt import run_dynamic_gmppt_benchmark
from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.partial_shading import ShadedPVString, get_benchmark_shading_profiles
from src.pv_model import SolarPVModel
from src.visualizer import (
    plot_en50530_benchmark,
    plot_global_mppt_comparison,
    plot_mppt_tracking,
    plot_partial_shading_curves,
    plot_pv_characteristics,
    plot_thermal_derating,
)


def main():
    pv = SolarPVModel()
    converter = BoostConverterInterface(v_out=60.0, efficiency=0.95)
    shaded_string = ShadedPVString(pv, n_modules=1, n_bypass_per_module=3)

    ref = pv.REFERENCE_MODULE
    print("=" * 70)
    print("SOLAR PV DYNAMIC CHARACTERIZATION & MPPT SOLVER BENCHMARK")
    print("=" * 70)
    print(f"Reference module:  {ref['name']}")
    print(f"Dataset:           {ref['database']} ({ref['record_version']})")
    print(f"Extracted diode params: n = {pv.n:.6f}, "
          f"I_ph_ref = {pv.I_ph_ref:.6f} A, I_o_ref = {pv.I_o_ref:.6e} A")
    print("-" * 70)

    # Generate Plots
    plot_pv_characteristics(pv)
    plot_thermal_derating(pv)

    # Partial-Shading Bypass-Diode Characterization
    profiles = get_benchmark_shading_profiles()
    shading_cases = {
        "Uniform STC (1000 / 1000 / 1000 W/m²)": shaded_string.solve_curve(
            profiles["uniform_stc"], num_points=600
        ),
        "Chimney Shade (1000 / 700 / 350 W/m²)": shaded_string.solve_curve(
            profiles["moderate_chimney_shade"], num_points=600
        ),
        "Cloud + Soiling (850 / 520 / 200 W/m²)": shaded_string.solve_curve(
            profiles["severe_cloud_edge"], num_points=600
        ),
    }
    plot_partial_shading_curves(shading_cases)

    # Global MPPT vs. Conventional MPPT under 3-Phase Dynamic Partial Shading
    gmppt_bench = run_dynamic_gmppt_benchmark(shaded_string)
    plot_global_mppt_comparison(gmppt_bench)

    # CENELEC EN 50530 Dynamic Ramp & Coupled Electro-Thermal Cell Transient
    en50530_res = run_en50530_dynamic_benchmark(pv)
    plot_en50530_benchmark(en50530_res)

    # Run Dynamic MPPT Profile (bounded to the physical operating window)
    irradiance_profile = [1000.0]*50 + [600.0]*50 + [900.0]*50
    time_steps = np.arange(len(irradiance_profile))

    p_true, p_po, p_inc = [], [], []
    po = PerturbAndObserve(step_size=0.15, v_init=20.0, v_min=0.0, v_max=pv.V_oc)
    inc = IncrementalConductance(step_size=0.15, v_init=20.0,
                                 v_min=0.0, v_max=pv.V_oc)
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

    print(f"Tracking Voltage V_mpp:                 {avg_v_mpp:.2f} V")
    print(f"Boost Converter Duty Ratio (D):         {duty_cycle:.4f}")
    print(f"DC Bus Link Power Delivered (P_out):    {net_p_out:.2f} W (eta = 95%)")
    print("-" * 70)
    print("Bypass-Diode Partial-Shading Multi-Peak Diagnostics:")
    for label, res in shading_cases.items():
        print(
            f"  {label:<39} -> GMPP = {res.gmpp.power_w:6.2f} W @ "
            f"{res.gmpp.voltage_v:5.2f} V | peaks = {res.num_peaks} | "
            f"mismatch = {res.mismatch_loss_pct:4.1f}%"
        )
    print("-" * 70)
    print("Global MPPT Dynamic Shading Benchmark (3-Phase Partial Shading):")
    for name, traj in gmppt_bench.trajectories.items():
        print(
            f"  {name:<28} -> Dynamic Eff = {traj.tracking_efficiency_pct:5.2f}% | "
            f"Settled GMPP Eff = {traj.settled_efficiency_pct:5.2f}%"
        )
    print("-" * 70)
    print("CENELEC EN 50530 Dynamic Ramp & Coupled Electro-Thermal Benchmark:")
    for name, summary in en50530_res.controllers.items():
        print(
            f"  {name:<28} -> EN 50530 eta = {summary.en50530_efficiency_pct:5.2f}% | "
            f"RMS dV = {summary.rms_voltage_error_v:4.2f} V"
        )
    print("-" * 70)
    print("Thermal Derating Metrics (1000 W/m^2):")
    for T in [25, 45, 65]:
        _, _, p_t = pv.calculate_iv_curve(G=1000.0, T_c_celsius=T)
        print(f"  T = {T} degC -> Peak Power P_max = {max(p_t):.2f} W")
    print("=" * 70)


if __name__ == "__main__":
    main()
