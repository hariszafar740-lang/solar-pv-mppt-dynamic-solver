"""Solar PV Dynamic Characterization & MPPT Solver package."""

from src.boost_converter import BoostConverterInterface
from src.dynamic_boost import (
    BoostLossBreakdown,
    BoostPowerStageParams,
    BoostTransientResult,
    CascadedBoostPIController,
    ConverterEfficiencyCurve,
    DynamicBoostConverter,
    compute_weighted_efficiencies,
    evaluate_converter_efficiency_curve,
    run_closed_loop_boost_transient,
)
from src.en50530_benchmark import (
    DeltaPPO,
    EN50530BenchmarkResult,
    EN50530ControllerSummary,
    EN50530RampSegment,
    LumpedThermalModel,
    build_en50530_composite_profile,
    generate_trapezoidal_irradiance_profile,
    run_en50530_dynamic_benchmark,
)
from src.global_mppt import (
    ControllerTrajectory,
    GlobalMPPTBenchmark,
    HybridGlobalScanINC,
    ParticleSwarmMPPT,
    run_dynamic_gmppt_benchmark,
)
from src.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from src.partial_shading import (
    BypassDiodeSpec,
    PowerPeak,
    ShadedPVString,
    ShadingResult,
    get_benchmark_shading_profiles,
)
from src.pv_model import SolarPVModel

__all__ = [
    "BoostConverterInterface",
    "BoostLossBreakdown",
    "BoostPowerStageParams",
    "BoostTransientResult",
    "BypassDiodeSpec",
    "CascadedBoostPIController",
    "ControllerTrajectory",
    "ConverterEfficiencyCurve",
    "DeltaPPO",
    "DynamicBoostConverter",
    "EN50530BenchmarkResult",
    "EN50530ControllerSummary",
    "EN50530RampSegment",
    "GlobalMPPTBenchmark",
    "HybridGlobalScanINC",
    "IncrementalConductance",
    "LumpedThermalModel",
    "ParticleSwarmMPPT",
    "PerturbAndObserve",
    "PowerPeak",
    "ShadedPVString",
    "ShadingResult",
    "SolarPVModel",
    "build_en50530_composite_profile",
    "compute_weighted_efficiencies",
    "evaluate_converter_efficiency_curve",
    "generate_trapezoidal_irradiance_profile",
    "get_benchmark_shading_profiles",
    "run_closed_loop_boost_transient",
    "run_dynamic_gmppt_benchmark",
    "run_en50530_dynamic_benchmark",
]
