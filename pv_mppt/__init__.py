"""Solar PV Dynamic Characterization & MPPT Solver package."""

from pv_mppt.boost_converter import BoostConverterInterface
from pv_mppt.dynamic_boost import (
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
from pv_mppt.en50530_benchmark import (
    DeltaPPO,
    EN50530BenchmarkResult,
    EN50530ControllerSummary,
    EN50530RampSegment,
    LumpedThermalModel,
    build_en50530_composite_profile,
    generate_trapezoidal_irradiance_profile,
    run_en50530_dynamic_benchmark,
)
from pv_mppt.global_mppt import (
    ControllerTrajectory,
    GlobalMPPTBenchmark,
    HybridGlobalScanINC,
    ParticleSwarmMPPT,
    run_dynamic_gmppt_benchmark,
)
from pv_mppt.mppt_algorithms import IncrementalConductance, PerturbAndObserve
from pv_mppt.partial_shading import (
    BypassDiodeSpec,
    PowerPeak,
    ShadedPVString,
    ShadingResult,
    get_benchmark_shading_profiles,
)
from pv_mppt.pv_model import SolarPVModel
from pv_mppt.web_dashboard import (
    FIGURE_FILENAMES,
    build_static_site,
    export_benchmark_payload,
)

__version__ = "1.0.1"

__all__ = [
    "__version__",
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
    "FIGURE_FILENAMES",
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
    "build_static_site",
    "compute_weighted_efficiencies",
    "evaluate_converter_efficiency_curve",
    "export_benchmark_payload",
    "generate_trapezoidal_irradiance_profile",
    "get_benchmark_shading_profiles",
    "run_closed_loop_boost_transient",
    "run_dynamic_gmppt_benchmark",
    "run_en50530_dynamic_benchmark",
]
