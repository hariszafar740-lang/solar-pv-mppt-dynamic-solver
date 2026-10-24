"""Series-string PV solver with anti-parallel Schottky bypass diodes under partial shading.

Physical formulation
--------------------
A crystalline-silicon PV string consists of ``M`` series-connected cell groups
(substrings), each protected by an anti-parallel bypass diode.  By default, a
single 60-cell reference module (`SolarPVModel`) comprises ``n_bypass = 3``
substrings of 20 cells each; a multi-module series string of ``n_modules``
panels comprises ``M = n_modules * n_bypass`` substrings.

Under non-uniform plane-of-array irradiance ``G_m`` and cell temperature
``T_m``, all series substrings carry the same string current ``I``, while the
total string voltage follows Kirchhoff's Voltage Law (KVL):

    V_string(I) = sum_{m=1..M} V_m(I)

For each substring ``m``, scaled single-diode parameters are derived from the
parent `SolarPVModel`:

    V_t,m   = V_t(T_m) / n_bypass
    R_s,m   = R_s / n_bypass
    R_sh,m  = R_sh(G_m) / n_bypass

When string current ``I`` exceeds a shaded substring's photocurrent, that
substring becomes reverse-biased (``V_m < 0``) and its anti-parallel Schottky
bypass diode conducts the excess current:

    I_bp(V_m) = I_o,bp * [exp(-V_m / V_t,bp) - 1]     for V_m < 0
              = 0                                     for V_m >= 0

Because ``I_bp(V_m) == 0`` identically for ``V_m >= 0``, uniform irradiance
reproduces the parent `SolarPVModel` I-V and P-V curves to solver precision
while non-uniform irradiance produces physical current-staircase I-V steps and
multi-peak P-V characteristics with distinct Global (GMPP) and Local (LMPP)
maximum power points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.pv_model import (
    MAX_OPERATING_TEMP_C,
    MIN_OPERATING_TEMP_C,
    SolarPVModel,
    _require_finite_real,
)

__all__ = [
    "BypassDiodeSpec",
    "PowerPeak",
    "ShadedPVString",
    "ShadingResult",
    "get_benchmark_shading_profiles",
]

_Q = 1.602176634e-19
_K = 1.380649e-23


@dataclass(frozen=True)
class BypassDiodeSpec:
    """Anti-parallel Schottky bypass diode electrical parameters."""

    i_sat_a: float = 2.0e-5
    ideality_factor: float = 1.25

    def __post_init__(self) -> None:
        i_sat = _require_finite_real(self.i_sat_a, "i_sat_a")
        n_bp = _require_finite_real(self.ideality_factor, "ideality_factor")
        if i_sat <= 0.0:
            raise ValueError(f"Bypass diode i_sat_a must be positive, got {i_sat}")
        if not (0.5 <= n_bp <= 3.0):
            raise ValueError(
                f"Bypass diode ideality_factor must lie in [0.5, 3.0], got {n_bp}"
            )


@dataclass(frozen=True)
class PowerPeak:
    """Detected local or global maximum power point on a P-V curve."""

    voltage_v: float
    current_a: float
    power_w: float
    is_global: bool


@dataclass
class ShadingResult:
    """Complete I-V / P-V solution and multi-peak diagnostics for a PV string."""

    voltages: np.ndarray
    currents: np.ndarray
    powers: np.ndarray
    irradiances: tuple[float, ...]
    temperatures_c: tuple[float, ...]
    v_oc_string: float
    i_sc_string: float
    gmpp: PowerPeak
    peaks: list[PowerPeak]
    active_bypass_count_at_gmpp: int
    ideal_uncoupled_power_w: float
    mismatch_loss_pct: float

    @property
    def num_peaks(self) -> int:
        return len(self.peaks)

    @property
    def highest_voltage_local_peak(self) -> PowerPeak:
        """Return the rightmost P-V peak (typical local trap for high-V MPPT start)."""
        return max(self.peaks, key=lambda pk: pk.voltage_v)

    def sample_at_voltage(self, v: float) -> tuple[float, float]:
        """Interpolate string ``(current_a, power_w)`` at operating voltage ``v``."""
        v_val = _require_finite_real(v, "v")
        if self.v_oc_string <= 0.0 or v_val <= 0.0 or v_val >= self.v_oc_string:
            if v_val == 0.0 and self.v_oc_string > 0.0:
                return float(self.i_sc_string), 0.0
            return 0.0, 0.0
        i_val = float(np.interp(v_val, self.voltages, self.currents))
        return i_val, v_val * i_val


class ShadedPVString:
    """Series-connected PV substring solver with anti-parallel bypass diodes."""

    def __init__(
        self,
        module: SolarPVModel | None = None,
        *,
        n_modules: int = 1,
        n_bypass_per_module: int = 3,
        bypass_diode: BypassDiodeSpec | None = None,
    ) -> None:
        self.module = module if module is not None else SolarPVModel()
        if isinstance(n_modules, (bool, np.bool_)) or not isinstance(
            n_modules, (int, np.integer)
        ) or n_modules < 1:
            raise ValueError(f"n_modules must be an integer >= 1, got {n_modules!r}")
        if isinstance(n_bypass_per_module, (bool, np.bool_)) or not isinstance(
            n_bypass_per_module, (int, np.integer)
        ) or n_bypass_per_module < 1:
            raise ValueError(
                "n_bypass_per_module must be an integer >= 1, "
                f"got {n_bypass_per_module!r}"
            )
        if self.module.N_s % int(n_bypass_per_module) != 0:
            raise ValueError(
                f"Module N_s={self.module.N_s} must be divisible by "
                f"n_bypass_per_module={n_bypass_per_module}"
            )

        self.n_modules = int(n_modules)
        self.n_bypass_per_module = int(n_bypass_per_module)
        self.n_substrings = self.n_modules * self.n_bypass_per_module
        self.bypass_diode = bypass_diode if bypass_diode is not None else BypassDiodeSpec()

    def _normalize_conditions(
        self,
        irradiances: Sequence[float],
        temperatures_c: float | Sequence[float] = 25.0,
    ) -> tuple[list[float], list[float]]:
        if isinstance(irradiances, (str, bytes)) or not hasattr(irradiances, "__len__"):
            raise ValueError("irradiances must be a sequence of length n_substrings")
        if len(irradiances) != self.n_substrings:
            raise ValueError(
                f"Expected {self.n_substrings} substring irradiances "
                f"({self.n_modules} module(s) x {self.n_bypass_per_module} bypass groups), "
                f"got {len(irradiances)}"
            )

        g_list: list[float] = []
        for idx, raw_g in enumerate(irradiances):
            g_val = _require_finite_real(raw_g, f"irradiances[{idx}]")
            if g_val < 0.0:
                raise ValueError(
                    f"irradiances[{idx}] must be non-negative, got {g_val}"
                )
            g_list.append(g_val)

        if isinstance(temperatures_c, (int, float, np.number)) and not isinstance(
            temperatures_c, (bool, np.bool_)
        ):
            t_val = _require_finite_real(temperatures_c, "temperatures_c")
            t_list = [t_val] * self.n_substrings
        else:
            if isinstance(temperatures_c, (str, bytes)) or not hasattr(
                temperatures_c, "__len__"
            ):
                raise ValueError(
                    "temperatures_c must be a scalar or sequence of length n_substrings"
                )
            if len(temperatures_c) != self.n_substrings:
                raise ValueError(
                    f"Expected {self.n_substrings} substring temperatures, "
                    f"got {len(temperatures_c)}"
                )
            t_list = [
                _require_finite_real(raw_t, f"temperatures_c[{idx}]")
                for idx, raw_t in enumerate(temperatures_c)
            ]

        for idx, t_val in enumerate(t_list):
            if not (MIN_OPERATING_TEMP_C <= t_val <= MAX_OPERATING_TEMP_C):
                raise ValueError(
                    f"temperatures_c[{idx}] must lie in "
                    f"[{MIN_OPERATING_TEMP_C}, {MAX_OPERATING_TEMP_C}] degC, got {t_val}"
                )

        return g_list, t_list

    def _substring_current_at_voltage(
        self,
        v_sub: np.ndarray,
        g_val: float,
        t_c: float,
    ) -> np.ndarray:
        """Evaluate total substring current ``I_cell(V_sub) + I_bp(V_sub)``."""
        t_k = t_c + 273.15
        nb = float(self.n_bypass_per_module)
        v_t_bp = self.bypass_diode.ideality_factor * _K * t_k / _Q
        i_sat_bp = self.bypass_diode.i_sat_a

        if g_val == 0.0:
            r_sh_sub = (self.module.R_sh * 1e6) / nb
            r_s_sub = self.module.R_s / nb
            i_cell = -v_sub / (r_s_sub + r_sh_sub)
        else:
            i_ph, i_o, v_t_mod, r_sh_mod = self.module._state_at(g_val, t_k)
            v_t_sub = v_t_mod / nb
            r_s_sub = self.module.R_s / nb
            r_sh_sub = r_sh_mod / nb

            i_cell = np.full(v_sub.shape, i_ph, dtype=np.float64)
            for _ in range(80):
                arg = np.clip((v_sub + i_cell * r_s_sub) / v_t_sub, -50.0, 50.0)
                exp_arg = np.exp(arg)
                f = (
                    i_ph
                    - i_o * (exp_arg - 1.0)
                    - (v_sub + i_cell * r_s_sub) / r_sh_sub
                    - i_cell
                )
                f_prime = -i_o * (r_s_sub / v_t_sub) * exp_arg - r_s_sub / r_sh_sub - 1.0
                delta = f / f_prime
                i_cell = i_cell - delta
                if np.max(np.abs(delta)) < 1e-12:
                    break

        bp_arg = np.clip(-v_sub / v_t_bp, -50.0, 60.0)
        i_bp = np.where(v_sub < 0.0, i_sat_bp * (np.exp(bp_arg) - 1.0), 0.0)
        return i_cell + i_bp

    def _invert_substring_voltage(
        self,
        i_targets: np.ndarray,
        g_val: float,
        t_c: float,
    ) -> np.ndarray:
        """Vectorized bisection solving ``I_sub(V_sub) = I_target`` for every target current."""
        nb = float(self.n_bypass_per_module)
        t_k = t_c + 273.15
        v_t_bp = self.bypass_diode.ideality_factor * _K * t_k / _Q
        max_i = float(np.max(i_targets)) + 1.0
        v_neg_bound = -v_t_bp * np.log(max_i / self.bypass_diode.i_sat_a + 1.0) - 0.25

        if g_val == 0.0:
            v_pos_bound = 0.05
        else:
            v_pos_bound = self.module.open_circuit_voltage_bound(g_val, t_c) / nb + 0.05

        lo = np.full(i_targets.shape, v_neg_bound, dtype=np.float64)
        hi = np.full(i_targets.shape, v_pos_bound, dtype=np.float64)

        for _ in range(62):
            mid = 0.5 * (lo + hi)
            i_mid = self._substring_current_at_voltage(mid, g_val, t_c)
            too_high_current = i_mid > i_targets
            lo = np.where(too_high_current, mid, lo)
            hi = np.where(too_high_current, hi, mid)
            if np.max(hi - lo) < 1e-11:
                break

        return 0.5 * (lo + hi)

    def solve_curve(
        self,
        irradiances: Sequence[float],
        temperatures_c: float | Sequence[float] = 25.0,
        *,
        num_points: int = 500,
    ) -> ShadingResult:
        """Compute full generating-quadrant I-V / P-V arrays and multi-peak telemetry."""
        if isinstance(num_points, (bool, np.bool_)) or not isinstance(
            num_points, (int, np.integer)
        ) or num_points < 20:
            raise ValueError(f"num_points must be an integer >= 20, got {num_points!r}")

        g_list, t_list = self._normalize_conditions(irradiances, temperatures_c)
        n_pts = int(num_points)

        if all(g == 0.0 for g in g_list):
            v_ref_max = sum(
                self.module.open_circuit_voltage_bound(self.module.G_ref, t_c)
                / self.n_bypass_per_module
                for t_c in t_list
            )
            v_arr = np.linspace(0.0, v_ref_max, n_pts)
            z_arr = np.zeros(n_pts, dtype=np.float64)
            zero_peak = PowerPeak(0.0, 0.0, 0.0, True)
            return ShadingResult(
                voltages=v_arr,
                currents=z_arr,
                powers=z_arr.copy(),
                irradiances=tuple(g_list),
                temperatures_c=tuple(t_list),
                v_oc_string=0.0,
                i_sc_string=0.0,
                gmpp=zero_peak,
                peaks=[zero_peak],
                active_bypass_count_at_gmpp=0,
                ideal_uncoupled_power_w=0.0,
                mismatch_loss_pct=0.0,
            )

        i_upper = max(
            float(
                self._substring_current_at_voltage(np.array([0.0]), g_val, t_c)[0]
            )
            for g_val, t_c in zip(g_list, t_list, strict=True)
        ) * 1.05 + 0.5

        lo_i, hi_i = 0.0, i_upper
        for _ in range(60):
            mid_i = 0.5 * (lo_i + hi_i)
            i_probe = np.array([mid_i])
            v_sum = sum(
                float(self._invert_substring_voltage(i_probe, g_val, t_c)[0])
                for g_val, t_c in zip(g_list, t_list, strict=True)
            )
            if v_sum > 0.0:
                lo_i = mid_i
            else:
                hi_i = mid_i
            if hi_i - lo_i < 1e-11:
                break
        i_sc_string = 0.5 * (lo_i + hi_i)

        i_sweep = np.linspace(i_sc_string, 0.0, n_pts)
        v_sub_matrix = np.vstack(
            [
                self._invert_substring_voltage(i_sweep, g_val, t_c)
                for g_val, t_c in zip(g_list, t_list, strict=True)
            ]
        )
        v_raw = np.sum(v_sub_matrix, axis=0)
        v_raw[0] = 0.0
        v_raw = np.maximum.accumulate(np.maximum(v_raw, 0.0))

        v_uniform = np.linspace(0.0, float(v_raw[-1]), n_pts)
        v_combined = np.unique(np.concatenate([v_raw, v_uniform]))
        i_interp = np.interp(v_combined, v_raw, i_sweep)

        v_grid = np.linspace(0.0, float(v_raw[-1]), n_pts)
        i_grid = np.interp(v_grid, v_combined, i_interp)
        p_grid = v_grid * i_grid

        peaks = _detect_pv_peaks(v_grid, i_grid, p_grid)
        gmpp = next(pk for pk in peaks if pk.is_global)

        v_sub_gmpp = [
            float(
                self._invert_substring_voltage(np.array([gmpp.current_a]), g_val, t_c)[0]
            )
            for g_val, t_c in zip(g_list, t_list, strict=True)
        ]
        bypassed_at_gmpp = sum(1 for v_m in v_sub_gmpp if v_m < -0.05)

        nb = float(self.n_bypass_per_module)
        ideal_power = 0.0
        for g_val, t_c in zip(g_list, t_list, strict=True):
            if g_val > 0.0:
                _, _, p_mod = self.module.calculate_iv_curve(
                    G=g_val, T_c_celsius=t_c, num_points=250
                )
                ideal_power += float(np.max(p_mod)) / nb

        mismatch_pct = (
            100.0 * max(0.0, ideal_power - gmpp.power_w) / ideal_power
            if ideal_power > 0.0
            else 0.0
        )

        return ShadingResult(
            voltages=v_grid,
            currents=i_grid,
            powers=p_grid,
            irradiances=tuple(g_list),
            temperatures_c=tuple(t_list),
            v_oc_string=float(v_grid[-1]),
            i_sc_string=float(i_sc_string),
            gmpp=gmpp,
            peaks=peaks,
            active_bypass_count_at_gmpp=bypassed_at_gmpp,
            ideal_uncoupled_power_w=ideal_power,
            mismatch_loss_pct=mismatch_pct,
        )


def _detect_pv_peaks(
    v_grid: np.ndarray,
    i_grid: np.ndarray,
    p_grid: np.ndarray,
    *,
    min_prominence_w: float = 1.5,
    min_separation_v: float = 2.0,
) -> list[PowerPeak]:
    """Identify local and global P-V maxima, filtering numerical discretization ripples."""
    candidate_indices: list[int] = []
    n = len(p_grid)
    for idx in range(1, n - 1):
        if p_grid[idx] >= p_grid[idx - 1] and p_grid[idx] >= p_grid[idx + 1]:
            if p_grid[idx] > min_prominence_w:
                candidate_indices.append(idx)

    if not candidate_indices:
        best_idx = int(np.argmax(p_grid))
        return [
            PowerPeak(
                voltage_v=float(v_grid[best_idx]),
                current_a=float(i_grid[best_idx]),
                power_w=float(p_grid[best_idx]),
                is_global=True,
            )
        ]

    merged: list[int] = []
    for idx in candidate_indices:
        if not merged:
            merged.append(idx)
            continue
        prev = merged[-1]
        if abs(float(v_grid[idx] - v_grid[prev])) < min_separation_v:
            if p_grid[idx] > p_grid[prev]:
                merged[-1] = idx
        else:
            merged.append(idx)

    filtered: list[int] = [merged[0]]
    for idx in merged[1:]:
        prev = filtered[-1]
        valley = float(np.min(p_grid[prev : idx + 1]))
        smaller_peak = float(min(p_grid[prev], p_grid[idx]))
        if smaller_peak - valley >= min_prominence_w:
            filtered.append(idx)
        elif p_grid[idx] > p_grid[prev]:
            filtered[-1] = idx

    global_idx = max(filtered, key=lambda k: float(p_grid[k]))
    return [
        PowerPeak(
            voltage_v=float(v_grid[k]),
            current_a=float(i_grid[k]),
            power_w=float(p_grid[k]),
            is_global=(k == global_idx),
        )
        for k in filtered
    ]


def get_benchmark_shading_profiles() -> dict[str, tuple[float, ...]]:
    """Return canonical 3-substring partial-shading benchmark irradiance patterns (W/m^2)."""
    return {
        "uniform_stc": (1000.0, 1000.0, 1000.0),
        "moderate_chimney_shade": (1000.0, 700.0, 350.0),
        "severe_cloud_edge": (850.0, 520.0, 200.0),
        "heavy_two_thirds_shade": (1000.0, 300.0, 300.0),
    }
