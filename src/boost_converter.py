"""DC-DC boost converter state interface.

Translates the PV module terminal voltage ``V_pv`` into the PWM duty ratio
``D`` required to hold a fixed DC-link voltage ``V_out`` (continuous
conduction mode), and applies the converter efficiency to the PV power.
"""

import numpy as np

__all__ = ["BoostConverterInterface"]


def _as_finite_float_array(value, name):
    """Validate a real, finite scalar-or-array input; return float ndarray."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a real number (or array of real numbers), got {value!r}"
        ) from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite real values, got {value!r}")
    return arr


class BoostConverterInterface:
    """DC-DC boost converter duty-cycle translator.

    Parameters
    ----------
    v_out : float
        Target DC-link (bus) voltage in volts; must satisfy ``v_out > 2`` so a
        valid duty-cycle window exists.
    efficiency : float
        Power conversion efficiency ``eta`` in the open interval ``(0, 1]``.
    """

    def __init__(self, v_out=60.0, efficiency=0.95):
        v_out = float(v_out)
        efficiency = float(efficiency)
        if not np.isfinite(v_out) or v_out <= 2.0:
            raise ValueError(f"v_out must be finite and > 2.0 V, got {v_out}")
        if not np.isfinite(efficiency) or not (0.0 < efficiency <= 1.0):
            raise ValueError(
                f"efficiency must be finite and within (0, 1], got {efficiency}"
            )
        self.v_out = v_out
        self.efficiency = efficiency

    def compute_duty_cycle(self, v_pv):
        """Ideal CCM duty cycle ``D = 1 - V_pv / V_out``.

        ``v_pv`` is first confined to the physically boostable window
        ``[1, V_out - 1]`` volts, then ``D`` is limited to ``[0.05, 0.95]``.
        Accepts a scalar or an array; raises ``ValueError`` on non-finite or
        complex input.
        """
        v_pv = _as_finite_float_array(v_pv, "v_pv")
        v_pv_clamped = np.clip(v_pv, 1.0, self.v_out - 1.0)
        duty_cycle = 1.0 - (v_pv_clamped / self.v_out)
        duty_cycle = np.clip(duty_cycle, 0.05, 0.95)
        if duty_cycle.ndim == 0:
            return float(duty_cycle)
        return duty_cycle

    def compute_output_power(self, p_pv):
        """Net converter power delivered to the DC bus: ``P_out = eta * P_pv``.

        Raises ``ValueError`` on non-finite, complex, or negative input power
        (a PV source cannot deliver negative power through the converter).
        """
        p_pv = _as_finite_float_array(p_pv, "p_pv")
        if np.any(p_pv < 0.0):
            raise ValueError(
                f"p_pv must be non-negative (W); got {p_pv!r}. Negative PV "
                "power is non-physical for this converter interface."
            )
        out = p_pv * self.efficiency
        if out.ndim == 0:
            return float(out)
        return out
