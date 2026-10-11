"""MPPT control algorithms: Perturb & Observe and Incremental Conductance.

Both controllers expose the same call signature as before
(``step(v_curr, measurement) -> v_next``) and remain stateless with respect to
the PV model: the caller measures the operating point and feeds it in.

An optional voltage window ``[v_min, v_max]`` can now be supplied so the
reference voltage can never be driven outside the physically valid operating
range of the array (e.g. negative voltages or voltages beyond ``V_oc``).
With the defaults (``None``) the algorithms are unbounded, exactly as before.
"""

import numpy as np

__all__ = ["PerturbAndObserve", "IncrementalConductance"]


def _validate_step_size(step_size):
    step_size = float(step_size)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError(f"step_size must be a positive finite number, got {step_size}")
    return step_size


def _validate_optional_bound(value, name):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite or None, got {value}")
    return value


class _BoundedMPPTController:
    """Shared bound handling and validation for MPPT controllers."""

    def __init__(self, step_size, v_init, v_min, v_max):
        self.step_size = _validate_step_size(step_size)
        v_init = float(v_init)
        if not np.isfinite(v_init):
            raise ValueError(f"v_init must be finite, got {v_init}")
        self.v_min = _validate_optional_bound(v_min, "v_min")
        self.v_max = _validate_optional_bound(v_max, "v_max")
        if (
            self.v_min is not None
            and self.v_max is not None
            and self.v_min >= self.v_max
        ):
            raise ValueError(
                f"v_min must be < v_max, got v_min={self.v_min}, v_max={self.v_max}"
            )
        self.v_prev = self._clamp(v_init)

    def _clamp(self, v):
        """Confine a voltage reference to the configured operating window."""
        if self.v_min is not None:
            v = max(v, self.v_min)
        if self.v_max is not None:
            v = min(v, self.v_max)
        return v


class PerturbAndObserve(_BoundedMPPTController):
    """Perturb and Observe (P&O) MPPT control engine.

    Iteratively perturbs the operating voltage and evaluates the power
    differential ``dP``: if power increased, the perturbation direction is
    kept; otherwise it is reversed.  At the maximum power point the controller
    oscillates within +/- ``step_size`` of ``V_mpp`` (inherent to P&O).

    Parameters
    ----------
    step_size : float
        Voltage perturbation magnitude per control step (V), > 0.
    v_init : float
        Initial previous-voltage state (V).
    v_min, v_max : float or None
        Optional operating window (V); returned references are clamped to it.
    """

    def __init__(self, step_size=0.15, v_init=25.0, v_min=None, v_max=None):
        super().__init__(step_size, v_init, v_min, v_max)
        self.p_prev = 0.0

    def step(self, v_curr, p_curr):
        """Advance one control step. Returns the next voltage reference (V)."""
        v_curr = float(v_curr)
        p_curr = float(p_curr)
        if not np.isfinite(v_curr) or not np.isfinite(p_curr):
            raise ValueError(
                f"v_curr and p_curr must be finite, got v={v_curr}, p={p_curr}"
            )
        delta_p = p_curr - self.p_prev
        delta_v = v_curr - self.v_prev

        if delta_p == 0:
            v_next = v_curr
        elif delta_p > 0:
            if delta_v >= 0:
                v_next = v_curr + self.step_size
            else:
                v_next = v_curr - self.step_size
        else:
            if delta_v >= 0:
                v_next = v_curr - self.step_size
            else:
                v_next = v_curr + self.step_size

        self.v_prev = v_curr
        self.p_prev = p_curr
        return self._clamp(v_next)


class IncrementalConductance(_BoundedMPPTController):
    """Incremental Conductance (INC) MPPT control engine.

    Compares the instantaneous conductance ``-I/V`` with the incremental
    conductance ``dI/dV`` to decide the perturbation direction; at the maximum
    power point the two are equal, which (unlike P&O) allows the controller to
    stop perturbing in steady state.

    Parameters
    ----------
    step_size : float
        Voltage perturbation magnitude per control step (V), > 0.
    v_init : float
        Initial previous-voltage state (V).
    v_min, v_max : float or None
        Optional operating window (V); returned references are clamped to it.
    conductance_tol : float
        Absolute Siemens tolerance used to declare ``dI/dV == -I/V`` (i.e. the
        MPP has been reached).  Default 1e-2 S preserves the historical
        behavior.
    """

    def __init__(self, step_size=0.15, v_init=25.0, v_min=None, v_max=None,
                 conductance_tol=1e-2):
        super().__init__(step_size, v_init, v_min, v_max)
        conductance_tol = float(conductance_tol)
        if not np.isfinite(conductance_tol) or conductance_tol <= 0.0:
            raise ValueError(
                f"conductance_tol must be positive and finite, got {conductance_tol}"
            )
        self.conductance_tol = conductance_tol
        self.i_prev = 0.0

    def step(self, v_curr, i_curr):
        """Advance one control step. Returns the next voltage reference (V)."""
        v_curr = float(v_curr)
        i_curr = float(i_curr)
        if not np.isfinite(v_curr) or not np.isfinite(i_curr):
            raise ValueError(
                f"v_curr and i_curr must be finite, got v={v_curr}, i={i_curr}"
            )
        delta_v = v_curr - self.v_prev
        delta_i = i_curr - self.i_prev

        if delta_v == 0:
            if delta_i == 0:
                v_next = v_curr
            elif delta_i > 0:
                v_next = v_curr + self.step_size
            else:
                v_next = v_curr - self.step_size
        else:
            inc_cond = delta_i / delta_v
            inst_cond = -i_curr / v_curr if v_curr > 0 else 0.0

            if np.isclose(inc_cond, inst_cond, atol=self.conductance_tol):
                v_next = v_curr
            elif inc_cond > inst_cond:
                v_next = v_curr + self.step_size
            else:
                v_next = v_curr - self.step_size

        self.v_prev = v_curr
        self.i_prev = i_curr
        return self._clamp(v_next)
