"""Physics-based single-diode solar photovoltaic (PV) module solver.

Model formulation
-----------------
The module solves the implicit single-diode equation for current ``I`` at each
terminal voltage ``V``:

    I = I_ph - I_0 * [exp((V + I*R_s) / V_t) - 1] - (V + I*R_s) / R_sh

with the module thermal voltage ``V_t = n * N_s * k * T / q``.

Reference-parameter extraction
------------------------------
The five model parameters ``(I_ph_ref, I_o_ref, n)`` are *derived*, not
guessed.  Given datasheet values at standard test conditions (STC:
1000 W/m^2, 25 degC) -- ``I_sc``, ``V_oc``, ``I_mp``, ``V_mp`` -- plus the
resistances ``R_s`` and ``R_sh``, the constructor solves the three STC
constraints

    I(V = 0)    = I_sc
    I(V = V_oc) = 0
    I(V = V_mp) = I_mp

exactly.  For a fixed ideality factor ``n`` the first two constraints are
linear in ``(I_ph_ref, I_o_ref)``; the third constraint is then a scalar
root-finding problem in ``n`` (bracketed bisection).  If the datasheet values
are mutually inconsistent with the single-diode model for any *physical*
ideality factor (``0.5 <= n <= 2.5``), the constructor raises ``ValueError``
instead of silently returning a curve that misses the nameplate maximum power
point.  This was the root cause of the historical STC defect (nameplate
300.08 W vs. computed 266.90 W): the previous implementation stored
``I_mp``/``V_mp`` but never used them, fixing ``n = 1.3`` by hand and
deriving ``I_0`` from ``V_oc`` alone.

Temperature / irradiance scaling
--------------------------------
The scaling follows the De Soto et al. (2006) / CEC ("six-parameter")
convention used by pvlib's ``calcparams_cec``:

    V_t(T)  = n * N_s * k * T / q                       (n constant)
    I_ph    = (I_ph_ref + K_i*(T - T_ref)) * (G/G_ref)
    I_0(T)  = I_o_ref * (T/T_ref)^3
              * exp(E_g,ref/(k_eV*T_ref) - E_g(T)/(k_eV*T))
    E_g(T)  = E_g,ref * (1 + dE_gdT * (T - T_ref))      (E_g,ref = 1.121 eV)
    R_sh(G) = R_sh_ref * (G_ref/G)                      (R_s constant)

Known simplification vs. pvlib/SAM: the CEC ``Adjust`` factor is taken as 0,
i.e. ``K_i`` is applied unadjusted to the photocurrent.

Validity envelope (validated on every call)
-------------------------------------------
* Irradiance ``G``: real, finite, ``0 <= G``; ``G < 0`` raises ``ValueError``
  (it previously produced complex-valued voltage/power arrays).  ``G == 0``
  is valid: the module generates nothing, so zero current/power is returned
  on a well-formed voltage grid spanning ``[0, V_oc_bound(T, G_ref)]``.
* Cell temperature ``T``: real, finite, ``-40 degC <= T <= +85 degC``
  (typical crystalline-silicon module operating envelope).  Absolute zero
  previously crashed with ``ZeroDivisionError``.
* ``num_points``: integer >= 2.

All returned arrays are real-valued ``float64`` and finite.  Current is
reported in the generating quadrant only (values beyond ``V_oc``, where the
implicit equation yields negative current, are clamped to 0).

Reference module
----------------
Default constructor values are the datasheet-level parameters of the
AU Optronics PM060MBR_300W monocrystalline 60-cell module as published in
the NREL SAM / California Energy Commission (CEC) module database, record
version "SAM 2018.11.11 r2" (dated 2019-01-03), exactly as bundled with
pvlib 0.16.1 (``sam-library-cec-modules-2019-03-05.csv``).  See README.md and
tests/test_reference_cec.py for the full provenance and verification record.
The identity of the module that the original hard-coded defaults were meant
to represent could not be established; those defaults were mutually
inconsistent with the single-diode model at any physical ideality factor and
are rejected by this constructor.
"""

import numpy as np

__all__ = ["SolarPVModel"]

# Physical constants (CODATA 2018 exact SI values)
_Q = 1.602176634e-19    # elementary charge (C)
_K = 1.380649e-23       # Boltzmann constant (J/K)
_K_EV = _K / _Q         # Boltzmann constant (eV/K) = 8.617333262e-5

# De Soto / CEC crystalline-silicon band-gap model constants (pvlib defaults)
_EG_REF_EV = 1.121      # Si band-gap energy at T_ref (eV)
_DEG_DT = -0.0002677    # band-gap temperature coefficient (eV/K... 1/K factor)

# Documented physical brackets for the extracted diode ideality factor
_N_MIN_PHYSICAL = 0.5
_N_MAX_PHYSICAL = 2.5
_N_SCAN_MIN = 0.30
_N_SCAN_MAX = 3.00

# Supported operating envelope (deg C)
MIN_OPERATING_TEMP_C = -40.0
MAX_OPERATING_TEMP_C = 85.0


def _require_finite_real(value, name):
    """Return ``float(value)`` or raise ``ValueError`` for non-real input."""
    if isinstance(value, (bool, complex)) or isinstance(
        value, (np.bool_, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


class SolarPVModel:
    """Physics-based single-diode solar photovoltaic module solver.

    The default parameters are the STC datasheet values of the AU Optronics
    PM060MBR_300W module from the pinned SAM/CEC database record (see the
    module docstring and README.md for provenance).  The diode parameters
    ``(I_ph_ref, I_o_ref, n)`` are extracted at construction time so that the
    model reproduces the datasheet ``I_sc``, ``V_oc`` and maximum power point
    ``(V_mp, I_mp)`` exactly at STC.

    Parameters
    ----------
    I_sc, V_oc, I_mp, V_mp : float
        Datasheet short-circuit current (A), open-circuit voltage (V), and
        maximum-power-point current/voltage (A, V) at STC.
    N_s : int
        Number of series-connected cells.
    R_s : float
        Series resistance (Ohm), assumed temperature independent.
    R_sh : float
        Shunt resistance (Ohm) *at reference irradiance* (CEC convention);
        scaled as ``R_sh(G) = R_sh * G_ref / G`` at other irradiance levels.
    n : float
        Initial guess for the diode ideality factor.  The value actually used
        is re-extracted from the datasheet constraints; after construction
        ``self.n`` holds the extracted factor.
    K_i : float
        Temperature coefficient of the short-circuit/photocurrent (A/K),
        applied as ``alpha_sc`` in the CEC convention (``Adjust = 0``).

    Raises
    ------
    ValueError
        If any input is non-finite or non-physical, or if the datasheet values
        are mutually inconsistent with the single-diode model for any physical
        ideality factor in ``[0.5, 2.5]``.
    """

    # Provenance of the default reference module (pinned record)
    REFERENCE_MODULE = {
        "name": "AU Optronics PM060MBR_300W",
        "technology": "Mono-c-Si",
        "database": "NREL SAM / CEC module database (CECMod)",
        "record_version": "SAM 2018.11.11 r2",
        "record_date": "2019-01-03",
        "bundled_with": "pvlib 0.16.1 (sam-library-cec-modules-2019-03-05.csv)",
        "dataset_sha256": (
            "a7c3b1ad3dabb5425368615c16322f2e35185fc416380b471c4e48dd545b1920"
        ),
        "cec_stc_power_W": 300.1152,
    }

    def __init__(self, I_sc=9.91, V_oc=40.45, I_mp=9.24, V_mp=32.48,
                 N_s=60, R_s=0.358977, R_sh=171.705124, n=1.0, K_i=0.004162):
        self.I_sc = _require_finite_real(I_sc, "I_sc")
        self.V_oc = _require_finite_real(V_oc, "V_oc")
        self.I_mp = _require_finite_real(I_mp, "I_mp")
        self.V_mp = _require_finite_real(V_mp, "V_mp")
        self.R_s = _require_finite_real(R_s, "R_s")
        self.R_sh = _require_finite_real(R_sh, "R_sh")
        self.K_i = _require_finite_real(K_i, "K_i")
        self.n_guess = _require_finite_real(n, "n")

        if isinstance(N_s, (bool, np.bool_)):
            raise ValueError(f"N_s must be an integer >= 1, got {N_s!r}")
        if not isinstance(N_s, (int, np.integer)) or N_s < 1:
            raise ValueError(f"N_s must be an integer >= 1, got {N_s!r}")
        self.N_s = int(N_s)

        if self.I_sc <= 0.0:
            raise ValueError(f"I_sc must be positive, got {self.I_sc}")
        if self.V_oc <= 0.0:
            raise ValueError(f"V_oc must be positive, got {self.V_oc}")
        if not (0.0 < self.I_mp < self.I_sc):
            raise ValueError(
                f"I_mp must satisfy 0 < I_mp < I_sc, got I_mp={self.I_mp}, "
                f"I_sc={self.I_sc}"
            )
        if not (0.0 < self.V_mp < self.V_oc):
            raise ValueError(
                f"V_mp must satisfy 0 < V_mp < V_oc, got V_mp={self.V_mp}, "
                f"V_oc={self.V_oc}"
            )
        if self.R_s < 0.0:
            raise ValueError(f"R_s must be non-negative, got {self.R_s}")
        if self.R_sh <= 0.0:
            raise ValueError(f"R_sh must be positive, got {self.R_sh}")
        if not (_N_SCAN_MIN < self.n_guess <= _N_SCAN_MAX):
            raise ValueError(
                f"Initial ideality-factor guess n must lie in "
                f"({_N_SCAN_MIN}, {_N_SCAN_MAX}], got {self.n_guess}"
            )

        # Physical constants (public attributes preserved from prior API)
        self.q = _Q
        self.k = _K
        self.T_ref = 298.15        # STC temperature (25 degC in Kelvin)
        self.G_ref = 1000.0        # STC irradiance (W/m^2)

        # Self-consistent extraction of (I_ph_ref, I_o_ref, n) from the
        # datasheet constraints.  Sets self.n, self.I_ph_ref, self.I_o_ref,
        # self.a_ref.
        self._extract_reference_parameters()

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------
    def _diode_state_from_n(self, n):
        """Solve the two linear STC constraints for (I_ph_ref, I_o_ref).

        Uses I(V=0) = I_sc and I(V=V_oc) = 0 at fixed thermal voltage
        ``V_t = n * N_s * k * T_ref / q``.
        """
        v_t = n * self.N_s * _K * self.T_ref / _Q
        exp_sc = np.expm1(self.I_sc * self.R_s / v_t)
        exp_oc = np.expm1(self.V_oc / v_t)
        denom = exp_oc - exp_sc
        if denom <= 0.0 or not np.isfinite(denom):
            return None, None
        rhs = self.I_sc * (1.0 + self.R_s / self.R_sh) - self.V_oc / self.R_sh
        i_o = rhs / denom
        i_ph = self.V_oc / self.R_sh + i_o * exp_oc
        return i_ph, i_o

    def _mpp_residual(self, n):
        """Current misfit at the datasheet MPP for a candidate ideality n."""
        i_ph, i_o = self._diode_state_from_n(n)
        if i_ph is None or i_o <= 0.0 or i_ph <= 0.0:
            return -np.inf
        v_t = n * self.N_s * _K * self.T_ref / _Q
        v_diode = self.V_mp + self.I_mp * self.R_s
        i_model = i_ph - i_o * np.expm1(v_diode / v_t) - v_diode / self.R_sh
        return i_model - self.I_mp

    def _extract_reference_parameters(self):
        """Find the ideality factor satisfying all three STC constraints."""
        n_grid = np.linspace(_N_SCAN_MIN, _N_SCAN_MAX, 271)
        residuals = np.array([self._mpp_residual(n) for n in n_grid])
        crossings = [
            j for j in range(len(n_grid) - 1)
            if np.isfinite(residuals[j]) and np.isfinite(residuals[j + 1])
            and residuals[j] * residuals[j + 1] <= 0.0
        ]
        if not crossings:
            raise ValueError(
                "Datasheet values (I_sc, V_oc, I_mp, V_mp, R_s, R_sh) are "
                "mutually inconsistent: the single-diode equation has no "
                "solution passing through the specified maximum power point "
                f"for any ideality factor in [{_N_SCAN_MIN}, {_N_SCAN_MAX}]."
            )
        lo, hi = n_grid[crossings[0]], n_grid[crossings[0] + 1]
        f_lo = self._mpp_residual(lo)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = self._mpp_residual(mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
            if (hi - lo) <= 1e-14 * max(1.0, abs(hi)):
                break
        n_root = 0.5 * (lo + hi)
        if not (_N_MIN_PHYSICAL <= n_root <= _N_MAX_PHYSICAL):
            raise ValueError(
                "Datasheet values (I_sc, V_oc, I_mp, V_mp, R_s, R_sh) are "
                "mutually inconsistent with the single-diode model: they "
                f"require the non-physical ideality factor n = {n_root:.4f} "
                f"(physical range [{_N_MIN_PHYSICAL}, {_N_MAX_PHYSICAL}]). "
                "The nameplate maximum power point cannot be reproduced by "
                "any physically valid parameterization; refusing to build a "
                "model that would silently under-predict STC power."
            )
        if abs(self._mpp_residual(n_root)) > 1e-8:
            raise ValueError(
                "Reference-parameter extraction failed to converge to the "
                "datasheet maximum power point (residual "
                f"{self._mpp_residual(n_root):.3e} A)."
            )
        i_ph, i_o = self._diode_state_from_n(n_root)
        self.n = float(n_root)
        self.I_ph_ref = float(i_ph)
        self.I_o_ref = float(i_o)
        # Modified ideality factor a = n*N_s*k*T_ref/q (pvlib/CEC "a_ref", V)
        self.a_ref = float(self.n * self.N_s * _K * self.T_ref / _Q)

    # ------------------------------------------------------------------
    # Environmental state
    # ------------------------------------------------------------------
    def _state_at(self, G, T_k):
        """Return (I_ph, I_o, v_t, R_sh) at irradiance G and temperature T_k."""
        v_t = self.a_ref * (T_k / self.T_ref)
        e_g = _EG_REF_EV * (1.0 + _DEG_DT * (T_k - self.T_ref))
        i_o = (
            self.I_o_ref
            * (T_k / self.T_ref) ** 3
            * np.exp(_EG_REF_EV / (_K_EV * self.T_ref) - e_g / (_K_EV * T_k))
        )
        i_ph = (self.I_ph_ref + self.K_i * (T_k - self.T_ref)) * (G / self.G_ref)
        r_sh = self.R_sh * (self.G_ref / G) if G > 0.0 else np.inf
        return i_ph, i_o, v_t, r_sh

    def open_circuit_voltage_bound(self, G, T_c_celsius):
        """Upper bound on the model open-circuit voltage (V).

        ``v_t * ln(I_ph/I_o + 1)`` neglects the shunt current and is therefore
        a strict upper bound on the true model ``V_oc`` at the same (G, T).
        """
        G = _require_finite_real(G, "G")
        T_c = _require_finite_real(T_c_celsius, "T_c_celsius")
        if G < 0.0:
            raise ValueError(f"Irradiance G must be non-negative, got {G}")
        if not (MIN_OPERATING_TEMP_C <= T_c <= MAX_OPERATING_TEMP_C):
            raise ValueError(
                f"Cell temperature must be within [{MIN_OPERATING_TEMP_C}, "
                f"{MAX_OPERATING_TEMP_C}] degC, got {T_c}"
            )
        T_k = T_c + 273.15
        i_ph_stc_g, i_o, v_t, _ = self._state_at(self.G_ref, T_k)
        if G == 0.0:
            # No generation: report the reference-irradiance bound so callers
            # always receive a usable, monotonic voltage axis.
            return float(v_t * np.log(i_ph_stc_g / i_o + 1.0))
        i_ph, i_o, v_t, _ = self._state_at(G, T_k)
        return float(v_t * np.log(max(i_ph, 0.0) / i_o + 1.0))

    # ------------------------------------------------------------------
    # I-V curve solver
    # ------------------------------------------------------------------
    def calculate_iv_curve(self, G=1000.0, T_c_celsius=25.0, num_points=200):
        """Solve the I-V and P-V characteristics at given conditions.

        Parameters
        ----------
        G : float
            Plane-of-array irradiance in W/m^2.  Must be finite and >= 0.
            ``G == 0`` returns zero current and power on a well-formed grid.
        T_c_celsius : float
            Cell temperature in degC, within [-40, +85] degC.
        num_points : int
            Number of voltage grid points (>= 2).

        Returns
        -------
        v_array, i_array, p_array : numpy.ndarray
            Real-valued, finite ``float64`` arrays.  ``v_array`` spans
            ``[0, V_oc_bound(G, T)]``; ``i_array`` is the generating-quadrant
            current (clamped at 0 beyond ``V_oc``); ``p_array = v_array *
            i_array``.

        Raises
        ------
        ValueError
            On negative, non-finite, or non-real ``G``/``T_c_celsius``, on
            temperatures outside the supported operating envelope, or on an
            invalid ``num_points``.
        """
        G = _require_finite_real(G, "G")
        T_c = _require_finite_real(T_c_celsius, "T_c_celsius")
        if isinstance(num_points, (bool, np.bool_)) or not isinstance(
            num_points, (int, np.integer)
        ):
            raise ValueError(
                f"num_points must be an integer >= 2, got {num_points!r}"
            )
        if num_points < 2:
            raise ValueError(f"num_points must be an integer >= 2, got {num_points}")
        if G < 0.0:
            raise ValueError(
                f"Irradiance G must be non-negative (W/m^2), got {G}. "
                "Negative irradiance is non-physical."
            )
        if not (MIN_OPERATING_TEMP_C <= T_c <= MAX_OPERATING_TEMP_C):
            raise ValueError(
                f"Cell temperature must be within [{MIN_OPERATING_TEMP_C}, "
                f"{MAX_OPERATING_TEMP_C}] degC, got {T_c}"
            )

        T_k = T_c + 273.15
        n_pts = int(num_points)

        if G == 0.0:
            # Zero irradiance: the module generates no power.  Return a
            # monotonic voltage axis (bounded by the reference-irradiance
            # open-circuit bound at this temperature) with zero current so
            # downstream interpolation remains well defined.
            v_bound = self.open_circuit_voltage_bound(self.G_ref, T_c)
            v_array = np.linspace(0.0, v_bound, n_pts)
            i_array = np.zeros(n_pts)
            return v_array, i_array, v_array * i_array

        i_ph, i_o, v_t, r_sh = self._state_at(G, T_k)
        v_max = v_t * np.log(i_ph / i_o + 1.0)
        v_array = np.linspace(0.0, v_max, n_pts)
        i_array = self._solve_current(v_array, i_ph, i_o, v_t, r_sh)
        return v_array, i_array, v_array * i_array

    def _solve_current(self, v, i_ph, i_o, v_t, r_sh):
        """Vectorized Newton solve of the implicit single-diode equation.

        ``f(I) = i_ph - i_o*exp((V + I*Rs)/v_t) + i_o - (V + I*Rs)/Rsh - I``
        is strictly decreasing and concave in ``I`` (``f' < 0``, ``f'' < 0``),
        and ``f(i_ph) < 0``; Newton's method started at ``I = i_ph`` therefore
        converges monotonically from above without overshoot.  A bisection
        fallback on a guaranteed bracket handles any point that fails to
        converge (defensive; not exercised for valid parameters).
        """
        current = np.full(v.shape, i_ph, dtype=np.float64)
        converged = False
        for _ in range(100):
            arg = np.clip((v + current * self.R_s) / v_t, -50.0, 50.0)
            exp_arg = np.exp(arg)
            f = i_ph - i_o * (exp_arg - 1.0) - (v + current * self.R_s) / r_sh - current
            f_prime = -i_o * (self.R_s / v_t) * exp_arg - self.R_s / r_sh - 1.0
            delta = f / f_prime
            current = current - delta
            if np.max(np.abs(delta)) < 1e-12:
                converged = True
                break

        if not converged or not np.all(np.isfinite(current)):
            current = self._bisect_current(v, i_ph, i_o, v_t, r_sh, current)

        # Generating quadrant only: beyond V_oc the implicit solution is
        # negative (the module would consume power); report 0 A there.
        return np.maximum(current, 0.0)

    def _bisect_current(self, v, i_ph, i_o, v_t, r_sh, initial):
        """Bisection fallback with a guaranteed sign bracket per point."""
        hi = np.full(v.shape, i_ph, dtype=np.float64)     # f(i_ph) < 0
        lo = -(np.abs(v) / r_sh + i_ph + i_o + 1.0)       # f(lo) > 0

        def f(cur):
            arg = np.clip((v + cur * self.R_s) / v_t, -50.0, 50.0)
            return (
                i_ph - i_o * (np.exp(arg) - 1.0)
                - (v + cur * self.R_s) / r_sh - cur
            )

        f_lo = f(lo)
        # Expand the lower bracket defensively (should not trigger).
        for _ in range(60):
            bad = f_lo <= 0.0
            if not np.any(bad):
                break
            lo = np.where(bad, 2.0 * lo - 1.0, lo)
            f_lo = f(lo)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            go_right = f_mid < 0.0
            hi = np.where(go_right, mid, hi)
            lo = np.where(go_right, lo, mid)
            if np.max(hi - lo) < 1e-12:
                break
        result = 0.5 * (lo + hi)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Single-diode current solve failed to produce finite values."
            )
        return result
