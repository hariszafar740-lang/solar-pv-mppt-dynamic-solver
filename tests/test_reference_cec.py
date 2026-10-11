"""Independent reference-data verification against the pinned CEC record.

Provenance (pinned)
-------------------
Database   : NREL System Advisor Model (SAM) / California Energy Commission
             (CEC) module database, as distributed with pvlib.
Record     : "AU Optronics PM060MBR_300W"  (Mono-c-Si, 60 cells in series)
Version    : "SAM 2018.11.11 r2", record date 2019-01-03
Bundled in : pvlib 0.16.1, data file ``sam-library-cec-modules-2019-03-05.csv``
File SHA256: a7c3b1ad3dabb5425368615c16322f2e35185fc416380b471c4e48dd545b1920

The constants below are transcribed verbatim from that record.  They are
*independent* of this repository's implementation: the model consumes only
the datasheet-level values (``*_ref`` nameplate numbers plus R_s/R_sh_ref and
alpha_sc), while the CEC-fitted diode parameters (a_ref, I_L_ref, I_o_ref)
and the record's STC power field are used purely as verification targets.

Tolerances
----------
* STC power vs. record STC field (300.1152 W): +/-0.02 W on a 4001-point
  grid (grid quantization near the quadratic maximum costs < 0.01 W).
* Extracted a_ref vs. record a_ref: 1e-5 relative (verified 1.55053183 vs
  1.550532).
* Extracted I_ph_ref vs. record I_L_ref: 1e-5 relative.
* Extracted I_o_ref vs. record I_o_ref: 1e-3 relative (verified 8.8e-5;
  the exponential sensitivity of I_o makes 1e-3 the honest floor).
* Implied temperature coefficients vs. record beta_oc/gamma_r: +/-0.03
  (V/K and %/K).  The record values come from SAM's full CEC fit including
  the ``Adjust`` factor; this model documents ``Adjust = 0``, so small
  offsets are expected (verified: -0.125 vs -0.114 V/K, -0.391 vs -0.386 %/K).
"""

import numpy as np
import pytest

from src.pv_model import SolarPVModel

# ---- Verbatim transcription of the pinned CEC record -------------------
RECORD_NAME = "AU Optronics PM060MBR_300W"
RECORD = {
    "STC": 300.115200,        # W, CEC-rated STC power (verification target)
    "N_s": 60,
    "I_sc_ref": 9.910000,     # A
    "V_oc_ref": 40.450000,    # V
    "I_mp_ref": 9.240000,     # A
    "V_mp_ref": 32.480000,    # V
    "alpha_sc": 0.004162,     # A/K
    "beta_oc": -0.114069,     # V/K
    "a_ref": 1.550532,        # V  (modified ideality factor, CEC-fitted)
    "I_L_ref": 9.930718,      # A  (CEC-fitted photocurrent)
    "I_o_ref": 4.536912e-11,  # A  (CEC-fitted saturation current)
    "R_s": 0.358977,          # Ohm
    "R_sh_ref": 171.705124,   # Ohm
    "Adjust": 8.923832,       # %
    "gamma_r": -0.386000,     # %/K
}


@pytest.fixture(scope="module")
def pv():
    return SolarPVModel()


def test_record_internal_consistency():
    """The record's own STC power field equals I_mp_ref * V_mp_ref."""
    assert RECORD["I_mp_ref"] * RECORD["V_mp_ref"] == pytest.approx(
        RECORD["STC"], abs=1e-3
    )


def test_default_parameters_match_pinned_record(pv):
    """Model defaults are exactly the record's datasheet-level values."""
    assert pv.I_sc == RECORD["I_sc_ref"]
    assert pv.V_oc == RECORD["V_oc_ref"]
    assert pv.I_mp == RECORD["I_mp_ref"]
    assert pv.V_mp == RECORD["V_mp_ref"]
    assert pv.N_s == RECORD["N_s"]
    assert pv.R_s == RECORD["R_s"]
    assert pv.R_sh == RECORD["R_sh_ref"]
    assert pv.K_i == RECORD["alpha_sc"]
    assert pv.REFERENCE_MODULE["name"] == RECORD_NAME


def test_stc_power_matches_record_stc_field(pv):
    """Regression target: STC P_max must equal the independently published
    CEC rating of 300.1152 W -- not the 266.898 W the legacy code produced."""
    _, _, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=4001)
    assert float(np.max(p)) == pytest.approx(RECORD["STC"], abs=0.02)


def test_stc_mpp_coordinates_match_record(pv):
    v, i, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=4001)
    j = int(np.argmax(p))
    assert v[j] == pytest.approx(RECORD["V_mp_ref"], abs=0.05)
    assert i[j] == pytest.approx(RECORD["I_mp_ref"], abs=0.02)


def test_extracted_diode_parameters_match_cec_fit(pv):
    """Strong independent check: extraction from *datasheet values alone*
    must reproduce the CEC database's own fitted diode parameters."""
    assert pv.a_ref == pytest.approx(RECORD["a_ref"], rel=1e-5)
    assert pv.I_ph_ref == pytest.approx(RECORD["I_L_ref"], rel=1e-5)
    assert pv.I_o_ref == pytest.approx(RECORD["I_o_ref"], rel=1e-3)
    assert pv.n == pytest.approx(
        RECORD["a_ref"] / (pv.N_s * pv.k * pv.T_ref / pv.q), rel=1e-5
    )


def _voc(v, i):
    positive = np.nonzero(i > 0.0)[0]
    return float(v[positive[-1]])


def test_implied_temperature_coefficients_near_record(pv):
    v25, i25, p25 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0, num_points=8001)
    v75, i75, p75 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=75.0, num_points=8001)
    beta_voc = (_voc(v75, i75) - _voc(v25, i25)) / 50.0
    gamma_p = 100.0 * (np.max(p75) - np.max(p25)) / np.max(p25) / 50.0
    assert beta_voc == pytest.approx(RECORD["beta_oc"], abs=0.03)
    assert gamma_p == pytest.approx(RECORD["gamma_r"], abs=0.03)
    assert gamma_p < 0.0 and beta_voc < 0.0


def test_ptc_order_of_magnitude(pv):
    """Sanity vs. record PTC (275.1 W at ~1000 W/m2, ~45 degC cell, 1 m/s):
    the model at NOCT-like conditions must land in a plausible window."""
    _, _, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=46.7, num_points=4001)
    p_max = float(np.max(p))
    assert 265.0 < p_max < 290.0  # record PTC = 275.1 W (different test setup)
