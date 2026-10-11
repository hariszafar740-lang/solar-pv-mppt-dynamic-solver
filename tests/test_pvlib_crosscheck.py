"""Cross-verification of the repository solver against pvlib's independent
CEC single-diode implementation (``pvlib.pvsystem.calcparams_cec`` +
``singlediode``, which uses ``scipy.optimize.brentq`` -- a completely
different numerical method from this repository's Newton solver).

These tests run only when pvlib is installed (``pip install -r
requirements-dev.txt``); CI installs it, hermetic local runs may skip it.
The extracted model parameters are handed to pvlib in its own CEC
convention (Adjust = 0, matching this model's documented simplification),
so the two implementations must agree to solver precision.
"""

import numpy as np
import pytest

pvlib = pytest.importorskip("pvlib", reason="optional cross-check dependency")

from pvlib.pvsystem import calcparams_cec, i_from_v, singlediode  # noqa: E402

from src.pv_model import SolarPVModel  # noqa: E402

CONDITIONS = [
    (1000.0, 25.0),
    (1000.0, 75.0),
    (1000.0, -10.0),
    (900.0, 60.0),
    (600.0, 25.0),
    (400.0, 45.0),
    (200.0, 25.0),
    (50.0, 10.0),
    (1100.0, 30.0),
]


@pytest.fixture(scope="module")
def pv():
    return SolarPVModel()


def cec_kwargs(pv):
    return dict(
        alpha_sc=pv.K_i,
        a_ref=pv.a_ref,
        I_L_ref=pv.I_ph_ref,
        I_o_ref=pv.I_o_ref,
        R_sh_ref=pv.R_sh,
        R_s=pv.R_s,
        Adjust=0.0,  # documented simplification of this model
    )


@pytest.mark.parametrize("G,T_c", CONDITIONS)
def test_mpp_matches_pvlib(pv, G, T_c):
    _, _, p = pv.calculate_iv_curve(G=G, T_c_celsius=T_c, num_points=20001)
    p_max = float(np.max(p))
    params = calcparams_cec(G, T_c, **cec_kwargs(pv))
    sd = singlediode(*params)
    assert p_max == pytest.approx(float(sd["p_mp"]), abs=0.02)


@pytest.mark.parametrize("G,T_c", CONDITIONS)
def test_isc_and_voc_match_pvlib(pv, G, T_c):
    v, i, _ = pv.calculate_iv_curve(G=G, T_c_celsius=T_c, num_points=20001)
    params = calcparams_cec(G, T_c, **cec_kwargs(pv))
    sd = singlediode(*params)
    assert float(i[0]) == pytest.approx(float(sd["i_sc"]), abs=1e-4)
    positive = np.nonzero(i > 0.0)[0]
    voc_ours = float(v[positive[-1]])
    assert voc_ours == pytest.approx(float(sd["v_oc"]), abs=0.02)


@pytest.mark.parametrize("G,T_c", CONDITIONS)
def test_full_iv_curve_matches_pvlib(pv, G, T_c):
    """Point-by-point I-V comparison; pvlib evaluates the curve with the
    Lambert-W method (``i_from_v``), independent of both brentq and Newton.

    pvlib is evaluated directly on this model's own voltage grid (both are
    exact solvers of the same implicit equation at a given V), so the
    comparison is not polluted by interpolation error near the steep knee.
    """
    v, i, _ = pv.calculate_iv_curve(G=G, T_c_celsius=T_c, num_points=2001)
    params = calcparams_cec(G, T_c, **cec_kwargs(pv))
    i_pvlib = np.maximum(i_from_v(v, *params), 0.0)
    assert np.max(np.abs(i - i_pvlib)) < 1e-8


def test_pvlib_reproduces_record_stc_from_raw_cec_params():
    """The pinned CEC record's own fitted parameters, evaluated with pvlib,
    reproduce its published STC power (300.1152 W) -- the reference value
    used throughout this repair (historical local reconstruction: 300.1151 W)."""
    params = calcparams_cec(
        1000.0,
        25.0,
        alpha_sc=0.004162,
        a_ref=1.550532,
        I_L_ref=9.930718,
        I_o_ref=4.536912e-11,
        R_sh_ref=171.705124,
        R_s=0.358977,
        Adjust=8.923832,
    )
    sd = singlediode(*params)
    assert float(sd["p_mp"]) == pytest.approx(300.1152, abs=0.01)
    assert float(sd["v_mp"]) == pytest.approx(32.48, abs=0.01)
    assert float(sd["i_mp"]) == pytest.approx(9.24, abs=0.01)
