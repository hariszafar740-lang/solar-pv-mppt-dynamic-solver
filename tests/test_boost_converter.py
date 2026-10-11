"""Tests for the DC-DC boost converter interface and thermal derating."""

import numpy as np
import pytest

from src.boost_converter import BoostConverterInterface
from src.pv_model import SolarPVModel


def test_boost_duty_cycle_math():
    converter = BoostConverterInterface(v_out=60.0)

    # At V_pv = 30V -> D = 1 - (30/60) = 0.50
    d = converter.compute_duty_cycle(30.0)
    assert pytest.approx(d, abs=1e-2) == 0.50


def test_thermal_power_derating():
    pv = SolarPVModel()
    _, _, p_25 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0)
    _, _, p_75 = pv.calculate_iv_curve(G=1000.0, T_c_celsius=75.0)

    # Increasing cell temperature must reduce maximum power output
    assert max(p_25) > max(p_75)


def test_duty_cycle_clamping():
    converter = BoostConverterInterface(v_out=60.0)
    # V_pv below the 1 V window -> D limited to 0.95
    assert converter.compute_duty_cycle(0.0) == pytest.approx(0.95)
    assert converter.compute_duty_cycle(-50.0) == pytest.approx(0.95)
    # V_pv near/above V_out -> D limited to 0.05
    assert converter.compute_duty_cycle(59.5) == pytest.approx(0.05)
    assert converter.compute_duty_cycle(120.0) == pytest.approx(0.05)


def test_duty_cycle_array_input():
    converter = BoostConverterInterface(v_out=60.0)
    d = converter.compute_duty_cycle(np.array([0.5, 30.0, 59.0, 200.0]))
    assert d.shape == (4,)
    assert np.all(np.isfinite(d))
    assert np.all((d >= 0.05) & (d <= 0.95))


def test_output_power():
    converter = BoostConverterInterface(v_out=60.0, efficiency=0.95)
    assert converter.compute_output_power(300.0) == pytest.approx(285.0)
    assert converter.compute_output_power(0.0) == pytest.approx(0.0)
    out = converter.compute_output_power(np.array([0.0, 100.0, 300.0]))
    assert np.allclose(out, [0.0, 95.0, 285.0])


@pytest.mark.parametrize("p_bad", [-1.0, -1e-9, float("nan"), float("inf")])
def test_negative_or_nonfinite_power_rejected(p_bad):
    converter = BoostConverterInterface(v_out=60.0)
    with pytest.raises(ValueError):
        converter.compute_output_power(p_bad)


@pytest.mark.parametrize("v_bad", [float("nan"), float("inf"), complex(30, 0)])
def test_invalid_voltage_rejected(v_bad):
    converter = BoostConverterInterface(v_out=60.0)
    with pytest.raises(ValueError):
        converter.compute_duty_cycle(v_bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(v_out=0.0),
        dict(v_out=-60.0),
        dict(v_out=2.0),
        dict(v_out=float("nan")),
        dict(efficiency=0.0),
        dict(efficiency=1.5),
        dict(efficiency=-0.9),
        dict(efficiency=float("inf")),
    ],
)
def test_invalid_construction_rejected(kwargs):
    with pytest.raises(ValueError):
        BoostConverterInterface(**kwargs)
