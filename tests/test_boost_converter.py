import pytest
from src.pv_model import SolarPVModel
from src.boost_converter import BoostConverterInterface

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
