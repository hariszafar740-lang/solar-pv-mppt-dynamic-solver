import pytest
import numpy as np
from src.pv_model import SolarPVModel

def test_pv_stc_boundaries():
    pv = SolarPVModel()
    v, i, p = pv.calculate_iv_curve(G=1000.0, T_c_celsius=25.0)
    
    # Short-circuit current test (at V = 0)
    assert np.isclose(i[0], pv.I_sc, rtol=1e-2)
    
    # Open-circuit current test (at V max, I should be ~0)
    assert i[-1] <= 0.1
    
    # Power positivity test
    assert np.all(p >= 0.0)

def test_irradiance_scaling():
    pv = SolarPVModel()
    _, i_high, p_high = pv.calculate_iv_curve(G=1000.0)
    _, i_low, p_low = pv.calculate_iv_curve(G=400.0)
    
    # Lower irradiance must yield lower peak power
    assert np.max(p_high) > np.max(p_low)
