import pytest
import numpy as np
from src.pv_model import SolarPVModel
from src.mppt_algorithms import PerturbAndObserve, IncrementalConductance

def test_po_mppt_efficiency():
    pv = SolarPVModel()
    po = PerturbAndObserve(step_size=0.15, v_init=20.0)
    v_arr, _, p_arr = pv.calculate_iv_curve(G=1000.0)
    p_true_mpp = np.max(p_arr)
    
    v_track = 20.0
    for _ in range(150):
        p_track = np.interp(v_track, v_arr, p_arr)
        v_track = po.step(v_track, p_track)
        
    efficiency = (p_track / p_true_mpp) * 100.0
    assert efficiency >= 98.0

def test_inc_mppt_efficiency():
    pv = SolarPVModel()
    inc = IncrementalConductance(step_size=0.15, v_init=20.0)
    v_arr, i_arr, p_arr = pv.calculate_iv_curve(G=1000.0)
    p_true_mpp = np.max(p_arr)
    
    v_track = 20.0
    for _ in range(150):
        p_track = np.interp(v_track, v_arr, p_arr)
        i_track = np.interp(v_track, v_arr, i_arr)
        v_track = inc.step(v_track, i_track)
        
    efficiency = (p_track / p_true_mpp) * 100.0
    assert efficiency >= 98.0
