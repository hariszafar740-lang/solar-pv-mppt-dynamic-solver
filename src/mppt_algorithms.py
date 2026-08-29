import numpy as np

class PerturbAndObserve:
    """
    Perturb and Observe (P&O) MPPT Control Engine.
    Iteratively perturbs operating voltage and evaluates power differential ΔP.
    """
    def __init__(self, step_size=0.15, v_init=25.0):
        self.step_size = step_size
        self.v_prev = v_init
        self.p_prev = 0.0

    def step(self, v_curr, p_curr):
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
        return v_next


class IncrementalConductance:
    """
    Incremental Conductance (INC) MPPT Control Engine.
    Uses instantaneous (I/V) and incremental (dI/dV) conductance derivatives 
    to eliminate steady-state oscillations around the Maximum Power Point.
    """
    def __init__(self, step_size=0.15, v_init=25.0):
        self.step_size = step_size
        self.v_prev = v_init
        self.i_prev = 0.0

    def step(self, v_curr, i_curr):
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
            
            if np.isclose(inc_cond, inst_cond, atol=1e-2):
                v_next = v_curr
            elif inc_cond > inst_cond:
                v_next = v_curr + self.step_size
            else:
                v_next = v_curr - self.step_size
                
        self.v_prev = v_curr
        self.i_prev = i_curr
        return v_next
