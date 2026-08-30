import numpy as np

class BoostConverterInterface:
    """
    DC-DC Boost Converter State Interface.
    Translates PV module terminal voltage V_pv into PWM switching duty ratio D
    for a target fixed DC link voltage V_out.
    """
    def __init__(self, v_out=60.0, efficiency=0.95):
        self.v_out = v_out            # Target DC bus link voltage (V)
        self.efficiency = efficiency   # Power conversion efficiency (eta)

    def compute_duty_cycle(self, v_pv):
        """
        Calculates ideal continuous conduction mode duty cycle:
        D = 1 - (V_pv / V_out)
        """
        v_pv_clamped = np.clip(v_pv, 1.0, self.v_out - 1.0)
        duty_cycle = 1.0 - (v_pv_clamped / self.v_out)
        return np.clip(duty_cycle, 0.05, 0.95)

    def compute_output_power(self, p_pv):
        """Calculates net converter power delivered to DC bus."""
        return p_pv * self.efficiency
