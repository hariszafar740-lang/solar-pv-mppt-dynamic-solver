import numpy as np

class SolarPVModel:
    """
    Physics-based Single-Diode Solar Photovoltaic Module Solver.
    Default parameters based on standard 300W PV module specifications.
    """
    def __init__(self, I_sc=9.35, V_oc=38.2, I_mp=8.80, V_mp=34.1, 
                 N_s=60, R_s=0.221, R_sh=415.6, n=1.3, K_i=0.005):
        self.I_sc = I_sc    # Short-circuit current (A)
        self.V_oc = V_oc    # Open-circuit voltage (V)
        self.I_mp = I_mp    # Maximum power point current (A)
        self.V_mp = V_mp    # Maximum power point voltage (V)
        self.N_s = N_s      # Number of series connected cells
        self.R_s = R_s      # Series resistance (Ohms)
        self.R_sh = R_sh    # Shunt resistance (Ohms)
        self.n = n          # Diode ideality factor
        self.K_i = K_i      # Temperature coefficient of I_sc (A/K)
        
        # Physical Constants
        self.q = 1.602176634e-19   # Electron charge (C)
        self.k = 1.380649e-23      # Boltzmann constant (J/K)
        self.T_ref = 298.15        # STC Temperature (25 deg C in Kelvin)
        self.G_ref = 1000.0        # STC Irradiance (W/m^2)

    def calculate_iv_curve(self, G=1000.0, T_c_celsius=25.0, num_points=200):
        """
        Solves I-V and P-V characteristic arrays for given Irradiance G (W/m^2) 
        and Cell Temperature T_c (deg C) using dynamic current iteration.
        """
        T_k = T_c_celsius + 273.15
        V_t = (self.N_s * self.n * self.k * T_k) / self.q  # Thermal voltage
        
        # Photocurrent (I_ph) & Reverse Saturation Current (I_0)
        I_ph = (self.I_sc + self.K_i * (T_k - self.T_ref)) * (G / self.G_ref)
        I_0 = (self.I_sc / (np.exp(self.V_oc / V_t) - 1)) * ((T_k / self.T_ref) ** 3)
        
        v_array = np.linspace(0, self.V_oc * (G / self.G_ref)**0.05, num_points)
        i_array = np.zeros(num_points)
        
        for idx, V in enumerate(v_array):
            # Newton-Raphson iteration for implicit Current I(V)
            I_k = I_ph
            for _ in range(20):
                arg = (V + I_k * self.R_s) / V_t
                arg = np.clip(arg, -50, 50)  # Prevent exponential overflow
                f_val = I_ph - I_0 * (np.exp(arg) - 1.0) - (V + I_k * self.R_s) / self.R_sh - I_k
                f_prime = -I_0 * (self.R_s / V_t) * np.exp(arg) - (self.R_s / self.R_sh) - 1.0
                I_next = I_k - (f_val / f_prime)
                if abs(I_next - I_k) < 1e-6:
                    I_k = I_next
                    break
                I_k = I_next
            i_array[idx] = max(0.0, I_k)
            
        p_array = v_array * i_array
        return v_array, i_array, p_array
