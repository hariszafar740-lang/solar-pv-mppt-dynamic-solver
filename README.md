# Solar PV Dynamic Characterization & MPPT Solver

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Build Status](https://img.shields.io/badge/pytest-passing-brightgreen)
![Domain](https://img.shields.io/badge/Focus-Renewable%20Energy%20Integration-orange)

Physics-based Python implementation of the **Single-Diode Solar Photovoltaic (PV) Module Model**, dynamic **MPPT Control Engines** (P&O and Incremental Conductance), and a **DC-DC Boost Converter Duty Cycle Translator** for grid-tied inverter links.

## Mathematical System Formulation

### 1. Single-Diode Cell Equation
$$I = I_{ph} - I_0 \left[ \exp\left( \frac{V + I R_s}{n V_t} \right) - 1 \right] - \frac{V + I R_s}{R_{sh}}$$

### 2. Boost Converter Duty Cycle Translation
To match the dynamic PV array MPPT voltage $V_{mpp}$ to a fixed DC link voltage $V_{out}$:

$$D = 1 - \frac{V_{mpp}}{V_{out}}$$

## Simulation Artifacts & Dynamic Results

### STC PV Characteristic Curves
![PV Characteristics](results/pv_characteristics.png)

### Dynamic MPPT Step Response
![MPPT Tracking](results/mppt_tracking.png)

### Thermal Power Derating
![Thermal Derating](results/thermal_derating.png)

## How to Run
```bash
source ~/ee_portfolio/ee_portfolio_env/bin/activate
PYTHONPATH=. pytest tests/
python3 main.py
