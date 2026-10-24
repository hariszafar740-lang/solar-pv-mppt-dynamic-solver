# Solar PV Dynamic Characterization & MPPT Solver

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![CI](https://github.com/hariszafar740-lang/solar-pv-mppt-dynamic-solver/actions/workflows/ci.yml/badge.svg)
![Domain](https://img.shields.io/badge/Focus-Renewable%20Energy%20Integration-orange)

Physics-based Python implementation of the **Single-Diode Solar Photovoltaic (PV) Module Model**, dynamic **MPPT Control Engines** (P&O and Incremental Conductance), and a **DC-DC Boost Converter Duty Cycle Translator** for grid-tied inverter links.

## Mathematical System Formulation

### 1. Single-Diode Cell Equation

$$I = I_{ph} - I_0 \left[ \exp\left( \frac{V + I R_s}{n N_s k T / q} \right) - 1 \right] - \frac{V + I R_s}{R_{sh}(G)}$$

solved for $I$ at every grid voltage with a vectorized, monotone-convergent Newton iteration (bisection fallback).

### 2. Self-Consistent Reference-Parameter Extraction

The diode parameters $(I_{ph,ref}, I_{0,ref}, n)$ are **not** hand-picked. At construction time the solver exactly satisfies the three STC datasheet constraints

$$I(V{=}0) = I_{sc}, \qquad I(V{=}V_{oc}) = 0, \qquad I(V{=}V_{mp}) = I_{mp}$$

For fixed $n$, the first two constraints are linear in $(I_{ph,ref}, I_{0,ref})$; the third is a scalar root-find in $n$ (bracketed bisection). If the datasheet values are inconsistent with the single-diode model for any physical ideality factor ($0.5 \le n \le 2.5$), the constructor raises `ValueError` rather than silently under-predicting STC power.

### 3. Temperature / Irradiance Scaling (De Soto / CEC convention)

$$I_{ph}(G,T) = \left(I_{ph,ref} + K_i (T - T_{ref})\right)\frac{G}{G_{ref}}, \qquad
I_0(T) = I_{0,ref}\left(\frac{T}{T_{ref}}\right)^3 \exp\!\left(\frac{E_{g,ref}}{k_{eV} T_{ref}} - \frac{E_g(T)}{k_{eV} T}\right)$$

$$E_g(T) = 1.121\,\mathrm{eV}\cdot\left(1 - 2.677\times10^{-4}\,(T - T_{ref})\right), \qquad
R_{sh}(G) = R_{sh,ref}\,\frac{G_{ref}}{G}$$

### 4. Boost Converter Duty Cycle Translation

To match the dynamic PV array MPPT voltage $V_{mpp}$ to a fixed DC link voltage $V_{out}$:

$$D = 1 - \frac{V_{mpp}}{V_{out}}$$

### 5. Series-String Bypass-Diode Partial Shading

For a series string of $M = N_{\text{mod}} \times N_{\text{bp}}$ cell substrings under non-uniform irradiance $G_m$ and cell temperature $T_m$, Kirchhoff's Voltage Law gives $V_{\text{string}}(I) = \sum_{m=1}^{M} V_m(I)$, where each substring is protected by an anti-parallel Schottky bypass diode conducting in reverse bias ($V_m < 0$):

$$I_{\text{sub},m}(V_m) = I_{\text{cell},m}(V_m) + \begin{cases} I_{0,\text{bp}}\left[\exp(-V_m / V_{t,\text{bp}}) - 1\right], & V_m < 0 \\ 0, & V_m \ge 0 \end{cases}$$

### 6. Global MPPT Under Multi-Peak Partial Shading

Conventional hill-climbing trackers (`PerturbAndObserve`, `IncrementalConductance`) follow local $dP/dV$ sign and become trapped in high-voltage local maxima (LMPPs). `src/global_mppt.py` provides two Global MPPT (GMPPT) engines with zero steady-state oscillation lock and bypassed-plateau cloud-clearance watchdog probing:

* **`ParticleSwarmMPPT`**: Stratified $N_p$-particle voltage swarm with annealed inertia $w(k): w_{\max} \to w_{\min}$, velocity clamping, convergence lock, and event-triggered re-dispersion.
* **`HybridGlobalScanINC`**: Fast coarse global voltage sweep across $[V_{\min}, V_{\max}]$ followed by Variable-Step Adaptive Incremental Conductance ($\Delta V_k = \operatorname{clip}(\alpha |dP/dV|, \Delta V_{\min}, \Delta V_{\max})$).

## Reference Module & Data Provenance

Default constructor parameters are the STC datasheet values of the
**AU Optronics PM060MBR_300W** monocrystalline 60-cell module, taken verbatim from the
**NREL SAM / CEC module database**, record version **"SAM 2018.11.11 r2"** (dated 2019-01-03), as bundled with **pvlib 0.16.1** (`sam-library-cec-modules-2019-03-05.csv`, SHA-256 `a7c3b1ad3dabb5425368615c16322f2e35185fc416380b471c4e48dd545b1920`).

| Quantity | Value | Source |
|---|---|---|
| `I_sc_ref` | 9.91 A | record datasheet value (model input) |
| `V_oc_ref` | 40.45 V | record datasheet value (model input) |
| `I_mp_ref` | 9.24 A | record datasheet value (model input) |
| `V_mp_ref` | 32.48 V | record datasheet value (model input) |
| `R_s` | 0.358977 Ω | record CEC fit (model input) |
| `R_sh_ref` | 171.705124 Ω | record CEC fit (model input) |
| `alpha_sc` | 0.004162 A/K | record (model input, `K_i`) |
| `STC` | **300.1152 W** | record rating — *verification target only* |
| `a_ref` | 1.550532 V | record CEC fit — *verification target only* |
| `I_L_ref` / `I_o_ref` | 9.930718 A / 4.536912e-11 A | record CEC fit — *verification targets only* |

Independent verification (see `tests/test_reference_cec.py` and `tests/test_pvlib_crosscheck.py`):

* Extraction from the datasheet values alone reproduces the record's own CEC-fitted diode parameters: `a = 1.55053183 V` vs `a_ref = 1.550532`, `I_ph_ref = 9.93071844 A` vs `I_L_ref = 9.930718`, `I_o_ref = 4.5369e-11 A` vs `4.536912e-11`.
* Model STC maximum power = **300.1152 W**, matching the record's `STC` field and pvlib's own `singlediode` result (300.1151 W) to solver precision.
* Full I-V curves agree with pvlib's Lambert-W evaluation (`i_from_v`) to < 1e-8 A across 9 (G, T) conditions from −10 °C to 75 °C and 50–1100 W/m².
* Implied temperature coefficients are physical and close to the record's: β_voc ≈ −0.125 V/K (record −0.114069), γ_p ≈ −0.391 %/K (record −0.386).

> **Provenance disclaimer.** The identity of the module that the *original* hard-coded defaults
> (I_sc=9.35, V_oc=38.2, I_mp=8.80, V_mp=34.1, R_s=0.221, R_sh=415.6, n=1.3) were meant to
> represent **could not be established**: no CEC/SAM record matches them, and they are mutually
> inconsistent with the single-diode model — reproducing their nameplate 300.08 W would require a
> non-physical ideality factor (n ≈ 0.47). The old implementation silently ignored I_mp/V_mp and
> produced 266.898 W at STC. The constructor now rejects such inconsistent sets with a
> `ValueError` (regression-tested). The AU Optronics record above is a **pinned, traceable
> replacement reference**, not a claim that it was the original panel.

## Validity Envelope

* Irradiance: finite, real, `G >= 0` W/m² (`G < 0` raises `ValueError`; `G = 0` returns zero generation on a well-formed grid).
* Cell temperature: finite, real, **−40 °C ≤ T ≤ +85 °C** (standard c-Si module operating envelope).
* All outputs are real, finite `float64`; current is reported in the generating quadrant (clamped at 0 beyond V_oc).
* Known simplifications vs. SAM/pvlib: CEC `Adjust` factor taken as 0; `R_s` temperature independent; ideality factor constant with temperature. The model is an engineering-grade single-diode solver, not a manufacturer-level performance prediction tool.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: independent pvlib cross-validation + lint tooling (Python >= 3.11 for pvlib)
pip install -r requirements-dev.txt -r requirements-validation.txt

python -m pytest                     # run the test suite
python main.py                       # regenerate benchmarks and plots in results/
```

## Simulation Artifacts & Dynamic Results

### STC PV Characteristic Curves
![PV Characteristics](results/pv_characteristics.png)

### Dynamic MPPT Step Response
![MPPT Tracking](results/mppt_tracking.png)

### Thermal Power Derating
![Thermal Derating](results/thermal_derating.png)

### Bypass-Diode Partial-Shading & Multi-Peak P-V Curves
![Partial Shading Curves](results/partial_shading_curves.png)

### Global MPPT (PSO & Hybrid Scan + Adaptive INC) vs. Local-Trap Conventional MPPT
![Global MPPT Comparison](results/global_mppt_comparison.png)

## Testing & CI

* `pytest` suite (178 tests) covers the STC reference point, the historical 300.08 W / 266.898 W defect, invalid-input rejection (negative/NaN/complex irradiance, out-of-envelope temperature), zero & low irradiance, thermal derating, MPPT convergence/steady-state stability/dynamic step response/voltage bounds, boost-converter math & clamping, and the pinned CEC reference data.
* pvlib cross-checks (`tests/test_pvlib_crosscheck.py`) run automatically when pvlib is installed and skip otherwise — the core suite is hermetic and needs no network access.
* GitHub Actions (`.github/workflows/ci.yml`) runs ruff lint + pytest on Python 3.10/3.11/3.12 (pvlib cross-checks on 3.11+).
