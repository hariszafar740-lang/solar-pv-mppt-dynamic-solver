# Contributing to Solar PV Dynamic Characterization & MPPT Solver

Thank you for your interest in contributing to this photovoltaic physics and MPPT control simulation suite.

## Engineering & Numerical Verification Standards

1. **Physical & Mathematical Integrity:**
   - Changes to the single-diode equation, De Soto temperature/irradiance scaling, or bypass-diode string solvers must preserve exact STC constraint consistency ($I(0)=I_{sc}$, $I(V_{oc})=0$, $I(V_{mp})=I_{mp}$) and remain cross-validated against `pvlib` (`tests/test_pvlib_crosscheck.py`).
   - Reference module provenance (`AU Optronics PM060MBR_300W`, NREL SAM / CEC database) must remain traceable in `tests/test_reference_cec.py`.
2. **Fail-Closed Input Validation:**
   - Non-finite, complex, negative irradiance, or out-of-envelope temperatures ($T < -40\,^\circ\mathrm{C}$ or $T > +85\,^\circ\mathrm{C}$) must raise `ValueError`.
   - Physically inconsistent datasheet tuples that cannot be satisfied for $0.5 \le n \le 2.5$ must raise `ValueError`.

## Local Development Workflow (Modular Git + Safe Rebase)

~~~bash
git checkout main
git pull --rebase origin main
git checkout -b feat/your-feature-name

python -m pip install -r requirements.txt -r requirements-dev.txt
ruff check .
python -m pytest -v
python main.py
~~~

All pull requests must pass the Python 3.10–3.13 CI matrix before merge.
