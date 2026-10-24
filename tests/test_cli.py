"""Tests for PEP 621 package metadata, `py.typed`, and `pv-mppt` CLI (`src/cli.py`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import src
from src.cli import main


def test_package_version_and_py_typed_marker() -> None:
    assert src.__version__ == "1.0.0"
    py_typed = Path(src.__file__).resolve().parent / "py.typed"
    assert py_typed.is_file()


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "pv-mppt 1.0.0" in out


def test_cli_info_human_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["info"])
    assert rc == 0
    human_out = capsys.readouterr().out
    assert "AU Optronics PM060MBR_300W" in human_out
    assert "300.11" in human_out

    rc_json = main(["info", "--json"])
    assert rc_json == 0
    data = json.loads(capsys.readouterr().out)
    assert data["package"] == "pv-mppt-solver"
    assert data["version"] == "1.0.0"
    assert data["reference_module"]["stc_p_max_w"] == pytest.approx(300.1135, abs=0.05)
    assert data["extracted_single_diode"]["ideality_factor_n"] == pytest.approx(
        1.005823, abs=1e-4
    )


def test_cli_iv_stc_dark_plot_and_domain_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot_file = tmp_path / "iv_test.png"
    rc = main(["iv", "--irradiance", "1000", "--temp", "25", "--plot", str(plot_file), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["p_mpp_w"] == pytest.approx(300.11, abs=0.1)
    assert 70.0 < payload["fill_factor_pct"] < 85.0
    assert plot_file.is_file() and plot_file.stat().st_size > 20_000

    # Zero irradiance dark operating point
    rc_dark = main(["iv", "--irradiance", "0", "--temp", "25", "--json"])
    assert rc_dark == 0
    dark = json.loads(capsys.readouterr().out)
    assert dark["p_mpp_w"] == 0.0
    assert dark["fill_factor_pct"] == 0.0

    # Human table branch
    assert main(["iv", "--irradiance", "800", "--temp", "35"]) == 0
    assert "Single-Diode Operating Point" in capsys.readouterr().out

    # Out-of-envelope temperature returns clean POSIX exit code 2
    rc_err = main(["iv", "--irradiance", "1000", "--temp", "120"])
    assert rc_err == 2
    err_text = capsys.readouterr().err
    assert "pv-mppt error:" in err_text


def test_cli_shading_peaks_gmppt_and_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sh_plot = tmp_path / "shade.png"
    rc = main(
        [
            "shading",
            "--irradiances",
            "1000",
            "700",
            "350",
            "--run-gmppt",
            "--plot",
            str(sh_plot),
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_peaks"] == 3
    assert data["gmpp"]["power_w"] == pytest.approx(148.16, abs=0.5)
    assert data["gmppt_benchmark"]["Hybrid Scan + Adaptive INC"][
        "settled_gmpp_efficiency_pct"
    ] > 98.5
    assert sh_plot.is_file() and sh_plot.stat().st_size > 20_000

    # Human output
    assert main(["shading", "--irradiances", "1000", "700", "350"]) == 0
    assert "Detected Peaks: 3" in capsys.readouterr().out

    # Wrong substring count -> exit code 2
    assert main(["shading", "--irradiances", "1000", "700"]) == 2
    capsys.readouterr()


def test_cli_en50530_json_and_human(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    en_plot = tmp_path / "en50530.png"
    rc = main(["en50530", "--plot", str(en_plot), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    sera = data["controllers"]["Sera dP-P&O (Drift-Free)"]["en50530_efficiency_pct"]
    po = data["controllers"]["Standard P&O"]["en50530_efficiency_pct"]
    assert sera > 99.0
    assert sera > po + 5.0
    assert en_plot.is_file() and en_plot.stat().st_size > 20_000

    assert main(["en50530", "--dt", "0.4"]) == 0
    assert "CENELEC EN 50530 Dynamic Ramp Benchmark" in capsys.readouterr().out


def test_cli_boost_json_stages_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bo_plot = tmp_path / "boost.png"
    rc = main(
        [
            "boost",
            "--v-bus",
            "60",
            "--stages",
            "1000:35",
            "600:35",
            "--plot",
            str(bo_plot),
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert 96.5 < data["euro_efficiency_pct"] < 99.0
    bd = data["stc_loss_breakdown_w"]
    comp_sum = (
        bd["inductor_copper_w"]
        + bd["diode_conduction_w"]
        + bd["mosfet_conduction_w"]
        + bd["mosfet_switching_w"]
        + bd["magnetic_core_w"]
    )
    assert bd["total_loss_w"] == pytest.approx(comp_sum, abs=1e-2)
    assert data["closed_loop_transient"]["rms_servo_tracking_error_mv"] < 150.0
    assert bo_plot.is_file() and bo_plot.stat().st_size > 30_000

    assert main(["boost", "--v-bus", "60", "--stages", "1000:25"]) == 0
    assert "State-Space DC-DC Boost Stage" in capsys.readouterr().out

    # Malformed stage spec triggers argparse exit code 2
    with pytest.raises(SystemExit) as exc_info:
        main(["boost", "--stages", "invalid_stage"])
    assert exc_info.value.code == 2
    capsys.readouterr()

    # Non-positive switching frequency triggers domain error code 2
    assert main(["boost", "--f-sw-khz", "-10"]) == 2
    capsys.readouterr()


def test_cli_benchmark_generates_all_seven_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "figs"
    rc = main(["benchmark", "--output-dir", str(out_dir), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["figures"]) == 7
    for fig_path in data["figures"]:
        p = Path(fig_path)
        assert p.is_file() and p.stat().st_size > 20_000


def test_python_dash_m_src_invocation() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src", "info", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(proc.stdout)
    assert parsed["package"] == "pv-mppt-solver"
    assert parsed["version"] == "1.0.0"
