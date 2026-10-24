"""Tests for static interactive GitHub Pages explorer (`src/web_dashboard.py`)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.cli import main
from src.web_dashboard import (
    FIGURE_FILENAMES,
    build_static_site,
    export_benchmark_payload,
)


def test_export_benchmark_payload_matches_verified_physics() -> None:
    payload = export_benchmark_payload()
    assert payload["package"] == "pv-mppt-solver"
    assert payload["version"] == "1.0.0"

    mod = payload["module_params"]
    assert mod["name"] == "AU Optronics PM060MBR_300W"
    assert mod["stc_p_max_w"] == pytest.approx(300.1135, abs=0.05)
    assert mod["n_cells"] == 60
    assert mod["n_substrings"] == 3

    gmppt = payload["gmppt"]
    assert gmppt["controllers"]["Hybrid Scan + Adaptive INC"]["dynamic_eff_pct"] > 90.0
    assert gmppt["controllers"]["PSO Global MPPT"]["settled_eff_pct"] > 99.0

    en = payload["en50530"]
    assert en["controllers"]["Sera dP-P&O (Drift-Free)"]["en50530_eff_pct"] > 99.5
    assert en["controllers"]["Standard P&O"]["en50530_eff_pct"] < 90.0

    boost = payload["boost"]
    assert 97.0 < boost["bus_60v"]["euro_efficiency_pct"] < 98.5
    assert boost["bus_60v"]["euro_efficiency_pct"] > boost["bus_120v"]["euro_efficiency_pct"]
    assert boost["transient"]["rms_servo_mv"] < 120.0


def test_build_static_site_bundle_and_asset_links(tmp_path: Path) -> None:
    out_dir = tmp_path / "site"
    manifest = build_static_site(out_dir, results_dir="results")

    index_path = Path(manifest["index_html"])
    json_path = Path(manifest["benchmark_json"])
    assert index_path.is_file() and index_path.stat().st_size > 15_000
    assert json_path.is_file() and json_path.stat().st_size > 10_000
    assert (out_dir / ".nojekyll").is_file()
    assert len(manifest["assets"]) == len(FIGURE_FILENAMES)

    html = index_path.read_text(encoding="utf-8")
    assert "AU Optronics PM060MBR_300W" in html
    assert "CENELEC EN 50530" in html

    # Every <img src="assets/..."> in index.html must resolve to a real non-empty file
    img_srcs = re.findall(r'<img\s+src="(assets/[^"]+)"', html)
    assert len(img_srcs) == 7
    for rel_src in img_srcs:
        asset_file = out_dir / rel_src
        assert asset_file.is_file() and asset_file.stat().st_size > 20_000


def test_cli_web_subcommand_human_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site_dir = tmp_path / "cli_site"
    rc = main(["web", "--output-dir", str(site_dir), "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert Path(parsed["index_html"]).is_file()
    assert len(parsed["assets"]) == 7

    site_dir2 = tmp_path / "cli_site_human"
    assert main(["web", "--output-dir", str(site_dir2)]) == 0
    assert "Static Interactive Explorer Built" in capsys.readouterr().out
