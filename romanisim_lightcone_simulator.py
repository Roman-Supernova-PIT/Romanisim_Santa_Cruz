#!/usr/bin/env python3
"""Create Romanisim L1 products from the GalSim lightcone truth simulator."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover - optional dependency
    fits = None

from roman_galsim_simulator import canonical_band_name


ROMANISIM_FILTER_NAMES: Dict[str, str] = {
    "R062": "F062",
    "Z087": "F087",
    "Y106": "F106",
    "J129": "F129",
    "W146": "F146",
    "H158": "F158",
    "F184": "F184",
    "K213": "F213",
}
ROMANISIM_ACTIVE_SHAPE: Tuple[int, int] = (4088, 4088)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ideal GalSim truth images and wrap them into Romanisim L1/L2 products."
    )
    parser.add_argument("catalog", type=Path, help="Input catalog for roman_galsim_simulator.py.")
    parser.add_argument("--catalog-format", default="auto", choices=("auto", "csv", "table", "lightcone"))
    parser.add_argument("--output-dir", type=Path, default=Path("romanisim_output"))
    parser.add_argument("--truth-dir", type=Path, default=Path("romanisim_output/truth"))
    parser.add_argument("--filters", nargs="+", default=["H158"])
    parser.add_argument("--sca", type=int, default=7)
    parser.add_argument("--pointing-ra", type=float, required=True)
    parser.add_argument("--pointing-dec", type=float, required=True)
    parser.add_argument("--position-angle", type=float, default=0.0)
    parser.add_argument("--date", default="2027-07-01T00:00:00")
    parser.add_argument("--edge-padding-arcsec", type=float, default=10.0)
    parser.add_argument("--disk-knot-fraction", type=float, default=0.2)
    parser.add_argument("--disk-knot-count", type=int, default=20)
    parser.add_argument("--disk-knot-radius-scale", type=float, default=0.8)
    parser.add_argument("--level", type=int, default=1, choices=(1, 2), help="Romanisim output level.")
    parser.add_argument("--ma-table-number", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=12345)
    parser.add_argument("--usecrds", action="store_true", help="Pass --usecrds to romanisim-make-image.")
    parser.add_argument(
        "--psftype",
        choices=("epsf", "galsim", "stpsf"),
        default=None,
        help="Optional Romanisim PSF type. Omit by default when using --extra-counts truth images.",
    )
    parser.add_argument(
        "--count-scale",
        type=float,
        default=1.0,
        help="Multiply the ideal FITS image before passing it as Romanisim --extra-counts.",
    )
    parser.add_argument(
        "--extra-counts-shape",
        default="4088,4088",
        help="Shape expected by Romanisim --extra-counts as ny,nx. Use 4096,4096 only if your Romanisim build expects full-frame arrays.",
    )
    parser.add_argument(
        "--integerize-counts",
        choices=("poisson", "round", "none"),
        default="poisson",
        help="Convert extra-counts to integers for Romanisim L1. Use 'poisson' for shot-noise realization or 'round' for deterministic counts.",
    )
    parser.add_argument("--verbose-footprint", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable GalSim truth progress bars.")
    parser.add_argument(
        "--keep-scaled-counts",
        action="store_true",
        help="Keep intermediate count-scaled FITS files when --count-scale is not 1.",
    )
    return parser.parse_args()


def parse_shape(shape_text: str) -> Tuple[int, int]:
    parts = [part.strip() for part in shape_text.split(",")]
    if len(parts) != 2:
        raise ValueError("--extra-counts-shape must be formatted as ny,nx, e.g. 4088,4088")
    return int(parts[0]), int(parts[1])


def romanisim_band_name(band: str) -> str:
    canonical = canonical_band_name(band)
    return ROMANISIM_FILTER_NAMES.get(canonical, canonical)


def truth_fits_path(truth_dir: Path, sca: int, band: str) -> Path:
    canonical = canonical_band_name(band)
    return truth_dir / f"roman_ideal_sca{sca:02d}_{canonical}.fits"


def run_command(command: List[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_console_entry_point(name: str):
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        matches = entry_points.select(group="console_scripts", name=name)
    else:
        matches = [ep for ep in entry_points.get("console_scripts", []) if ep.name == name]
    for entry_point in matches:
        return entry_point
    return None


def run_python_entry_point(name: str, args: List[str]) -> bool:
    entry_point = find_console_entry_point(name)
    if entry_point is None:
        return False

    print(f"{name} {' '.join(args)}", flush=True)
    old_argv = sys.argv[:]
    try:
        sys.argv = [name, *args]
        entry_point.load()()
    finally:
        sys.argv = old_argv
    return True


def make_truth_images(args: argparse.Namespace) -> None:
    script = Path(__file__).with_name("roman_galsim_simulator.py")
    command = [
        sys.executable,
        str(script),
        str(args.catalog),
        "--catalog-format",
        args.catalog_format,
        "--output-dir",
        str(args.truth_dir),
        "--filters",
        *args.filters,
        "--sca",
        str(args.sca),
        "--pointing-ra",
        str(args.pointing_ra),
        "--pointing-dec",
        str(args.pointing_dec),
        "--position-angle",
        str(args.position_angle),
        "--date",
        args.date,
        "--edge-padding-arcsec",
        str(args.edge_padding_arcsec),
        "--disk-knot-fraction",
        str(args.disk_knot_fraction),
        "--disk-knot-count",
        str(args.disk_knot_count),
        "--disk-knot-radius-scale",
        str(args.disk_knot_radius_scale),
    ]
    if args.verbose_footprint:
        command.append("--verbose-footprint")
    if args.no_progress:
        command.append("--no-progress")
    run_command(command)


def crop_center(data, target_shape: Tuple[int, int]):
    target_y, target_x = target_shape
    current_y, current_x = data.shape
    if (current_y, current_x) == target_shape:
        return data
    if target_y > current_y or target_x > current_x:
        raise ValueError(
            f"Cannot crop extra-counts image from {(current_y, current_x)} to larger shape {target_shape}."
        )
    y0 = (current_y - target_y) // 2
    x0 = (current_x - target_x) // 2
    return data[y0 : y0 + target_y, x0 : x0 + target_x]


def integerize_counts(data, method: str, rng_seed: int):
    if method == "none":
        return data
    nonnegative = np.clip(data, 0.0, None)
    if method == "round":
        return np.rint(nonnegative).astype(np.int64)
    rng = np.random.default_rng(rng_seed)
    return rng.poisson(nonnegative).astype(np.int64)


def prepared_counts_file(
    source: Path,
    scale: float,
    output_dir: Path,
    keep: bool,
    target_shape: Tuple[int, int],
    integerize_method: str,
    rng_seed: int,
) -> Path:
    needs_new_file = scale != 1.0 or integerize_method != "none"
    if fits is not None:
        with fits.open(source) as hdul:
            shape = hdul[0].data.shape
        needs_new_file = needs_new_file or shape != target_shape

    if not needs_new_file:
        return source
    if fits is None:
        raise RuntimeError("astropy is required to crop or scale extra-counts FITS files.")

    if keep:
        destination = output_dir / f"{source.stem}_extra_counts_{target_shape[0]}x{target_shape[1]}_scale_{scale:g}.fits"
    else:
        handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        destination = Path(handle.name)
        handle.close()

    with fits.open(source) as hdul:
        prepared_data = crop_center(hdul[0].data, target_shape) * scale
        prepared_data = integerize_counts(prepared_data, integerize_method, rng_seed)
        hdul[0].data = prepared_data
        hdul[0].header["CTSCALE"] = scale
        hdul[0].header["CROPY"] = target_shape[0]
        hdul[0].header["CROPX"] = target_shape[1]
        hdul[0].header["INTMETH"] = integerize_method
        hdul.writeto(destination, overwrite=True)
    return destination


def make_romanisim_product(args: argparse.Namespace, band: str, extra_counts: Path) -> None:
    romanisim_cli = shutil.which("romanisim-make-image")

    canonical = canonical_band_name(band)
    output = args.output_dir / f"romanisim_l{args.level}_sca{args.sca:02d}_{canonical}.asdf"
    romanisim_args = [
        str(output),
        "--level",
        str(args.level),
        "--nobj",
        "0",
        "--extra-counts",
        str(extra_counts),
        "--bandpass",
        romanisim_band_name(canonical),
        "--sca",
        str(args.sca),
        "--radec",
        str(args.pointing_ra),
        str(args.pointing_dec),
        "--boresight",
        "--roll",
        str(args.position_angle),
        "--date",
        args.date,
        "--rng_seed",
        str(args.rng_seed),
    ]
    if args.psftype is not None:
        romanisim_args.extend(["--psftype", args.psftype])
    if args.ma_table_number is not None:
        romanisim_args.extend(["--ma_table_number", str(args.ma_table_number)])
    if args.usecrds:
        romanisim_args.append("--usecrds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if romanisim_cli is not None:
        run_command([romanisim_cli, *romanisim_args])
        return

    if run_python_entry_point("romanisim-make-image", romanisim_args):
        return

    raise RuntimeError(
        "romanisim-make-image was not found on PATH, and no matching Python console entry point "
        "was found. Try reinstalling romanisim in this environment with `python -m pip install --force-reinstall romanisim`."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.truth_dir.mkdir(parents=True, exist_ok=True)

    make_truth_images(args)

    temp_files: List[Path] = []
    target_shape = parse_shape(args.extra_counts_shape)
    try:
        for band in args.filters:
            truth = truth_fits_path(args.truth_dir, args.sca, band)
            counts = prepared_counts_file(
                truth,
                scale=args.count_scale,
                output_dir=args.output_dir,
                keep=args.keep_scaled_counts,
                target_shape=target_shape,
                integerize_method=args.integerize_counts,
                rng_seed=args.rng_seed,
            )
            if counts != truth and not args.keep_scaled_counts:
                temp_files.append(counts)
            make_romanisim_product(args, band, counts)
    finally:
        for path in temp_files:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
