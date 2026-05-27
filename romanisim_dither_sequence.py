#!/usr/bin/env python3
"""Run a sequence of Romanisim lightcone simulations with random dithers and rolls."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


FILTER_ALIASES = {
    "F062": "R062",
    "R062": "R062",
    "F087": "Z087",
    "Z087": "Z087",
    "F106": "Y106",
    "Y106": "Y106",
    "F129": "J129",
    "J129": "J129",
    "F146": "W146",
    "W146": "W146",
    "W149": "W146",
    "F158": "H158",
    "H158": "H158",
    "F184": "F184",
    "F213": "K213",
    "K213": "K213",
}


def canonical_band_name(band: str) -> str:
    normalized = band.strip().upper()
    return FILTER_ALIASES.get(normalized, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random dither/roll pointings and run romanisim_lightcone_simulator.py."
    )
    parser.add_argument("catalog", type=Path, nargs="+", help="Input lightcone catalog(s).")
    parser.add_argument("--catalog-format", default="lightcone", choices=("auto", "csv", "table", "lightcone"))
    parser.add_argument("--output-dir", type=Path, default=Path("romanisim_dither_output"))
    parser.add_argument("--filters", nargs="+", default=["H158"])
    parser.add_argument("--sca", type=int, default=7)
    parser.add_argument("--base-ra", type=float, required=True, help="Base boresight RA in degrees.")
    parser.add_argument("--base-dec", type=float, required=True, help="Base boresight Dec in degrees.")
    parser.add_argument("--base-roll", type=float, default=0.0, help="Base position angle in degrees.")
    parser.add_argument("--date", default="2027-07-01T00:00:00")
    parser.add_argument("--n-exposures", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dither-radius-arcsec", type=float, default=60.0)
    parser.add_argument("--roll-range-deg", type=float, default=1.0)
    parser.add_argument("--level", type=int, default=2, choices=(1, 2))
    parser.add_argument("--count-scale", type=float, default=140.0)
    parser.add_argument("--edge-padding-arcsec", type=float, default=10.0)
    parser.add_argument("--disk-knot-fraction", type=float, default=0.2)
    parser.add_argument("--disk-knot-count", type=int, default=20)
    parser.add_argument("--disk-knot-radius-scale", type=float, default=0.8)
    parser.add_argument("--render-mode", choices=("achromatic", "chromatic"), default="achromatic")
    parser.add_argument("--psf-mode", choices=("achromatic", "chromatic", "none"), default="achromatic")
    parser.add_argument("--max-draw-objects", type=int, default=None)
    parser.add_argument("--max-mag", type=float, default=None)
    parser.add_argument("--integerize-counts", choices=("poisson", "round", "none"), default="poisson")
    parser.add_argument("--extra-counts-shape", default="4088,4088")
    parser.add_argument("--ma-table-number", type=int, default=None)
    parser.add_argument("--usecrds", action="store_true")
    parser.add_argument("--psftype", choices=("epsf", "galsim", "stpsf"), default=None)
    parser.add_argument("--verbose-footprint", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--convert-fits",
        action="store_true",
        default=True,
        help="Convert each Romanisim ASDF product to FITS after simulation.",
    )
    parser.add_argument("--no-convert-fits", action="store_false", dest="convert_fits")
    parser.add_argument(
        "--fits-all-slices",
        action="store_true",
        help="For L1 ASDF resultants, write every plane as a FITS extension.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the dither table and print commands without running them.")
    parser.add_argument("--write-sbatch", action="store_true", help="Write one Slurm sbatch script per exposure instead of running locally.")
    parser.add_argument("--submit", action="store_true", help="Submit generated sbatch scripts with sbatch.")
    parser.add_argument("--sbatch-dir", type=Path, default=None, help="Directory for generated sbatch scripts.")
    parser.add_argument("--job-name", default="catalog")
    parser.add_argument("--partition", default="shared,kill-shared")
    parser.add_argument("--time", default="0-24:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--conda-env", default="stenv")
    parser.add_argument("--git-pull", action="store_true", default=True)
    parser.add_argument("--no-git-pull", action="store_false", dest="git_pull")
    return parser.parse_args()


def random_dither(rng: np.random.Generator, radius_arcsec: float) -> Tuple[float, float]:
    radius = radius_arcsec * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return radius * math.cos(theta), radius * math.sin(theta)


def offset_radec(ra_deg: float, dec_deg: float, dra_arcsec: float, ddec_arcsec: float) -> Tuple[float, float]:
    dec_rad = math.radians(dec_deg)
    cos_dec = max(math.cos(dec_rad), 1.0e-6)
    return ra_deg + dra_arcsec / 3600.0 / cos_dec, dec_deg + ddec_arcsec / 3600.0


def write_dither_table(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_command(
    args: argparse.Namespace,
    row: dict,
    exposure_dir: Path,
    truth_dir: Path,
    python_executable: str = sys.executable,
) -> List[str]:
    script = Path(__file__).with_name("romanisim_lightcone_simulator.py")
    command = [
        python_executable,
        str(script),
        *[str(path) for path in args.catalog],
        "--catalog-format",
        args.catalog_format,
        "--output-dir",
        str(exposure_dir),
        "--truth-dir",
        str(truth_dir),
        "--filters",
        *args.filters,
        "--sca",
        str(args.sca),
        "--pointing-ra",
        f"{row['ra_deg']:.10f}",
        "--pointing-dec",
        f"{row['dec_deg']:.10f}",
        "--position-angle",
        f"{row['roll_deg']:.10f}",
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
        "--render-mode",
        args.render_mode,
        "--psf-mode",
        args.psf_mode,
        "--level",
        str(args.level),
        "--count-scale",
        str(args.count_scale),
        "--integerize-counts",
        args.integerize_counts,
        "--extra-counts-shape",
        args.extra_counts_shape,
        "--rng-seed",
        str(row["rng_seed"]),
    ]
    if args.max_draw_objects is not None:
        command.extend(["--max-draw-objects", str(args.max_draw_objects)])
    if args.max_mag is not None:
        command.extend(["--max-mag", str(args.max_mag)])
    if args.ma_table_number is not None:
        command.extend(["--ma-table-number", str(args.ma_table_number)])
    if args.usecrds:
        command.append("--usecrds")
    if args.psftype is not None:
        command.extend(["--psftype", args.psftype])
    if args.verbose_footprint:
        command.append("--verbose-footprint")
    if args.no_progress:
        command.append("--no-progress")
    return command


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_fits_conversion_commands(
    args: argparse.Namespace,
    exposure_dir: Path,
    python_executable: str = sys.executable,
) -> List[List[str]]:
    if not args.convert_fits:
        return []

    script = Path(__file__).with_name("roman_asdf_to_fits.py")
    commands: List[List[str]] = []
    for band in args.filters:
        canonical = canonical_band_name(band)
        asdf_path = exposure_dir / f"romanisim_l{args.level}_sca{args.sca:02d}_{canonical}.asdf"
        fits_path = exposure_dir / f"romanisim_l{args.level}_sca{args.sca:02d}_{canonical}.fits"
        command = [python_executable, str(script), str(asdf_path), str(fits_path)]
        if args.fits_all_slices:
            command.append("--all-slices")
        commands.append(command)
    return commands


def write_sbatch_script(args: argparse.Namespace, row: dict, commands: List[List[str]]) -> Path:
    sbatch_dir = args.sbatch_dir or (args.output_dir / "sbatch")
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    script_path = sbatch_dir / f"exp{row['exposure']:04d}_sca{args.sca:02d}.sbatch"
    log_prefix = args.output_dir / f"exp{row['exposure']:04d}" / "slurm"
    script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}_e{row['exposure']:04d}_s{args.sca:02d}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --error={log_prefix}-%A.err",
        f"#SBATCH --output={log_prefix}-%A.out",
        "",
        "set -euo pipefail",
        "",
        f"source activate {shlex.quote(args.conda_env)}",
    ]
    if args.git_pull:
        script.append("git pull")
    script.extend(
        [
            "",
            *[shell_join(command) for command in commands],
            "",
        ]
    )
    script_path.write_text("\n".join(script))
    script_path.chmod(0o755)
    return script_path


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: List[dict] = []

    for exposure in range(1, args.n_exposures + 1):
        dra, ddec = random_dither(rng, args.dither_radius_arcsec)
        ra, dec = offset_radec(args.base_ra, args.base_dec, dra, ddec)
        roll = args.base_roll + rng.uniform(-args.roll_range_deg, args.roll_range_deg)
        rows.append(
            {
                "exposure": exposure,
                "ra_deg": ra,
                "dec_deg": dec,
                "roll_deg": roll,
                "dra_arcsec": dra,
                "ddec_arcsec": ddec,
                "rng_seed": int(rng.integers(1, 2**31 - 1)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "dither_sequence.csv"
    write_dither_table(table_path, rows)
    print(f"Wrote {table_path}")

    for row in rows:
        exposure_dir = args.output_dir / f"exp{row['exposure']:04d}"
        truth_dir = exposure_dir / "truth"
        python_executable = "python" if (args.write_sbatch or args.submit) else sys.executable
        command = build_command(args, row, exposure_dir, truth_dir, python_executable=python_executable)
        commands = [command, *build_fits_conversion_commands(args, exposure_dir, python_executable=python_executable)]
        for planned_command in commands:
            print(" ".join(planned_command), flush=True)
        if args.write_sbatch or args.submit:
            script_path = write_sbatch_script(args, row, commands)
            print(f"Wrote {script_path}", flush=True)
            if args.submit and not args.dry_run:
                subprocess.run(["sbatch", str(script_path)], check=True)
            continue
        if not args.dry_run:
            for planned_command in commands:
                subprocess.run(planned_command, check=True)


if __name__ == "__main__":
    main()
