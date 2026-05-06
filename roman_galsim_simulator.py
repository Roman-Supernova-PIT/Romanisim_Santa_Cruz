#!/usr/bin/env python3
"""Render ideal Roman WFI images from a source catalog using GalSim."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import galsim
from galsim import roman
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from astropy.io import fits
    from astropy.table import Table
except ImportError:  # pragma: no cover - optional dependency
    fits = None
    Table = None

try:
    from astropy.cosmology import Planck18
except ImportError:  # pragma: no cover - optional dependency
    Planck18 = None


ROMAN_NATIVE_NPIX = 4096
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
ROMAN_LIGHTCONE_MAG_COLUMNS = {
    "R062": "Roman_F062",
    "Z087": "Roman_F087",
    "Y106": "Roman_F106",
    "J129": "Roman_F129",
    "W146": "Roman_F146",
    "H158": "Roman_F158",
    "F184": "Roman_F184",
    "K213": "Roman_F213",
}
CANONICAL_TO_ALIASES: Dict[str, Tuple[str, ...]] = {}
for canonical in sorted(set(FILTER_ALIASES.values())):
    CANONICAL_TO_ALIASES[canonical] = tuple(
        alias for alias, resolved in FILTER_ALIASES.items() if resolved == canonical
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ideal Roman WFI detector images from a morphology catalog."
    )
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="+",
        help="Input catalog(s): CSV, ECSV, FITS, ASCII, or one or more lightcone.dat files.",
    )
    parser.add_argument(
        "--catalog-format",
        choices=("auto", "csv", "table", "lightcone"),
        default="auto",
        help="Input format. 'auto' detects CSV/table vs. the semi-analytic lightcone .dat layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the simulated FITS images.",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["H158"],
        help="Roman filters to render. Example: --filters Y106 J129 H158",
    )
    parser.add_argument("--sca", type=int, default=7, help="Roman SCA number.")
    parser.add_argument(
        "--pointing-ra",
        type=float,
        required=True,
        help="Telescope boresight RA in degrees.",
    )
    parser.add_argument(
        "--pointing-dec",
        type=float,
        required=True,
        help="Telescope boresight Dec in degrees.",
    )
    parser.add_argument(
        "--position-angle",
        type=float,
        default=0.0,
        help="Roman observatory position angle in degrees.",
    )
    parser.add_argument(
        "--date",
        default="2027-07-01T00:00:00",
        help="Observation datetime in ISO format, e.g. 2027-07-01T00:00:00",
    )
    parser.add_argument(
        "--include-psf",
        action="store_true",
        default=True,
        help="Convolve sources with the Roman PSF for each filter.",
    )
    parser.add_argument(
        "--no-psf",
        action="store_false",
        dest="include_psf",
        help="Skip PSF convolution and render intrinsic morphology only.",
    )
    parser.add_argument(
        "--include-pixel",
        action="store_true",
        default=True,
        help="Include ideal detector pixel integration in the rendered image.",
    )
    parser.add_argument(
        "--no-pixel",
        action="store_false",
        dest="include_pixel",
        help="Skip detector pixel convolution and sample the model directly.",
    )
    parser.add_argument(
        "--mag-prefix",
        default="mag_",
        help="Magnitude column prefix for generic catalogs. Lightcone .dat files ignore this and use Roman_F### columns.",
    )
    parser.add_argument(
        "--verbose-footprint",
        action="store_true",
        help="Print source placement diagnostics for the chosen SCA.",
    )
    parser.add_argument(
        "--edge-padding-arcsec",
        type=float,
        default=10.0,
        help="Keep sources whose centers are this far outside the SCA footprint, so extended galaxies can overlap the image.",
    )
    parser.add_argument(
        "--disk-knot-fraction",
        type=float,
        default=0.2,
        help="Fraction of lightcone disk flux to put into GalSim RandomKnots.",
    )
    parser.add_argument(
        "--disk-knot-count",
        type=int,
        default=20,
        help="Number of RandomKnots points used for clumpy lightcone disk components.",
    )
    parser.add_argument(
        "--disk-knot-radius-scale",
        type=float,
        default=0.8,
        help="RandomKnots half-light radius as a fraction of the disk half-light radius.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress while reading and rendering.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable progress output.",
    )
    return parser.parse_args()


def progress_iter(iterable, *, total: Optional[int] = None, desc: str = "", enabled: bool = True):
    if not enabled:
        return iterable
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="row")
    return iterable


def progress_update(index: int, total: Optional[int], desc: str, enabled: bool, interval: int = 1000) -> None:
    if not enabled or tqdm is not None:
        return
    if index == 1 or index % interval == 0 or (total is not None and index == total):
        if total is None:
            print(f"{desc}: {index} rows", flush=True)
        else:
            print(f"{desc}: {index}/{total} rows", flush=True)


def detect_catalog_format(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "csv"
    try:
        with path.open("r") as handle:
            saw_indexed_header = False
            for line in handle:
                if line.startswith("#"):
                    stripped = line[1:].strip()
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        saw_indexed_header = True
                    if "Roman_F158" in line or "Roman_F184" in line:
                        return "lightcone"
                    continue
                if saw_indexed_header and path.suffix.lower() == ".dat":
                    return "lightcone"
                break
    except OSError:
        pass
    if path.suffix.lower() in {".fits", ".fit", ".ecsv", ".txt", ".dat"}:
        return "table"
    return "table"


def parse_lightcone_header(path: Path) -> List[str]:
    columns: List[str] = []
    with path.open("r") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            stripped = line[1:].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            index = int(parts[0])
            name = parts[1]
            while len(columns) <= index:
                columns.append(f"col{len(columns)}")
            columns[index] = name
    return columns


def count_data_rows(path: Path) -> int:
    n_rows = 0
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                n_rows += 1
    return n_rows


def read_lightcone_catalog(path: Path, show_progress: bool) -> List[MutableMapping[str, object]]:
    columns = parse_lightcone_header(path)
    if show_progress:
        print(f"Counting rows in {path}...", flush=True)
    total_rows = count_data_rows(path) if show_progress else None
    rows: List[MutableMapping[str, object]] = []
    with path.open("r") as handle:
        data_lines = (line for line in handle if line.strip() and not line.startswith("#"))
        for line in progress_iter(data_lines, total=total_rows, desc="Reading lightcone", enabled=show_progress):
            stripped = line.strip()
            values = stripped.split()
            row = {
                columns[idx] if idx < len(columns) else f"col{idx}": value
                for idx, value in enumerate(values)
            }
            rows.append(row)
    return rows


def read_one_catalog(path: Path, catalog_format: str, show_progress: bool) -> List[MutableMapping[str, object]]:
    if catalog_format == "auto":
        catalog_format = detect_catalog_format(path)
    if catalog_format == "csv":
        with path.open("r", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if catalog_format == "lightcone":
        return read_lightcone_catalog(path, show_progress=show_progress)
    if Table is not None:
        table = Table.read(path)
        return [{name: row[name] for name in table.colnames} for row in table]
    raise RuntimeError(
        "Only CSV and lightcone .dat input are supported without astropy. Install astropy for FITS/ECSV catalogs."
    )


def read_catalogs(paths: Sequence[Path], catalog_format: str, show_progress: bool) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    for path in paths:
        print(f"Reading catalog {path}...", flush=True)
        rows.extend(read_one_catalog(path, catalog_format, show_progress=show_progress))
    print(f"Read {len(rows)} total catalog rows from {len(paths)} file(s).", flush=True)
    return rows


def normalize_catalog_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {str(key).strip(): value for key, value in row.items()}


def maybe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
        value = float(value)
        return default if math.isnan(value) else value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return float(text)


def maybe_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def lookup_value(row: Mapping[str, object], *names: str) -> object:
    lower_map = {key.lower(): value for key, value in row.items()}
    for name in names:
        if name in row:
            return row[name]
        lowered = name.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    raise KeyError(names[0])


def optional_value(row: Mapping[str, object], *names: str, default: object = None) -> object:
    try:
        return lookup_value(row, *names)
    except KeyError:
        return default


def canonical_band_name(band: str) -> str:
    normalized = band.strip().upper()
    return FILTER_ALIASES.get(normalized, normalized)


def band_aliases(band: str) -> Tuple[str, ...]:
    canonical = canonical_band_name(band)
    aliases = CANONICAL_TO_ALIASES.get(canonical, (canonical,))
    if canonical not in aliases:
        aliases = aliases + (canonical,)
    return aliases


def magnitude_column_names(band: str, prefix: str) -> Sequence[str]:
    names: List[str] = []
    canonical = canonical_band_name(band)
    roman_lightcone_name = ROMAN_LIGHTCONE_MAG_COLUMNS.get(canonical)
    if roman_lightcone_name is not None:
        names.extend(
            (
                roman_lightcone_name,
                f"{roman_lightcone_name}_dust",
                f"{roman_lightcone_name}_bulge",
            )
        )
    for alias in band_aliases(band):
        names.extend(
            (
                f"{prefix}{alias}",
                f"{prefix}{alias.lower()}",
                f"mag_{alias}",
                f"mag_{alias.lower()}",
                alias,
                alias.lower(),
            )
        )
    seen = set()
    ordered_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)
    return tuple(ordered_names)


def physical_kpc_to_arcsec(size_kpc: float, redshift: float) -> float:
    if size_kpc <= 0.0:
        return 0.0
    if Planck18 is not None and redshift > 0.0:
        kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(redshift).value / 60.0
        if kpc_per_arcsec > 0.0:
            return size_kpc / kpc_per_arcsec
    return size_kpc / 6.0


def deterministic_position_angle(row: Mapping[str, object]) -> float:
    seed_value = maybe_float(optional_value(row, "gal_id", "id", default=0.0), 0.0)
    return float((seed_value * 137.035999084) % 180.0)


def deterministic_seed(row: Mapping[str, object], salt: int = 0) -> int:
    halo = int(maybe_float(optional_value(row, "halo_id_nbody", default=0.0), 0.0))
    gal = int(maybe_float(optional_value(row, "gal_id", "id", default=0.0), 0.0))
    seed = (halo * 1103515245 + gal * 12345 + salt) % 2147483647
    return max(seed, 1)


def build_component(
    half_light_radius: float,
    flux: float,
    sersic_n: float,
    axis_ratio: float,
    position_angle_deg: float,
) -> galsim.GSObject:
    component = galsim.Sersic(n=sersic_n, half_light_radius=max(half_light_radius, 1.0e-4), flux=flux)
    axis_ratio = min(max(axis_ratio, 0.05), 1.0)
    return component.shear(q=axis_ratio, beta=position_angle_deg * galsim.degrees)


def build_lightcone_galaxy(
    row: Mapping[str, object],
    flux: float,
    disk_knot_fraction: float,
    disk_knot_count: int,
    disk_knot_radius_scale: float,
) -> galsim.GSObject:
    redshift = maybe_float(optional_value(row, "redshift", "z_nopec", default=0.5), 0.5)
    r_disk_kpc = maybe_float(optional_value(row, "r_disk", default=0.0), 0.0)
    r_bulge_kpc = maybe_float(optional_value(row, "rbulge", default=0.0), 0.0)
    mstar = maybe_float(optional_value(row, "mstar", default=0.0), 0.0)
    mbulge = maybe_float(optional_value(row, "mbulge", default=0.0), 0.0)
    cosi = abs(maybe_float(optional_value(row, "cosi", default=0.8), 0.8))
    disk_q = min(max(cosi, 0.1), 1.0)
    bulge_q = 0.8
    pa_deg = deterministic_position_angle(row)

    disk_hlr = 1.678 * physical_kpc_to_arcsec(r_disk_kpc, redshift)
    bulge_hlr = 1.678 * physical_kpc_to_arcsec(r_bulge_kpc, redshift)
    if disk_hlr <= 0.0 and bulge_hlr <= 0.0:
        return galsim.DeltaFunction(flux=flux)

    bulge_frac = 0.0
    if mstar > 0.0 and mbulge >= 0.0:
        bulge_frac = min(max(mbulge / mstar, 0.0), 1.0)
    elif bulge_hlr > 0.0 and disk_hlr <= 0.0:
        bulge_frac = 1.0
    elif bulge_hlr > 0.0:
        bulge_frac = 0.3

    components: List[galsim.GSObject] = []
    if bulge_frac > 0.0 and bulge_hlr > 0.0:
        components.append(build_component(bulge_hlr, flux * bulge_frac, 4.0, bulge_q, pa_deg))
    disk_frac = max(0.0, 1.0 - bulge_frac)
    if disk_frac > 0.0 and disk_hlr > 0.0:
        knot_fraction = min(max(disk_knot_fraction, 0.0), 1.0)
        smooth_disk_flux = flux * disk_frac * (1.0 - knot_fraction)
        knot_flux = flux * disk_frac * knot_fraction
        if smooth_disk_flux > 0.0:
            components.append(build_component(disk_hlr, smooth_disk_flux, 1.0, disk_q, pa_deg))
        if knot_flux > 0.0 and disk_knot_count > 0:
            rng = galsim.BaseDeviate(deterministic_seed(row, salt=1729))
            knots = galsim.RandomKnots(
                npoints=disk_knot_count,
                half_light_radius=max(disk_hlr * disk_knot_radius_scale, 1.0e-4),
                flux=knot_flux,
                rng=rng,
            ).shear(q=disk_q, beta=pa_deg * galsim.degrees)
            components.append(knots)
    if not components:
        hlr = max(disk_hlr, bulge_hlr, 0.05)
        components.append(build_component(hlr, flux, 1.0, disk_q, pa_deg))

    galaxy = components[0]
    for component in components[1:]:
        galaxy += component
    return galaxy


def build_galaxy(
    row: Mapping[str, object],
    flux: float,
    disk_knot_fraction: float = 0.0,
    disk_knot_count: int = 0,
    disk_knot_radius_scale: float = 0.8,
) -> galsim.GSObject:
    if "Roman_F158" in row or "r_disk" in row or "rbulge" in row:
        return build_lightcone_galaxy(
            row,
            flux,
            disk_knot_fraction=disk_knot_fraction,
            disk_knot_count=disk_knot_count,
            disk_knot_radius_scale=disk_knot_radius_scale,
        )

    profile_type = maybe_str(optional_value(row, "profile_type", "type", default="bulge_disk"))
    shared_pa = maybe_float(optional_value(row, "pa_deg", "theta_deg", default=0.0), 0.0)
    shared_q = maybe_float(optional_value(row, "q", "axis_ratio", default=0.8), 0.8)
    bulge_frac = maybe_float(optional_value(row, "bulge_frac", "b_over_t", default=0.3), 0.3)
    bulge_frac = min(max(bulge_frac, 0.0), 1.0)
    disk_frac = maybe_float(optional_value(row, "disk_frac", default=1.0 - bulge_frac), 1.0 - bulge_frac)
    if disk_frac < 0.0:
        disk_frac = max(0.0, 1.0 - bulge_frac)
    total_hlr = maybe_float(
        optional_value(row, "half_light_radius", "hlr", "rhalf_arcsec", default=0.18),
        0.18,
    )

    if profile_type.lower() in {"point", "star", "psf"}:
        return galsim.DeltaFunction(flux=flux)

    if profile_type.lower() in {"sersic", "single_sersic"}:
        sersic_n = maybe_float(optional_value(row, "sersic_n", "n", default=1.0), 1.0)
        return build_component(total_hlr, flux, sersic_n, shared_q, shared_pa)

    bulge_hlr = maybe_float(optional_value(row, "bulge_hlr", default=0.6 * total_hlr), 0.6 * total_hlr)
    disk_hlr = maybe_float(optional_value(row, "disk_hlr", default=1.3 * total_hlr), 1.3 * total_hlr)
    bulge_n = maybe_float(optional_value(row, "bulge_n", default=4.0), 4.0)
    disk_n = maybe_float(optional_value(row, "disk_n", default=1.0), 1.0)
    bulge_q = maybe_float(optional_value(row, "bulge_q", default=shared_q), shared_q)
    disk_q = maybe_float(optional_value(row, "disk_q", default=shared_q), shared_q)
    bulge_pa = maybe_float(optional_value(row, "bulge_pa_deg", default=shared_pa), shared_pa)
    disk_pa = maybe_float(optional_value(row, "disk_pa_deg", default=shared_pa), shared_pa)

    components: List[galsim.GSObject] = []
    if bulge_frac > 0.0:
        components.append(build_component(bulge_hlr, flux * bulge_frac, bulge_n, bulge_q, bulge_pa))
    if disk_frac > 0.0:
        components.append(build_component(disk_hlr, flux * disk_frac, disk_n, disk_q, disk_pa))
    if not components:
        components.append(build_component(total_hlr, flux, 1.0, shared_q, shared_pa))

    galaxy = components[0]
    for component in components[1:]:
        galaxy += component

    g1 = maybe_float(optional_value(row, "g1", default=0.0), 0.0)
    g2 = maybe_float(optional_value(row, "g2", default=0.0), 0.0)
    if g1 or g2:
        galaxy = galaxy.shear(g1=g1, g2=g2)

    dx_arcsec = maybe_float(optional_value(row, "dx_arcsec", default=0.0), 0.0)
    dy_arcsec = maybe_float(optional_value(row, "dy_arcsec", default=0.0), 0.0)
    if dx_arcsec or dy_arcsec:
        galaxy = galaxy.shift(dx_arcsec, dy_arcsec)
    return galaxy


def build_world_pos(row: Mapping[str, object], pointing: galsim.CelestialCoord) -> galsim.CelestialCoord:
    ra_deg = maybe_float(optional_value(row, "ra", "ra_deg"))
    dec_deg = maybe_float(optional_value(row, "dec", "dec_deg"))
    if ra_deg is not None and dec_deg is not None:
        return galsim.CelestialCoord(ra_deg * galsim.degrees, dec_deg * galsim.degrees)

    dra_arcsec = maybe_float(optional_value(row, "dra_arcsec", "x_arcsec", default=0.0), 0.0)
    ddec_arcsec = maybe_float(optional_value(row, "ddec_arcsec", "y_arcsec", default=0.0), 0.0)
    return pointing.deproject(dra_arcsec * galsim.arcsec, ddec_arcsec * galsim.arcsec, projection="gnomonic")


def build_wcs(
    sca: int,
    pointing: galsim.CelestialCoord,
    position_angle_deg: float,
    date: dt.datetime,
) -> galsim.BaseWCS:
    kwargs = {
        "world_pos": pointing,
        "PA": position_angle_deg * galsim.degrees,
        "date": date,
        "SCAs": [sca],
    }
    try:
        return roman.getWCS(**kwargs)[sca]
    except TypeError:
        kwargs["SCAs"] = sca
        try:
            wcs = roman.getWCS(**kwargs)
            return wcs[sca] if isinstance(wcs, dict) else wcs
        except TypeError:
            kwargs.pop("date", None)
            wcs = roman.getWCS(**kwargs)
            return wcs[sca] if isinstance(wcs, dict) else wcs


def get_bandpasses(filters: Iterable[str]) -> Dict[str, galsim.Bandpass]:
    try:
        bandpasses = roman.getBandpasses(AB_zeropoint=True)
    except TypeError:
        bandpasses = roman.getBandpasses()
        for name in list(bandpasses.keys()):
            if hasattr(bandpasses[name], "withZeropoint"):
                bandpasses[name] = bandpasses[name].withZeropoint("AB")
    return {band: bandpasses[canonical_band_name(band)] for band in filters}


def get_psf(sca: int, band: str) -> galsim.GSObject:
    canonical = canonical_band_name(band)
    try:
        return roman.getPSF(sca, canonical)
    except TypeError:
        return roman.getPSF(SCA=sca, bandpass=canonical)


def point_is_on_sca(image_pos: galsim.PositionD, padding_pixels: float = 0.0) -> bool:
    low = 0.5 - padding_pixels
    high = ROMAN_NATIVE_NPIX + 0.5 + padding_pixels
    return low <= image_pos.x <= high and low <= image_pos.y <= high


def sca_center_world_pos(wcs: galsim.BaseWCS) -> galsim.CelestialCoord:
    center = (ROMAN_NATIVE_NPIX + 1) / 2.0
    return wcs.toWorld(galsim.PositionD(center, center))


def image_position_to_stamp(
    obj: galsim.GSObject,
    wcs: galsim.BaseWCS,
    image_pos: galsim.PositionD,
    bandpass: galsim.Bandpass,
) -> galsim.Image:
    ix = int(math.floor(image_pos.x + 0.5))
    iy = int(math.floor(image_pos.y + 0.5))
    offset = galsim.PositionD(image_pos.x - ix, image_pos.y - iy)
    stamp = obj.drawImage(
        bandpass=bandpass,
        wcs=wcs.local(image_pos),
        method="no_pixel",
        offset=offset,
    )
    stamp.setCenter(ix, iy)
    return stamp


def write_fits_with_metadata(
    image: galsim.Image,
    output_path: Path,
    metadata: Mapping[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.write(str(output_path))

    if fits is None:
        return

    with fits.open(output_path, mode="update") as hdul:
        header = hdul[0].header
        for key, value in metadata.items():
            header[str(key).upper()[:8]] = value
        hdul.flush()


def render_filter_image(
    rows: Sequence[Mapping[str, object]],
    band: str,
    bandpass: galsim.Bandpass,
    wcs: galsim.BaseWCS,
    pointing: galsim.CelestialCoord,
    sca: int,
    include_psf: bool,
    include_pixel: bool,
    mag_prefix: str,
    verbose_footprint: bool = False,
    show_progress: bool = True,
    edge_padding_arcsec: float = 10.0,
    disk_knot_fraction: float = 0.2,
    disk_knot_count: int = 20,
    disk_knot_radius_scale: float = 0.8,
) -> tuple[galsim.ImageF, Dict[str, object]]:
    image = galsim.ImageF(ROMAN_NATIVE_NPIX, ROMAN_NATIVE_NPIX, wcs=wcs)
    image.setOrigin(1, 1)

    psf = get_psf(sca, band) if include_psf else None
    pixel = galsim.Pixel(scale=roman.pixel_scale) if include_pixel else None

    n_with_mag = 0
    sample_positions: List[str] = []
    on_sca_sources: List[Tuple[int, Dict[str, object], float, galsim.PositionD]] = []
    padding_pixels = max(edge_padding_arcsec, 0.0) / roman.pixel_scale

    row_iter = progress_iter(rows, total=len(rows), desc=f"Scanning {band}", enabled=show_progress)
    for idx, raw_row in enumerate(row_iter, start=1):
        progress_update(idx, len(rows), f"Scanning {band}", show_progress)
        row = normalize_catalog_row(raw_row)
        mag = maybe_float(optional_value(row, *magnitude_column_names(band, mag_prefix)))
        if mag is None:
            continue
        n_with_mag += 1

        world_pos = build_world_pos(row, pointing)
        image_pos = wcs.toImage(world_pos)
        if not np.isfinite(image_pos.x) or not np.isfinite(image_pos.y):
            continue

        if len(sample_positions) < 5:
            sample_positions.append(f"row {idx}: x={image_pos.x:.1f}, y={image_pos.y:.1f}")

        if not point_is_on_sca(image_pos, padding_pixels=padding_pixels):
            continue

        on_sca_sources.append((idx, row, mag, image_pos))

    draw_iter = progress_iter(
        on_sca_sources,
        total=len(on_sca_sources),
        desc=f"Drawing {band}",
        enabled=show_progress,
    )
    for draw_idx, (_idx, row, mag, image_pos) in enumerate(draw_iter, start=1):
        progress_update(draw_idx, len(on_sca_sources), f"Drawing {band}", show_progress)
        sed = galsim.SED(lambda wave: 1.0, wave_type="nm", flux_type="fphotons")
        sed = sed.withMagnitude(mag, bandpass) * roman.collecting_area
        obj = build_galaxy(
            row,
            1.0,
            disk_knot_fraction=disk_knot_fraction,
            disk_knot_count=disk_knot_count,
            disk_knot_radius_scale=disk_knot_radius_scale,
        ) * sed
        if psf is not None:
            obj = galsim.Convolve(obj, psf)
        if pixel is not None:
            obj = galsim.Convolve(obj, pixel)

        stamp = image_position_to_stamp(obj, wcs, image_pos, bandpass)
        overlap = stamp.bounds & image.bounds
        if overlap.isDefined():
            image[overlap] += stamp[overlap]

    diagnostics = {
        "band": band,
        "n_catalog": len(rows),
        "n_with_mag": n_with_mag,
        "n_on_sca": len(on_sca_sources),
        "edge_padding_arcsec": edge_padding_arcsec,
        "edge_padding_pixels": padding_pixels,
        "disk_knot_fraction": disk_knot_fraction,
        "disk_knot_count": disk_knot_count,
        "sample_positions": sample_positions,
    }
    if verbose_footprint:
        print(
            f"[{band}] catalog rows={len(rows)}, with_mag={n_with_mag}, "
            f"in_padded_footprint={len(on_sca_sources)}, edge_padding={edge_padding_arcsec:.2f} arcsec"
        )
        for line in sample_positions:
            print(f"[{band}] {line}")
    return image, diagnostics


def main() -> None:
    args = parse_args()
    rows = read_catalogs(args.catalog, args.catalog_format, show_progress=args.progress)
    pointing = galsim.CelestialCoord(args.pointing_ra * galsim.degrees, args.pointing_dec * galsim.degrees)
    date = dt.datetime.fromisoformat(args.date)
    filters = [canonical_band_name(band) for band in args.filters]
    bandpasses = get_bandpasses(filters)
    wcs = build_wcs(args.sca, pointing, args.position_angle, date)
    sca_center = sca_center_world_pos(wcs)
    print(
        f"SCA {args.sca} center sky position: RA={sca_center.ra / galsim.degrees:.8f} deg, Dec={sca_center.dec / galsim.degrees:.8f} deg"
    )

    for band in filters:
        image, diagnostics = render_filter_image(
            rows=rows,
            band=band,
            bandpass=bandpasses[band],
            wcs=wcs,
            pointing=pointing,
            sca=args.sca,
            include_psf=args.include_psf,
            include_pixel=args.include_pixel,
            mag_prefix=args.mag_prefix,
            verbose_footprint=args.verbose_footprint,
            show_progress=args.progress,
            edge_padding_arcsec=args.edge_padding_arcsec,
            disk_knot_fraction=args.disk_knot_fraction,
            disk_knot_count=args.disk_knot_count,
            disk_knot_radius_scale=args.disk_knot_radius_scale,
        )
        metadata = {
            "filter": band,
            "sca": args.sca,
            "pa_deg": args.position_angle,
            "dateobs": args.date,
            "ra_targ": args.pointing_ra,
            "dec_targ": args.pointing_dec,
            "nsource": len(rows),
            "nwithmag": diagnostics["n_with_mag"],
            "nonsca": diagnostics["n_on_sca"],
            "edgpad": diagnostics["edge_padding_arcsec"],
            "knfrac": diagnostics["disk_knot_fraction"],
            "nknots": diagnostics["disk_knot_count"],
            "psf": int(args.include_psf),
            "pixel": int(args.include_pixel),
        }
        output_path = args.output_dir / f"roman_ideal_sca{args.sca:02d}_{band}.fits"
        write_fits_with_metadata(image, output_path, metadata)
        print(f"Wrote {output_path}")
        print(
            f"[{band}] rendered {diagnostics['n_on_sca']} of {diagnostics['n_with_mag']} sources with magnitudes onto SCA {args.sca}"
        )


if __name__ == "__main__":
    main()
