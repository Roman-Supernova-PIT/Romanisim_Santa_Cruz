#!/usr/bin/env python3
"""Create a small example catalog centered on a chosen Roman SCA."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import galsim
from galsim import roman

ROMAN_NATIVE_NPIX = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an example catalog centered on a Roman SCA.")
    parser.add_argument("output", type=Path, help="Output CSV path.")
    parser.add_argument("--sca", type=int, default=7, help="Roman SCA number.")
    parser.add_argument("--pointing-ra", type=float, required=True, help="Boresight RA in degrees.")
    parser.add_argument("--pointing-dec", type=float, required=True, help="Boresight Dec in degrees.")
    parser.add_argument("--position-angle", type=float, default=0.0, help="Observatory position angle in degrees.")
    parser.add_argument("--date", default="2027-07-01T00:00:00", help="Observation datetime in ISO format.")
    return parser.parse_args()


def build_wcs(sca: int, pointing: galsim.CelestialCoord, position_angle_deg: float, date: dt.datetime):
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


def offset_world(center: galsim.CelestialCoord, dra_arcsec: float, ddec_arcsec: float) -> galsim.CelestialCoord:
    return center.deproject(dra_arcsec * galsim.arcsec, ddec_arcsec * galsim.arcsec, projection="gnomonic")


def main() -> None:
    args = parse_args()
    pointing = galsim.CelestialCoord(args.pointing_ra * galsim.degrees, args.pointing_dec * galsim.degrees)
    date = dt.datetime.fromisoformat(args.date)
    wcs = build_wcs(args.sca, pointing, args.position_angle, date)

    center_pix = (ROMAN_NATIVE_NPIX + 1) / 2.0
    sca_center = wcs.toWorld(galsim.PositionD(center_pix, center_pix))

    rows = [
        {
            "id": 1,
            "ra_deg": f"{(offset_world(sca_center, 0.0, 0.0).ra / galsim.degrees):.8f}",
            "dec_deg": f"{(offset_world(sca_center, 0.0, 0.0).dec / galsim.degrees):.8f}",
            "mag_H158": "23.0",
            "profile_type": "bulge_disk",
            "half_light_radius": "0.18",
            "bulge_frac": "0.35",
            "bulge_hlr": "0.10",
            "disk_hlr": "0.24",
            "bulge_n": "4.0",
            "disk_n": "1.0",
            "q": "0.72",
            "pa_deg": "18.0",
        },
        {
            "id": 2,
            "ra_deg": f"{(offset_world(sca_center, 22.0, -18.0).ra / galsim.degrees):.8f}",
            "dec_deg": f"{(offset_world(sca_center, 22.0, -18.0).dec / galsim.degrees):.8f}",
            "mag_H158": "23.7",
            "profile_type": "sersic",
            "half_light_radius": "0.11",
            "bulge_frac": "0.00",
            "bulge_hlr": "0.00",
            "disk_hlr": "0.00",
            "bulge_n": "0.0",
            "disk_n": "1.5",
            "q": "0.58",
            "pa_deg": "74.0",
        },
        {
            "id": 3,
            "ra_deg": f"{(offset_world(sca_center, -35.0, 28.0).ra / galsim.degrees):.8f}",
            "dec_deg": f"{(offset_world(sca_center, -35.0, 28.0).dec / galsim.degrees):.8f}",
            "mag_H158": "21.5",
            "profile_type": "point",
            "half_light_radius": "0.00",
            "bulge_frac": "0.00",
            "bulge_hlr": "0.00",
            "disk_hlr": "0.00",
            "bulge_n": "0.0",
            "disk_n": "0.0",
            "q": "1.00",
            "pa_deg": "0.0",
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {args.output} centered on SCA {args.sca}: "
        f"RA={sca_center.ra / galsim.degrees:.8f} deg, Dec={sca_center.dec / galsim.degrees:.8f} deg"
    )


if __name__ == "__main__":
    main()
