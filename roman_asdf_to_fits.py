#!/usr/bin/env python3
"""Convert Roman ASDF datamodel images to FITS for quick DS9 inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from astropy.io import fits

try:
    import roman_datamodels as rdm
except ImportError:  # pragma: no cover - optional dependency
    rdm = None

try:
    import asdf
except ImportError:  # pragma: no cover - optional dependency
    asdf = None


DEFAULT_CANDIDATES = (
    "data",
    "dq",
    "err",
    "var_poisson",
    "var_rnoise",
    "var_flat",
    "amp33",
    "resultants",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Roman ASDF image array to FITS.")
    parser.add_argument("input", type=Path, help="Input Roman ASDF file.")
    parser.add_argument("output", type=Path, nargs="?", help="Output FITS path.")
    parser.add_argument(
        "--array",
        default="auto",
        help="Array name to write. Defaults to first available science-like array.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="For 3D arrays such as L1 resultants, write this 0-based plane.",
    )
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="Write all planes of a 3D array as FITS image extensions.",
    )
    parser.add_argument(
        "--list-arrays",
        action="store_true",
        help="Print available array-like fields and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output FITS file.",
    )
    parser.add_argument(
        "--no-wcs",
        action="store_true",
        help="Do not try to translate ASDF GWCS metadata into FITS WCS keywords.",
    )
    return parser.parse_args()


def getattr_path(obj: Any, name: str) -> Any:
    current = obj
    for part in name.split("."):
        current = getattr(current, part)
    return current


def get_tree_value(tree: dict, name: str) -> Any:
    current: Any = tree
    for part in name.split("."):
        current = current[part]
    return current


def is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def open_model(path: Path) -> Any:
    if rdm is not None:
        return rdm.open(path)
    if asdf is not None:
        return asdf.open(path)
    raise RuntimeError("Install roman_datamodels or asdf to read Roman ASDF files.")


def close_model(model: Any) -> None:
    close = getattr(model, "close", None)
    if close is not None:
        close()


def iter_arrays_from_model(model: Any) -> Iterable[tuple[str, Any]]:
    for name in DEFAULT_CANDIDATES:
        try:
            value = getattr_path(model, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if is_array_like(value):
            yield name, value

    tree = getattr(model, "tree", None)
    if isinstance(tree, dict):
        for name in DEFAULT_CANDIDATES:
            try:
                value = get_tree_value(tree, name)
            except (KeyError, TypeError):
                continue
            if is_array_like(value):
                yield name, value


def find_array(model: Any, requested: str) -> tuple[str, Any]:
    arrays = list(iter_arrays_from_model(model))
    if requested != "auto":
        for name, value in arrays:
            if name == requested:
                return name, value
        try:
            value = getattr_path(model, requested)
        except (AttributeError, KeyError, TypeError):
            tree = getattr(model, "tree", None)
            if isinstance(tree, dict):
                value = get_tree_value(tree, requested)
            else:
                raise KeyError(f"Array field {requested!r} was not found.")
        if not is_array_like(value):
            raise TypeError(f"Field {requested!r} is not array-like.")
        return requested, value

    for name, value in arrays:
        if np.ndim(value) >= 2:
            return name, value
    raise RuntimeError("No 2D or 3D image-like array was found in the ASDF file.")


def output_path(input_path: Path, output: Optional[Path], array_name: str) -> Path:
    if output is not None:
        return output
    suffix = array_name.replace(".", "_")
    return input_path.with_name(f"{input_path.stem}_{suffix}.fits")


def fits_wcs_header(model: Any, data_shape: tuple[int, ...]) -> fits.Header:
    try:
        wcs = getattr_path(model, "meta.wcs")
    except (AttributeError, KeyError, TypeError):
        return fits.Header()

    if wcs is None or not hasattr(wcs, "to_fits_sip"):
        return fits.Header()

    ny, nx = data_shape[-2], data_shape[-1]
    bounding_box = ((0, nx - 1), (0, ny - 1))
    try:
        return wcs.to_fits_sip(bounding_box=bounding_box)
    except Exception as exc:
        print(f"Warning: could not convert ASDF GWCS to FITS-SIP WCS: {exc}")
        return fits.Header()


def apply_wcs_header(hdu: fits.ImageHDU, wcs_header: fits.Header) -> None:
    for key, value in wcs_header.items():
        if key in {"SIMPLE", "BITPIX", "EXTEND", "NAXIS", "NAXIS1", "NAXIS2"}:
            continue
        hdu.header[key] = value


def make_hdul(
    data: np.ndarray,
    source: Path,
    array_name: str,
    plane: Optional[int],
    all_slices: bool,
    wcs_header: fits.Header,
) -> fits.HDUList:
    primary = fits.PrimaryHDU()
    primary.header["SRCASDF"] = str(source)
    primary.header["ARRNAME"] = array_name

    if data.ndim == 2:
        image = fits.ImageHDU(data=np.asarray(data), name=array_name.upper()[:8])
        apply_wcs_header(image, wcs_header)
        return fits.HDUList([primary, image])

    if data.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array, got shape {data.shape}.")

    if all_slices:
        hdus = [primary]
        for idx in range(data.shape[0]):
            hdu = fits.ImageHDU(data=np.asarray(data[idx]), name=f"SLICE{idx:03d}"[:8])
            hdu.header["PLANE"] = idx
            apply_wcs_header(hdu, wcs_header)
            hdus.append(hdu)
        return fits.HDUList(hdus)

    if plane is None:
        plane = data.shape[0] - 1
    image = fits.ImageHDU(data=np.asarray(data[plane]), name=array_name.upper()[:8])
    image.header["PLANE"] = plane
    image.header["NPLANE"] = data.shape[0]
    apply_wcs_header(image, wcs_header)
    return fits.HDUList([primary, image])


def main() -> None:
    args = parse_args()
    model = open_model(args.input)
    try:
        arrays = list(iter_arrays_from_model(model))
        if args.list_arrays:
            for name, value in arrays:
                print(f"{name}: shape={value.shape}, dtype={value.dtype}")
            return

        array_name, value = find_array(model, args.array)
        data = np.asarray(value)
        output = output_path(args.input, args.output, array_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        wcs_header = fits.Header() if args.no_wcs else fits_wcs_header(model, data.shape)
        hdul = make_hdul(data, args.input, array_name, args.slice, args.all_slices, wcs_header)
        hdul.writeto(output, overwrite=args.overwrite)
        print(f"Wrote {output} from {array_name} shape={data.shape}")
    finally:
        close_model(model)


if __name__ == "__main__":
    main()
