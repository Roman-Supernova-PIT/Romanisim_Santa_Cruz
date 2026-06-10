#!/usr/bin/env python3
"""Simple point-kernel drizzle combiner for simulated Roman images."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class InputInfo:
    filename: Path
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    extension: int | str


def parse_fill_value(value: str) -> float:
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine FITS images onto a 2x oversampled grid with point-kernel drizzle."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, help="Input FITS images.")
    parser.add_argument("--output", type=Path, help="Output drizzled FITS image.")
    parser.add_argument("--extension", default=0, help="FITS image extension to read; default 0.")
    parser.add_argument("--weight-mode", choices=("uniform", "ivar"), default="uniform")
    parser.add_argument(
        "--weight-key",
        default=None,
        help="For ivar mode, FITS extension name/index or filename pattern for inverse-variance weights.",
    )
    parser.add_argument("--fill-value", type=parse_fill_value, default=np.nan)
    parser.add_argument("--make-count-map", action="store_true")
    parser.add_argument("--make-weight-map", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of finite pixels to transform at once; default 1000000.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run a small synthetic drizzle test.")
    return parser.parse_args()


def parse_extension(extension: int | str) -> int | str:
    if isinstance(extension, int):
        return extension
    try:
        return int(extension)
    except ValueError:
        return extension


def read_image_and_wcs(filename: Path, extension: int | str = 0) -> InputInfo:
    ext = parse_extension(extension)
    with fits.open(filename) as hdul:
        hdu = hdul[ext]
        if hdu.data is None:
            raise ValueError(f"{filename}[{extension}] does not contain image data.")
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
        wcs = WCS(header)
    if data.ndim != 2:
        raise ValueError(f"{filename}[{extension}] must be a 2D image, got shape {data.shape}.")
    if not wcs.has_celestial:
        raise ValueError(f"{filename}[{extension}] does not have a celestial FITS WCS.")
    return InputInfo(filename=filename, data=data, header=header, wcs=wcs, extension=ext)


def pixel_scale_matrix_half(wcs: WCS, oversample: int) -> np.ndarray:
    celestial = wcs.celestial
    if celestial.wcs.has_cd():
        return celestial.wcs.cd / oversample
    pc = celestial.wcs.get_pc()
    cdelt = celestial.wcs.cdelt
    return pc @ np.diag(cdelt / oversample)


def image_corner_pixels(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5], dtype=float)
    y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=float)
    return x, y


def make_output_wcs_and_shape(input_infos: Sequence[InputInfo], oversample: int = 2) -> tuple[WCS, tuple[int, int]]:
    if not input_infos:
        raise ValueError("At least one input image is required.")

    reference = input_infos[0]
    ref_ny, ref_nx = reference.data.shape
    ref_cx = (ref_nx - 1) / 2.0
    ref_cy = (ref_ny - 1) / 2.0
    ref_ra, ref_dec = reference.wcs.pixel_to_world_values(ref_cx, ref_cy)

    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output_wcs.wcs.cunit = ["deg", "deg"]
    output_wcs.wcs.radesys = reference.wcs.celestial.wcs.radesys or "ICRS"
    output_wcs.wcs.crval = [ref_ra, ref_dec]
    output_wcs.wcs.crpix = [1.0, 1.0]
    output_wcs.wcs.cd = pixel_scale_matrix_half(reference.wcs, oversample)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for info in input_infos:
        xin, yin = image_corner_pixels(info.data.shape)
        sky = info.wcs.pixel_to_world(xin, yin)
        xout, yout = output_wcs.world_to_pixel(sky)
        all_x.append(np.asarray(xout, dtype=float))
        all_y.append(np.asarray(yout, dtype=float))

    xcat = np.concatenate(all_x)
    ycat = np.concatenate(all_y)
    xmin = np.floor(np.nanmin(xcat)) - 1
    ymin = np.floor(np.nanmin(ycat)) - 1
    xmax = np.ceil(np.nanmax(xcat)) + 1
    ymax = np.ceil(np.nanmax(ycat)) + 1

    output_wcs.wcs.crpix[0] -= xmin
    output_wcs.wcs.crpix[1] -= ymin
    nx = int(xmax - xmin + 1)
    ny = int(ymax - ymin + 1)
    return output_wcs, (ny, nx)


def load_weight_image(input_filename: Path, weight_key: Optional[str] = None) -> np.ndarray:
    if weight_key is None:
        raise ValueError("weight-mode='ivar' requires --weight-key.")

    if "{" in weight_key:
        weight_filename = Path(weight_key.format(input=input_filename, stem=input_filename.stem))
        with fits.open(weight_filename) as hdul:
            return np.asarray(hdul[0].data, dtype=float)

    try:
        ext: int | str = int(weight_key)
    except ValueError:
        ext = weight_key

    try:
        with fits.open(input_filename) as hdul:
            return np.asarray(hdul[ext].data, dtype=float)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Could not load inverse-variance weights from {input_filename}[{weight_key!r}].") from exc


def drizzle_point_kernel(
    input_infos: Sequence[InputInfo],
    output_wcs: WCS,
    output_shape: tuple[int, int],
    weight_mode: str = "uniform",
    weight_key: Optional[str] = None,
    fill_value: float = np.nan,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    ny_out, nx_out = output_shape
    numerator = np.zeros(output_shape, dtype=float)
    denominator = np.zeros(output_shape, dtype=float)
    count = np.zeros(output_shape, dtype=np.int64)

    for info in input_infos:
        data = info.data
        finite = np.isfinite(data)
        if weight_mode == "uniform":
            weights = np.ones_like(data, dtype=float)
        elif weight_mode == "ivar":
            weights = load_weight_image(info.filename, weight_key=weight_key)
            if weights.shape != data.shape:
                raise ValueError(
                    f"Weight image for {info.filename} has shape {weights.shape}, expected {data.shape}."
                )
            finite &= np.isfinite(weights) & (weights > 0)
        else:
            raise ValueError(f"Unsupported weight mode {weight_mode!r}.")

        ypix, xpix = np.nonzero(finite)
        if len(xpix) == 0:
            continue

        for start in range(0, len(xpix), chunk_size):
            stop = min(start + chunk_size, len(xpix))
            xchunk = xpix[start:stop]
            ychunk = ypix[start:stop]

            sky = info.wcs.pixel_to_world(xchunk.astype(float), ychunk.astype(float))
            xout, yout = output_wcs.world_to_pixel(sky)
            xround = np.rint(xout).astype(np.int64)
            yround = np.rint(yout).astype(np.int64)
            inside = (xround >= 0) & (xround < nx_out) & (yround >= 0) & (yround < ny_out)
            if not np.any(inside):
                continue

            xo = xround[inside]
            yo = yround[inside]
            vals = data[ychunk[inside], xchunk[inside]]
            w = weights[ychunk[inside], xchunk[inside]]
            np.add.at(numerator, (yo, xo), w * vals)
            np.add.at(denominator, (yo, xo), w)
            np.add.at(count, (yo, xo), 1)

    output = np.full(output_shape, fill_value, dtype=float)
    populated = denominator > 0
    output[populated] = numerator[populated] / denominator[populated]
    return output, count, denominator


def write_output(
    filename: Path,
    image: np.ndarray,
    wcs: WCS,
    count: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    header_info: Optional[dict] = None,
) -> None:
    header = wcs.to_header()
    if header_info:
        for key, value in header_info.items():
            header[key] = value
    primary = fits.PrimaryHDU(data=image, header=header)
    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = [primary]
    if count is not None:
        hdus.append(fits.ImageHDU(data=count.astype(np.int64), name="COUNT"))
    if weight is not None:
        hdus.append(fits.ImageHDU(data=weight, name="WEIGHT"))
    filename.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def make_test_wcs(ra: float, dec: float, xshift: float = 0.0, yshift: float = 0.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [5.0 + xshift, 5.0 + yshift]
    wcs.wcs.cd = np.array([[-0.11 / 3600.0, 0.0], [0.0, 0.11 / 3600.0]])
    return wcs


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        inputs: list[Path] = []
        for idx, shift in enumerate([(0.0, 0.0), (0.35, -0.2), (-0.25, 0.3)], start=1):
            data = np.full((10, 10), 7.5, dtype=float)
            wcs = make_test_wcs(10.0, 20.0, xshift=shift[0], yshift=shift[1])
            path = tmp / f"test_{idx}.fits"
            fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(path)
            inputs.append(path)

        infos = [read_image_and_wcs(path) for path in inputs]
        out_wcs, out_shape = make_output_wcs_and_shape(infos, oversample=2)
        image, count, weight = drizzle_point_kernel(infos, out_wcs, out_shape)
        output = Path("drizzle_self_test_output.fits")
        write_output(
            output,
            image,
            out_wcs,
            count=count,
            weight=weight,
            header_info={"DRIZZLE": True, "OSAMP": 2, "KERNEL": "point", "NINPUT": len(infos)},
        )

        in_scale = np.sqrt(abs(np.linalg.det(infos[0].wcs.pixel_scale_matrix)))
        out_scale = np.sqrt(abs(np.linalg.det(out_wcs.pixel_scale_matrix)))
        if not np.isclose(out_scale, in_scale / 2.0, rtol=1e-6):
            raise AssertionError("Output pixel scale is not 2x finer than input.")
        if not np.any(count > 0):
            raise AssertionError("Count map has no populated pixels.")
        populated = count > 0
        if not np.allclose(image[populated], 7.5):
            raise AssertionError("Constant-valued inputs did not preserve their value.")
        print(f"Self-test passed. Wrote {output}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    if not args.inputs:
        raise SystemExit("--inputs is required unless --self-test is set.")
    if args.output is None:
        raise SystemExit("--output is required unless --self-test is set.")

    infos = [read_image_and_wcs(path, extension=args.extension) for path in args.inputs]
    output_wcs, output_shape = make_output_wcs_and_shape(infos, oversample=2)
    image, count, weight = drizzle_point_kernel(
        infos,
        output_wcs,
        output_shape,
        weight_mode=args.weight_mode,
        weight_key=args.weight_key,
        fill_value=args.fill_value,
        chunk_size=args.chunk_size,
    )
    write_output(
        args.output,
        image,
        output_wcs,
        count=count if args.make_count_map else None,
        weight=weight if args.make_weight_map else None,
        header_info={"DRIZZLE": True, "OSAMP": 2, "KERNEL": "point", "NINPUT": len(infos)},
    )
    print(f"Wrote {args.output} shape={image.shape}")


if __name__ == "__main__":
    main()
