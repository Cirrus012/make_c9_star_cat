#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse_target_location.py  ——————————————————————————————————————————————

Convert chip‑local pixel coordinates (x_chip, y_chip) on a CSST WFI CMOS chip
into ICRS (RA, Dec) using an *equatorial* gnomonic inverse projection.

Key properties of this version
------------------------------
• **Single source of truth** for pixel scale: ARCSEC_PER_PIX = 0.074 by default.
• **CHIP_OFFSET is generated at runtime** from the FPA geometry (6×5 chips),
  fully consistent with target_location_check.py. No guessing / manual tables.
• **Units are explicit**:
    - chip coordinates (x_chip, y_chip): pixel (origin at chip corner)
    - focal‑plane global (x_img, y_img): pixel (origin at mosaic centre)
    - tangent plane (xi, eta): radians
• **Robust inverse gnomonic**: protects ρ→0 to avoid division by zero.
• Clean CLI for quick tests.

If your adopted pixel scale is not 0.074″/pix (e.g. a lab value), change
ARCSEC_PER_PIX below and everything else follows.
"""
### Known issues:
### Y-aix upside down!

from __future__ import annotations

import numpy as np
from math import radians, sin, cos
from typing import Dict, Tuple

# ———————————————————————— single source of truth ————————————————————————
# Pixel scale and plate constants. Adjust ARCSEC_PER_PIX if needed.
PIX_SIZE_UM: float    = 10.0        # µm per pixel (flight CMOS nominal)
ARCSEC_PER_PIX: float = 0.074       # arcsec / pixel (adopted nominal)
PLATE_SCALE: float    = ARCSEC_PER_PIX / PIX_SIZE_UM  # arcsec / µm
DEG_PER_PIX: float    = ARCSEC_PER_PIX / 3600.0       # deg / pixel
RAD_PER_PIX: float    = np.deg2rad(DEG_PER_PIX)       # rad / pixel

# ———————————————————————— FPA geometry (from target_location_check.py) ————
# Full mosaic is 6 (cols) × 5 (rows) = 30 chips.
# X direction has two kinds of inter‑column gaps; Y direction has single gap.
NCOL, NROW = 6, 5
XCHIP, YCHIP = 9216, 9232          # chip size (pixels)
XGAP1, XGAP2 = 534, 1309           # two different column gaps (pixels)
YGAP = 898                         # row gap (pixels)

# Helpers to reproduce target_location_check.getChipLim() behaviour
# NOTE: chip IDs here are 1..30 (column index varies fastest, RTL ordering).

def _chip_edges(chip_id: int) -> Tuple[float, float, float, float]:
    """Return (nx0, nx1, ny0, ny1) in the *mosaic‑global* pixel frame.
    Ported to match target_location_check geometry.
    Origin is the mosaic centre; +x right, +y down (same as official ccdLayout after invert_yaxis).
    """
    if not (1 <= chip_id <= NCOL * NROW):
        raise ValueError(f"chip_id must be in 1..{NCOL*NROW}, got {chip_id}")

    # 1‑based row/column indices; column indexing is right‑to‑left
    row_id = ((chip_id - 1) % NROW) + 1
    col_id = NCOL - ((chip_id - 1) // NROW)

    # X limits — symmetric positions with gap pattern adjustments
    xrem = 2 * (col_id - 1) - (NCOL - 1)   # e.g. for 6 cols → −5,−3,−1,1,3,5
    xcen = (XCHIP/2 + XGAP1/2) * xrem
    # edge/transition columns have the wider XGAP2 instead of XGAP1
    if chip_id >= 26 or chip_id == 21:
        xcen -= (XGAP2 - XGAP1)
    if chip_id <= 5 or chip_id == 10:
        xcen += (XGAP2 - XGAP1)
    nx0 = xcen - XCHIP/2
    nx1 = xcen + XCHIP/2 - 1

    # Y limits — evenly spaced rows with a single YGAP
    yrem = (row_id - 1) - (NROW // 2)      # for 5 rows → −2,−1,0,1,2
    ycen = (YCHIP + YGAP) * yrem
    ny0 = ycen - YCHIP/2
    ny1 = ycen + YCHIP/2 - 1

    return nx0, nx1, ny0, ny1


def _chip_center(chip_id: int) -> Tuple[float, float]:
    nx0, nx1, ny0, ny1 = _chip_edges(chip_id)
    return (nx0 + nx1) / 2.0, (ny0 + ny1) / 2.0


def build_chip_offset() -> Dict[int, Tuple[float, float]]:
    """Build CHIP_OFFSET = {chip_id: (xc, yc)} with centres in mosaic pixels."""
    return {cid: _chip_center(cid) for cid in range(1, NCOL*NROW + 1)}


CHIP_OFFSET: Dict[int, Tuple[float, float]] = build_chip_offset()

# Optional sanity check (can be disabled if desired)
for cid in range(1, NCOL*NROW + 1):
    nx0, nx1, ny0, ny1 = _chip_edges(cid)
    xc, yc = CHIP_OFFSET[cid]
    assert abs(xc - (nx0 + nx1) / 2) < 1e-6
    assert abs(yc - (ny0 + ny1) / 2) < 1e-6

# ————————————————————————— transforms ————————————————————————————————

def chip_to_focal(chip_id: int, x_chip: float, y_chip: float) -> Tuple[float, float]:
    """Chip‑local (x,y) → focal‑plane mosaic (x_img,y_img) in pixels.
    chip (origin at chip *corner*) → add chip centre offset, then minus half size
    to place the corner at the correct global origin.
    Equivalently, x_img = nx0 + x_chip, y_img = ny0 + y_chip.
    """
    nx0, nx1, ny0, ny1 = _chip_edges(chip_id)
    return nx0 + x_chip, ny0 + y_chip


def focal_to_chip(chip_id: int, x_img: float, y_img: float) -> Tuple[float, float]:
    """Inverse of chip_to_focal: mosaic global pixel → chip‑local pixel."""
    nx0, nx1, ny0, ny1 = _chip_edges(chip_id)
    return x_img - nx0, y_img - ny0


def focal_to_plane(x_img: float, y_img: float, theta_deg: float) -> Tuple[float, float]:
    """Mosaic pixels -> tangent-plane (xi, eta) in radians.

    IMPORTANT:
    - This is the convention that matches the *official* target_location_check.py as used by
      verify_inverse_vs_official.py in your repo.
    - The linear part here is NOT a pure rotation matrix; its inverse equals itself.
    """
    th = radians(theta_deg)
    c, s = np.cos(th), np.sin(th)

    # Tangent-plane coordinates in *pixel* units
    xi_pix  =  x_img * c - y_img * s
    eta_pix = -x_img * s - y_img * c
    return xi_pix * RAD_PER_PIX, eta_pix * RAD_PER_PIX

def plane_to_focal(xi_rad: float, eta_rad: float, theta_deg: float) -> Tuple[float, float]:
    """Exact inverse of focal_to_plane: tangent-plane radians -> mosaic pixels."""
    th = radians(theta_deg)
    c, s = np.cos(th), np.sin(th)

    xi_pix  = xi_rad / RAD_PER_PIX
    eta_pix = eta_rad / RAD_PER_PIX

    # Inverse of [[c, -s], [-s, -c]] is itself.
    x_img =  xi_pix * c - eta_pix * s
    y_img = -xi_pix * s - eta_pix * c
    return x_img, y_img

def inv_gnomonic_eq(ra0_deg: float, dec0_deg: float,
                    xi_rad, eta_rad) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse gnomonic on the celestial sphere (equatorial formulation).

    Parameters
    ----------
    ra0_deg, dec0_deg : float
        Tangent point (ICRS) in degrees.
    xi_rad, eta_rad : array‑like
        Tangent‑plane coordinates in radians.

    Returns
    -------
    (ra_deg, dec_deg) : arrays (same shape as xi/eta)
    """
    RA0  = np.deg2rad(ra0_deg)
    DEC0 = np.deg2rad(dec0_deg)
    xi   = np.asanyarray(xi_rad, dtype=float)
    eta  = np.asanyarray(eta_rad, dtype=float)

    rho  = np.hypot(xi, eta)
    c    = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)

    # Protect ρ=0 to avoid 0/0
    out_ra  = np.empty_like(rho)
    out_dec = np.empty_like(rho)
    m = rho > 0
    # Non‑centre pixels
    out_dec[m] = np.arcsin(cos_c[m] * np.sin(DEC0) + eta[m] * sin_c[m] * np.cos(DEC0) / rho[m])
    out_ra[m]  = RA0 + np.arctan2(xi[m] * sin_c[m],
                                  rho[m] * np.cos(DEC0) * cos_c[m] - eta[m] * np.sin(DEC0) * sin_c[m])
    # Exact centre → (ra0, dec0)
    out_dec[~m] = DEC0
    out_ra[~m]  = RA0

    ra_deg  = (np.rad2deg(out_ra) + 360.0) % 360.0
    dec_deg =  np.rad2deg(out_dec)
    return ra_deg, dec_deg


# —————————————————————— high‑level wrappers ————————————————————————

def ccd_to_radec(chip_id: int,
                 x_chip: float,
                 y_chip: float,
                 ra0_deg: float,
                 dec0_deg: float,
                 theta_deg: float) -> Tuple[float, float]:
    """Chip pixel → ICRS RA/Dec (degrees).

    Steps: chip‑local → focal mosaic (pixels) → tangent plane (radians)
           → ICRS via inverse gnomonic.
    """
    x_img, y_img       = chip_to_focal(chip_id, x_chip, y_chip)
    xi_rad, eta_rad    = focal_to_plane(x_img, y_img, theta_deg)
    ra_deg, dec_deg    = inv_gnomonic_eq(ra0_deg, dec0_deg, xi_rad, eta_rad)
    return float(ra_deg), float(dec_deg)


def radec_to_ccd(chip_id: int,
                 ra_deg: float,
                 dec_deg: float,
                 ra0_deg: float,
                 dec0_deg: float,
                 theta_deg: float) -> Tuple[float, float]:
    """Inverse of ccd_to_radec: ICRS (RA,Dec) → chip‑local pixels.
    Useful for checks and forward simulations.
    """
    # Forward gnomonic (equatorial)
    RA0  = np.deg2rad(ra0_deg)
    DEC0 = np.deg2rad(dec0_deg)
    RA   = np.deg2rad(ra_deg)
    DEC  = np.deg2rad(dec_deg)

    cosc = np.sin(DEC0) * np.sin(DEC) + np.cos(DEC0) * np.cos(DEC) * np.cos(RA - RA0)
    # ξ,η in radians
    xi   =  (np.cos(DEC) * np.sin(RA - RA0)) / cosc
    eta  =  (np.cos(DEC0) * np.sin(DEC) - np.sin(DEC0) * np.cos(DEC) * np.cos(RA - RA0)) / cosc

    # Back to focal mosaic pixels
    x_img, y_img = plane_to_focal(xi, eta, theta_deg)

    # To chip‑local pixels
    x_chip, y_chip = focal_to_chip(chip_id, x_img, y_img)
    return float(x_chip), float(y_chip)


# ————————————————————————————— CLI ————————————————————————————————
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Chip pixel ↔ RA/Dec via equatorial gnomonic (self‑consistent).")
    sub = p.add_subparsers(dest="cmd", required=False)

    # pixel → sky
    p1 = sub.add_parser("pix2sky", help="chip pixel → RA/Dec")
    p1.add_argument("chip", type=int, help="chip id (1..30)")
    p1.add_argument("x", type=float, help="x in chip pixels")
    p1.add_argument("y", type=float, help="y in chip pixels")
    p1.add_argument("ra0", type=float, help="FOV centre RA (deg, ICRS)")
    p1.add_argument("dec0", type=float, help="FOV centre Dec (deg, ICRS)")
    p1.add_argument("--theta", type=float, default=-113.4333,
                    help="camera roll CCW from sky East (deg)")

    # sky → pixel
    p2 = sub.add_parser("sky2pix", help="RA/Dec → chip pixel")
    p2.add_argument("chip", type=int, help="chip id (1..30)")
    p2.add_argument("ra", type=float, help="RA (deg, ICRS)")
    p2.add_argument("dec", type=float, help="Dec (deg, ICRS)")
    p2.add_argument("ra0", type=float, help="FOV centre RA (deg, ICRS)")
    p2.add_argument("dec0", type=float, help="FOV centre Dec (deg, ICRS)")
    p2.add_argument("--theta", type=float, default=-113.4333,
                    help="camera roll CCW from sky East (deg)")

    args = p.parse_args()
    if args.cmd == "sky2pix":
        x, y = radec_to_ccd(args.chip, args.ra, args.dec, args.ra0, args.dec0, args.theta)
        print(f"CCD {args.chip}: RA={args.ra:.6f}°, Dec={args.dec:.6f}° → pixel ({x:.3f}, {y:.3f})")
    else:
        # default to pix2sky for backward compatibility
        chip = getattr(args, "chip", None)
        if chip is None:
            p.print_help()
            raise SystemExit(2)
        ra, dec = ccd_to_radec(args.chip, args.x, args.y, args.ra0, args.dec0, args.theta)
        print(f"CCD {args.chip}: pixel ({args.x:.3f},{args.y:.3f}) → RA={ra:.6f}°, Dec={dec:.6f}°")
