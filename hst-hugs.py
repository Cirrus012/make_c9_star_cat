#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append HST HUGS ASCII catalogs to an existing CSST C9-style HDF5.

Policy (as requested):
- NO dedupe within HST inputs (even if meth1/2/3 are all provided).
- ONLY dedupe new HST entries against existing stars already in the HDF5.
  -> We drop the HST rows that match existing HDF5 stars (keep Gaia/original).

HDF5 layout:
  /star_catalog/<healpix_id>/{RA,DEC,app_sdss_g,teff,grav,z_met,AV,DM,mass,pmra,pmdec,RV,parallax}
"""

import argparse
import math
import numpy as np
import h5py
import healpy as hp
from astropy.table import Table

# -------------------------
# Column detection helpers
# -------------------------
def _norm(s: str) -> str:
    return s.strip().lower()

def detect_ra_dec(tab: Table):
    cols = tab.colnames
    lcols = {_norm(c): c for c in cols}

    # common variants
    ra_keys  = ["ra", "ra_deg", "radeg", "alpha_j2000", "alphaj2000", "ra2000"]
    dec_keys = ["dec", "dec_deg", "decdeg", "delta_j2000", "deltaj2000", "dec2000"]

    for rk in ra_keys:
        if rk in lcols:
            ra_col = lcols[rk]
            break
    else:
        # fallback: first column containing "ra"
        ra_col = next((c for c in cols if "ra" in _norm(c)), None)
    for dk in dec_keys:
        if dk in lcols:
            dec_col = lcols[dk]
            break
    else:
        dec_col = next((c for c in cols if "dec" in _norm(c)), None)

    if ra_col is None or dec_col is None:
        raise KeyError(f"Cannot find RA/DEC columns. First 30 columns: {cols[:30]}")

    ra = np.array(tab[ra_col], dtype=float)
    dec = np.array(tab[dec_col], dtype=float)
    return ra, dec, ra_col, dec_col

def detect_mag(tab: Table, mag_pref="f606w"):
    """
    Choose one magnitude column as a proxy for app_sdss_g.
    Prefer a column containing mag_pref (default f606w), else fall back among common HUGS filters.
    """
    cols = tab.colnames
    lpref = _norm(mag_pref)

    def pick_by_substr(sub):
        sub = _norm(sub)
        # Prefer columns that contain both filter name and "mag"
        cand = []
        for c in cols:
            lc = _norm(c)
            if sub in lc and ("mag" in lc or "m_" in lc):
                cand.append(c)
        if cand:
            # stable choice: shortest name first (usually the main mag column)
            cand.sort(key=lambda x: (len(x), x))
            return cand[0]
        # second chance: any column containing the filter substring
        cand2 = [c for c in cols if sub in _norm(c)]
        if cand2:
            cand2.sort(key=lambda x: (len(x), x))
            return cand2[0]
        return None

    for filt in [lpref, "f606w", "f814w", "f438w", "f336w", "f275w"]:
        col = pick_by_substr(filt)
        if col is not None:
            return np.array(tab[col], dtype=float), col

    # last resort: any "mag" column
    mag_cols = [c for c in cols if "mag" in _norm(c)]
    if mag_cols:
        mag_cols.sort(key=lambda x: (len(x), x))
        col = mag_cols[0]
        return np.array(tab[col], dtype=float), col

    return None, None

# -------------------------
# Geometry + matching
# -------------------------
def unitvec_from_radec_deg(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T

def chord_radius_from_tol_arcsec(tol_arcsec):
    theta = np.deg2rad(tol_arcsec / 3600.0)
    return 2.0 * np.sin(theta / 2.0)

def filter_against_existing_h5(f, hpix_id, ra_new, dec_new, tol_arcsec):
    """
    Keep only those new points that have NO neighbor in existing HDF5 within tol_arcsec,
    in the same healpix group. (So Gaia/original stays, HST duplicate is removed.)
    """
    root = f.require_group("star_catalog")
    gid = str(int(hpix_id))
    if gid not in root:
        return np.ones(len(ra_new), dtype=bool)

    grp = root[gid]
    if "RA" not in grp or "DEC" not in grp:
        return np.ones(len(ra_new), dtype=bool)

    ra_old = np.array(grp["RA"][:], dtype=float)
    dec_old = np.array(grp["DEC"][:], dtype=float)
    if ra_old.size == 0:
        return np.ones(len(ra_new), dtype=bool)

    # KDTree on unit sphere, fallback to coarse grid if scipy is missing
    try:
        from scipy.spatial import cKDTree
        vold = unitvec_from_radec_deg(ra_old, dec_old)
        vnew = unitvec_from_radec_deg(ra_new, dec_new)
        tree = cKDTree(vold)
        r = chord_radius_from_tol_arcsec(tol_arcsec)
        idxs = tree.query_ball_point(vnew, r)
        return np.array([len(nei) == 0 for nei in idxs], dtype=bool)
    except Exception:
        # coarse grid fallback (OK for small fields)
        ra_n = np.asarray(ra_new, dtype=float)
        dec_n = np.asarray(dec_new, dtype=float)
        ra_o = np.asarray(ra_old, dtype=float)
        dec_o = np.asarray(dec_old, dtype=float)

        x_o = ra_o * np.cos(np.deg2rad(dec_o)) * 3600.0
        y_o = dec_o * 3600.0
        qx_o = np.round(x_o / tol_arcsec).astype(np.int64)
        qy_o = np.round(y_o / tol_arcsec).astype(np.int64)
        old_keys = set(zip(qx_o, qy_o))

        x_n = ra_n * np.cos(np.deg2rad(dec_n)) * 3600.0
        y_n = dec_n * 3600.0
        qx_n = np.round(x_n / tol_arcsec).astype(np.int64)
        qy_n = np.round(y_n / tol_arcsec).astype(np.int64)

        return np.array([(k not in old_keys) for k in zip(qx_n, qy_n)], dtype=bool)

# -------------------------
# HDF5 append
# -------------------------
def ensure_dataset_append(group, name, data, dtype="f8"):
    data = np.asarray(data)
    if name not in group:
        group.create_dataset(name, data=data.astype(dtype), maxshape=(None,))
    else:
        ds = group[name]
        old = ds.shape[0]
        ds.resize((old + len(data),))
        ds[old:] = data.astype(dtype)

# -------------------------
# main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Append HUGS txt catalogs to an existing C9 HDF5; dedupe ONLY against existing HDF5 stars."
    )
    ap.add_argument("--in_txt", nargs="+", required=True, help="Input HUGS catalog txt files (meth1/2/3).")
    ap.add_argument("--out_h5", required=True, help="Existing output HDF5 to append into.")
    ap.add_argument("--nside", type=int, default=128, help="healpix NSIDE (default 128)")
    ap.add_argument("--nest", action="store_true", help="use NEST indexing (default RING)")
    ap.add_argument("--mag_pref", default="f606w", help="Preferred filter substring for magnitude (default f606w)")
    ap.add_argument("--tol_arcsec", type=float, default=0.20, help="Cross-source dedupe tolerance in arcsec (default 0.20)")
    ap.add_argument("--no_dedupe_h5", action="store_true", help="Disable dedupe against existing HDF5 (append all).")

    # level0-like fallback values for missing physical columns
    ap.add_argument("--teff_log10K", type=float, default=math.log10(5000.0))
    ap.add_argument("--logg", type=float, default=4.5)
    ap.add_argument("--feh", type=float, default=-0.2)
    ap.add_argument("--mass", type=float, default=1.0)
    return ap.parse_args()

def read_one_txt(path, mag_pref):
    """
    HUGS meth catalog has no column names; use fixed column indices defined in the header.

    1-based column mapping (from your header):
      3  : F275W mag
      9  : F336W mag
      15 : F435W mag   (also accept f438w as alias)
      21 : F606W mag
      27 : F814W mag
      34 : RA  (J2000, epoch 2015.0)
      35 : DEC (J2000, epoch 2015.0)

    We read only (mag, ra, dec) for speed and robustness.
    """
    pref = (mag_pref or "f606w").strip().lower()
    filt2col1 = {
        "f275w": 3,
        "f336w": 9,
        "f435w": 15,
        "f438w": 15,  # alias
        "f606w": 21,
        "f814w": 27,
    }
    mag_col1 = filt2col1.get(pref, 21)  # default F606W

    # genfromtxt uses 0-based indices
    usecols = (mag_col1 - 1, 34 - 1, 35 - 1)  # (mag, ra, dec)

    arr = np.genfromtxt(
        path,
        comments="#",
        usecols=usecols,
        dtype=float,
        invalid_raise=False,
    )

    if arr.size == 0:
        raise ValueError(f"No numeric rows read from {path}. Is the file empty or format unexpected?")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    mag = arr[:, 0].astype(float)
    ra  = arr[:, 1].astype(float)
    dec = arr[:, 2].astype(float)

    # HUGS missing/sentinel values: e.g. -99.9999 for missing mags
    mag[mag <= -90] = np.nan
    ra[~np.isfinite(ra)] = np.nan
    dec[~np.isfinite(dec)] = np.nan

    # basic sanity
    m = np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra < 360) & (dec >= -90) & (dec <= 90)
    ra = ra[m]
    dec = dec[m]
    mag = mag[m]

    # Return same signature as before
    ra_col = f"fixed_col34(ra_epoch2015)"
    dec_col = f"fixed_col35(dec_epoch2015)"
    mag_col = f"fixed_col{mag_col1}({pref})"

    return ra, dec, mag, (ra_col, dec_col, mag_col), int(m.sum()), int(arr.shape[0])

def main():
    args = parse_args()

    ra_list, dec_list, mag_list = [], [], []
    info = []
    for p in args.in_txt:
        ra, dec, mag, cols, nk, nt = read_one_txt(p, args.mag_pref)
        ra_list.append(ra); dec_list.append(dec); mag_list.append(mag)
        info.append((p, cols, nk, nt))

    ra = np.concatenate(ra_list)
    dec = np.concatenate(dec_list)
    mag = np.concatenate(mag_list)

    print("[INFO] Loaded HST inputs (NO internal dedupe):")
    for p, cols, nk, nt in info:
        print(f"  - {p}: cols(ra,dec,mag)={cols} keep={nk}/{nt}")

    # magnitude -> app_sdss_g proxy
    app_sdss_g = mag.copy()
    if not np.isfinite(app_sdss_g).all():
        med = np.nanmedian(app_sdss_g[np.isfinite(app_sdss_g)]) if np.isfinite(app_sdss_g).any() else 20.0
        app_sdss_g[~np.isfinite(app_sdss_g)] = med

    # healpix grouping
    hpix = hp.ang2pix(args.nside, ra, dec, lonlat=True, nest=args.nest)
    uniq = np.unique(hpix)

    # fallback phys values (level0 spirit)
    Z_sun = 0.02
    feh = float(args.feh)
    z_local = np.clip(Z_sun * (10.0 ** feh), 1e-5, 0.1)

    with h5py.File(args.out_h5, "a") as f:
        root = f.require_group("star_catalog")
        n_added_total = 0
        n_dropped_total = 0

        for pid in uniq:
            m = (hpix == pid)
            if not np.any(m):
                continue

            ra_i = ra[m]
            dec_i = dec[m]
            g_i = app_sdss_g[m]

            # dedupe ONLY against existing HDF5
            if not args.no_dedupe_h5:
                keep = filter_against_existing_h5(f, pid, ra_i, dec_i, args.tol_arcsec)
                dropped = np.count_nonzero(~keep)
                n_dropped_total += dropped
                ra_i = ra_i[keep]; dec_i = dec_i[keep]; g_i = g_i[keep]

            if ra_i.size == 0:
                continue

            grp = root.require_group(str(int(pid)))
            n = ra_i.size

            ensure_dataset_append(grp, "RA", ra_i)
            ensure_dataset_append(grp, "DEC", dec_i)
            ensure_dataset_append(grp, "app_sdss_g", g_i)

            ensure_dataset_append(grp, "teff", np.full(n, float(args.teff_log10K)))
            ensure_dataset_append(grp, "grav", np.full(n, float(args.logg)))
            ensure_dataset_append(grp, "z_met", np.full(n, float(z_local)))

            ensure_dataset_append(grp, "AV", np.zeros(n))
            ensure_dataset_append(grp, "DM", np.zeros(n))
            ensure_dataset_append(grp, "mass", np.full(n, float(args.mass)))

            ensure_dataset_append(grp, "pmra", np.zeros(n))
            ensure_dataset_append(grp, "pmdec", np.zeros(n))
            ensure_dataset_append(grp, "RV", np.zeros(n))
            ensure_dataset_append(grp, "parallax", np.zeros(n))

            n_added_total += n

        print(f"[OK] Appended {n_added_total} HST rows into {args.out_h5}")
        if not args.no_dedupe_h5:
            print(f"[OK] Dropped {n_dropped_total} HST rows as duplicates of existing HDF5 within {args.tol_arcsec}\"")

if __name__ == "__main__":
    main()

