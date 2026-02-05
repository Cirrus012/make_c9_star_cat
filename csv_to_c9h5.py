#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV -> C9-compatible HDF5 star catalog writer
- Groups by HEALPix RING pixels at NSIDE=128
- HDF5 layout: /star_catalog/<HID>/<dataset>
- All datasets are float64 1-D arrays
- Appends to existing files/groups/datasets if present

Usage:
  python csv_to_c9h5.py --csv stars.csv \
                        --out /path/to/C9_RA246p50_DECp35_custom_Nside_128_healpix.hdf5 \
                        --nside 128 --teff-is-linear  # 若 CSV 中 teff 是线性 K

Requirements: numpy, h5py, healpy
"""
import argparse, csv, math, os, sys
from collections import defaultdict
import numpy as np
import h5py as h5
import healpy as hp

FIELDS_REQUIRED = [
    "RA", "DEC",               # deg (J2000)
    "app_sdss_g",              # AB mag, used for normalization and selection
    "teff", "grav", "z_met",   # teff=log10(K) unless --teff-is-linear; grav=logg[cgs]; z_met=[Fe/H] dex
    "AV", "DM",                # mag; distance modulus
    "mass",                    # Msun
]
FIELDS_OPTIONAL = [
    "pmra", "pmdec",           # mas/yr
    "RV",                      # km/s
    "parallax",                # mas
    # "logage",                # not used by pipeline; you can include if desired (will be written if present)
]

ALL_FIELDS = FIELDS_REQUIRED + FIELDS_OPTIONAL

def ang2hid_ring_nside128(ra_deg: float, dec_deg: float, nside: int = 128) -> int:
    """Compute HEALPix ring pixel id for (RA,DEC) in degrees."""
    theta = np.deg2rad(90.0 - dec_deg)
    phi   = np.deg2rad(ra_deg % 360.0)
    return int(hp.ang2pix(nside, theta, phi, nest=False))

def ensure_dataset_append(grp: h5.Group, name: str, arr_new: np.ndarray):
    """Create or append to a 1-D float64 dataset."""
    if name in grp:
        ds = grp[name]
        old = ds.shape[0]
        ds.resize((old + arr_new.shape[0],))
        ds[old:] = arr_new
    else:
        # chunked + resizable for future appends
        grp.create_dataset(
            name,
            data=arr_new.astype("f8", copy=False),
            maxshape=(None,),
            chunks=True,
            compression=None,  # 可按需改为 "gzip"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV file (UTF-8).")
    ap.add_argument("--out", required=True, help="Output HDF5 file path.")
    ap.add_argument("--nside", type=int, default=128, help="HEALPix NSIDE (C9 uses 128).")
    ap.add_argument("--teff-is-linear", action="store_true", help="If set, convert Teff[K] to log10(K).")
    ap.add_argument("--allow-missing-optional", action="store_true", help="Missing optional fields default to 0.")
    args = ap.parse_args()

    # 读 CSV
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in reader.fieldnames or []]

        # 基本字段校验
        for k in FIELDS_REQUIRED:
            if k not in header:
                sys.exit(f"[ERROR] Required column missing: {k}")

        # 允许的额外列（比如 logage），只要你在 ALL_FIELDS 中或你愿意忽略
        extra_cols = [h for h in header if h not in ALL_FIELDS]
        if extra_cols:
            print(f"[WARN] Extra columns found and will be ignored in HDF5: {extra_cols}")

        # 预累积：hid -> {field -> list}
        by_hid = defaultdict(lambda: defaultdict(list))
        n_rows = 0

        for row in reader:
            n_rows += 1
            try:
                ra   = float(row["RA"])
                dec  = float(row["DEC"])
                hid  = ang2hid_ring_nside128(ra, dec, nside=args.nside)

                rec = {}
                # 必需字段
                rec["RA"]          = float(row["RA"])
                rec["DEC"]         = float(row["DEC"])
                rec["app_sdss_g"]  = float(row["app_sdss_g"])
                # teff: 默认期望 log10(K)
                teff_val = float(row["teff"])
                if args.teff_is_linear:
                    if teff_val <= 0:
                        raise ValueError("teff must be >0 when --teff-is-linear is set.")
                    teff_val = math.log10(teff_val)
                rec["teff"] = teff_val

                rec["grav"]        = float(row["grav"])
                rec["z_met"]       = float(row["z_met"])
                rec["AV"]          = float(row["AV"])
                rec["DM"]          = float(row["DM"])
                rec["mass"]        = float(row["mass"])

                # 可选字段
                for k in FIELDS_OPTIONAL:
                    if k in row and row[k] != "" and row[k] is not None:
                        rec[k] = float(row[k])
                    else:
                        if args.allow_missing_optional:
                            rec[k] = 0.0
                        else:
                            # 缺失可选列时，不写入该字段（HDF5 中将不存在对应 dataset）
                            pass

                # 累加到分桶
                for k, v in rec.items():
                    by_hid[hid][k].append(v)

            except Exception as e:
                print(f"[WARN] Skip row {n_rows} due to parse error: {e}")

    if not by_hid:
        sys.exit("[ERROR] No valid rows parsed. Abort.")

    # 写 HDF5（分像素组）
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with h5.File(args.out, "a") as h5f:
        root = h5f.require_group("star_catalog")
        total_written = 0
        n_groups = 0

        for hid, fields in by_hid.items():
            grp = root.require_group(str(hid))
            n_groups += 1

            # 将该像素组内的列表转成 array 并 append 到数据集
            # 固定写入顺序（可重复执行、可追加）
            keys_to_write = list(fields.keys())
            for k in keys_to_write:
                arr = np.asarray(fields[k], dtype="f8")
                ensure_dataset_append(grp, k, arr)
                total_written += arr.shape[0]

        print(f"[OK] Wrote {total_written} rows into {n_groups} HEALPix groups in {args.out}")

if __name__ == "__main__":
    main()

