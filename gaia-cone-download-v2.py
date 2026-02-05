#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaia-cone-download-v2.py

- Cone search 下载 Gaia DR3（TAP），新增一些对后续插值/质控有帮助的列：
  * ruwe
  * phot_*_mean_flux_over_error (SNR proxy)
  * (可选) LEFT JOIN gaiadr3.astrophysical_parameters：mass_flame, radius_flame, lum_flame, radius_gspphot 等
- 下载后立刻：
  1) 以 check_fits.py 的格式打印每列有效值比例
  2) 保存一个 completeness 横条图 PNG

沿用你现有的：代理 + DNS 劫持逻辑（尽量不重写成功部分）。
"""

import argparse
import sys
import os
import socket
from urllib.parse import urlparse

# matplotlib 用于保存图（服务器无显示也可）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.utils.tap.core import TapPlus
    import socks  # pysocks
except ImportError:
    print("Error: Missing required libraries. Run: pip install pysocks astropy astroquery matplotlib numpy")
    sys.exit(1)


# ----------------------------
# 代理 + DNS 劫持（继承原实现）
# ----------------------------
_original_getaddrinfo = socket.getaddrinfo

def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """
    自定义 DNS 解析：遇到常见 Gaia TAP 域名，绕过系统 DNS。
    """
    dns_map = {
        'gea.esac.esa.int': '193.147.153.153',          # ESA
        'gaia.ari.uni-heidelberg.de': '129.206.5.6',    # ARI
        'dc.g-vo.org': '134.102.48.5'                   # GAVO
    }
    if host in dns_map:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (dns_map[host], port))]
    return _original_getaddrinfo(host, port, family, type, proto, flags)

def setup_proxy(proxy_url=None):
    """
    可选代理设置：
    - 优先使用命令行传入的 --proxy
    - 若未提供，则尝试读取环境变量（socks_proxy/all_proxy/http_proxy/https_proxy）
    """
    proxy_url = proxy_url or os.environ.get('socks_proxy') or os.environ.get('all_proxy') or \
        os.environ.get('http_proxy') or os.environ.get('https_proxy')
    if not proxy_url:
        return
    print(f"[INFO] Configuring proxy settings from: {proxy_url}")
    parsed = urlparse(proxy_url)

    proxy_type = socks.HTTP
    if 'socks5' in parsed.scheme:
        proxy_type = socks.SOCKS5
    elif 'socks4' in parsed.scheme:
        proxy_type = socks.SOCKS4

    port = parsed.port
    if not port:
        port = 1080 if 'socks' in parsed.scheme else 8080

    socks.set_default_proxy(proxy_type, parsed.hostname, port, rdns=True)
    socket.socket = socks.socksocket
    socket.getaddrinfo = patched_getaddrinfo


# ----------------------------
# 输入清洗（继承原实现）
# ----------------------------
def sanitize_input(value_str):
    if value_str is None:
        return None
    return str(value_str).replace('−', '-').replace('–', '-').replace('—', '-')

def parse_args():
    p = argparse.ArgumentParser(description="Query Gaia DR3 via TAP (cone search) + completeness plot")
    p.add_argument("--ra", required=True, help="Center RA (deg or sexagesimal)")
    p.add_argument("--dec", required=True, help="Center Dec (deg or sexagesimal)")
    p.add_argument("--radius", default=1.0, type=float, help="Radius (deg)")
    p.add_argument("--gmax", default=21.0, type=float, help="phot_g_mean_mag < gmax")
    p.add_argument("--out", default="gaia_cone.fits", help="Output FITS filename")
    p.add_argument("--site", default="esa", choices=["esa", "ari", "gavo"], help="TAP service site")
    p.add_argument("--no_ap_join", action="store_true", help="Disable LEFT JOIN astrophysical_parameters (faster, fewer columns)")
    p.add_argument("--plot", default=None, help="Output PNG for completeness plot (default: <out>.completeness.png)")
    p.add_argument("--proxy", default=None, help="Optional proxy URL (e.g. socks5://127.0.0.1:1080)")
    return p.parse_args()

def parse_coord(ra_str, dec_str):
    ra_str = sanitize_input(ra_str)
    dec_str = sanitize_input(dec_str)
    try:
        coord = SkyCoord(ra=float(ra_str) * u.deg, dec=float(dec_str) * u.deg, frame="icrs")
    except ValueError:
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
    return coord


# ----------------------------
# ADQL 构建（新增列 + 可选 JOIN）
# ----------------------------
def build_adql(ra, dec, radius, gmax, join_ap=True):
    """
    默认尝试 LEFT JOIN gaiadr3.astrophysical_parameters，拿到：
      mass_flame, radius_flame, lum_flame, radius_gspphot 等（存在则有值，不存在则 NULL）
    同时保留你原来的 teff/logg/feh/a0（优先使用 gaia_source 的 gspphot 结果）
    """
    if join_ap:
        return f"""
        SELECT
          gs.source_id,
          gs.ra, gs.dec,
          gs.parallax, gs.pmra, gs.pmdec,
          gs.radial_velocity,
          gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
          gs.phot_g_mean_flux_over_error,
          gs.phot_bp_mean_flux_over_error,
          gs.phot_rp_mean_flux_over_error,
          gs.ruwe,

          gs.teff_gspphot AS teff,
          gs.logg_gspphot AS logg,
          gs.mh_gspphot   AS feh,
          gs.ag_gspphot   AS a0,

          ap.mass_flame   AS mass_flame,
          ap.radius_flame AS radius_flame,
          ap.lum_flame    AS lum_flame,
          ap.radius_gspphot AS radius_gspphot

        FROM gaiadr3.gaia_source AS gs
        LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
          ON gs.source_id = ap.source_id
        WHERE 1 = CONTAINS(
          POINT('ICRS', gs.ra, gs.dec),
          CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        AND gs.phot_g_mean_mag < {gmax}
        """
    else:
        return f"""
        SELECT
          gs.source_id,
          gs.ra, gs.dec,
          gs.parallax, gs.pmra, gs.pmdec,
          gs.radial_velocity,
          gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
          gs.phot_g_mean_flux_over_error,
          gs.phot_bp_mean_flux_over_error,
          gs.phot_rp_mean_flux_over_error,
          gs.ruwe,
          gs.teff_gspphot AS teff,
          gs.logg_gspphot AS logg,
          gs.mh_gspphot   AS feh,
          gs.ag_gspphot   AS a0
        FROM gaiadr3.gaia_source AS gs
        WHERE 1 = CONTAINS(
          POINT('ICRS', gs.ra, gs.dec),
          CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        AND gs.phot_g_mean_mag < {gmax}
        """


# ----------------------------
# 有效值统计 + 作图（继承 check_fits.py 输出风格）
# ----------------------------
def _is_valid(col):
    """
    Return boolean mask of valid (non-masked, and for floats also finite) entries.
    Works for int MaskedColumn too (no .filled(np.nan) on int dtype).
    """
    mask = getattr(col, "mask", None)

    # MaskedColumn case
    if mask is not None:
        valid = ~np.asarray(mask)

        # additionally exclude non-finite values for float columns
        if np.issubdtype(col.dtype, np.floating):
            data = np.asarray(col.data)
            valid &= np.isfinite(data)

        # for int/bool, mask alone is enough
        return valid

    # Non-masked column
    arr = np.asarray(col)
    if np.issubdtype(arr.dtype, np.number):
        return np.isfinite(arr)
    else:
        return arr != ""

def column_completeness(table):
    total = len(table)
    rows = []
    for name in table.colnames:
        valid_mask = _is_valid(table[name])
        valid_count = int(np.sum(valid_mask))
        ratio = 100.0 * valid_count / total if total > 0 else 0.0
        rows.append((name, valid_count, total, ratio))
    return rows

def print_completeness(rows):
    print("\nColumn Name          | Completeness                   | Ratio")
    print("-" * 75)
    for name, valid, total, ratio in rows:
        bar_len = int(ratio / 100.0 * 30)
        bar = "█" * bar_len + "-" * (30 - bar_len)
        print(f"{name:<20} |{bar}| {ratio:6.2f}% ({valid}/{total})")
    print("-" * 75)

def plot_completeness(rows, out_png):
    names = [r[0] for r in rows]
    ratios = [r[3] for r in rows]

    # 纵向太长时图会很高，做个自适应
    h = max(4, 0.28 * len(names) + 1.5)
    plt.figure(figsize=(10, h))
    y = np.arange(len(names))
    plt.barh(y, ratios)
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Valid ratio (%)")
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    args = parse_args()
    setup_proxy(args.proxy)
    coord = parse_coord(args.ra, args.dec)

    tap_urls = {
        "esa": "https://gea.esac.esa.int/tap-server/tap",
        "ari": "https://gaia.ari.uni-heidelberg.de/tap",
        "gavo": "http://dc.g-vo.org/tap",
    }
    url = tap_urls[args.site]
    print(f"[INFO] TAP URL: {url}")
    print(f"[INFO] Center (deg): RA={coord.ra.deg:.8f}, Dec={coord.dec.deg:.8f}, R={args.radius} deg")
    print(f"[INFO] G < {args.gmax}")

    out_png = args.plot or (args.out + ".completeness.png")

    # 先尝试 JOIN 版本；若失败自动退回 no-join（更稳）
    join_ap = (not args.no_ap_join)
    adql = build_adql(coord.ra.deg, coord.dec.deg, args.radius, args.gmax, join_ap=join_ap)
    tap = TapPlus(url=url)

    try:
        job = tap.launch_job_async(adql)
        table = job.get_results()
    except Exception as e:
        if join_ap:
            print(f"[WARN] JOIN astrophysical_parameters failed, fallback to no-join.\n        Error: {e}")
            adql = build_adql(coord.ra.deg, coord.dec.deg, args.radius, args.gmax, join_ap=False)
            job = tap.launch_job_async(adql)
            table = job.get_results()
        else:
            raise

    print(f"[INFO] Rows: {len(table)}")
    table.write(args.out, format="fits", overwrite=True)
    print(f"[INFO] Saved: {args.out}")

    rows = column_completeness(table)
    print_completeness(rows)
    plot_completeness(rows, out_png)
    print(f"[INFO] Completeness plot: {out_png}")


if __name__ == "__main__":
    main()
