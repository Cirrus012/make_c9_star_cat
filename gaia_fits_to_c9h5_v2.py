#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaia_fits_to_c9h5_v2.py

把 Gaia cone-search 的 FITS 表直接转换为 CSST Cycle9 星表 HDF5（按 healpix 分组），不经过 CSV。

与官方 Cycle9 星表的“真实数值形态”对齐（以官方 HDF5 文件实际数值为准）：
- teff：存储的是 log10(T/K)（而不是 K）
- z_met：存储的是金属质量分数 Z（线性量），由 Gaia 的 [M/H]（dex）换算：
        Z = Z_sun * 10^[M/H]   （脚本中仍沿用历史变量名 feh 表示 [M/H]）

level 开关（筛选强度从宽到严）：
- level 0：不做天测列筛选；缺失量允许插值或“兜底填值”（用于尽量不丢星，适合 PSF/畸变解算测试）
- level 1：仅要求 pmra/pmdec 有效；其余缺失允许插值或兜底（可能缺 BP/RP 或视差，物理参数质量较差）
- level 2：严格要求 parallax>0、pmra/pmdec、BP/RP 全有效；只允许 KNN 插值，不允许兜底；任何无法插值者丢弃

插值策略（在 (BP-RP, M_G) 的 HRD 空间做 KNN）：
1) [M/H]（dex）：优先 Gaia mh_gspphot；缺失则 KNN；仅 level 0/1 允许兜底
2) teff/logg：优先 Gaia teff_gspphot/logg_gspphot；缺失则 KNN；仅 level 0/1 允许兜底
3) mass：优先 Gaia mass_flame；其次用 (logg, radius) 估算；仍缺则 KNN；仅 level 0/1 允许兜底

输出字段（兼容 C9_Catalog 读取端）：
  RA, DEC, app_sdss_g, teff(log10K), grav(logg), z_met(Z), AV, DM, mass, pmra, pmdec, RV, parallax
注：AV/DM/RV 目前按“仅做像面/定标测试”的需求统一填 0。
"""


import argparse
import numpy as np
import h5py
import healpy as hp
from astropy.table import Table

# 尽量复用你现有实现（gaia_to_c9_csv_v2.py 的换算）
def compute_sdss_g_from_gaia(g_mag, bp_mag, rp_mag):
    """
    复用 gaia_to_c9_csv_v2.py 的经验换算：
    g_sdss ≈ G + a0 + a1*(BP-RP) + a2*(BP-RP)^2 + a3*(BP-RP)^3
    若 BP/RP 不可用，调用方决定是否回退到 G（level<=1 才允许）。
    """
    a0, a1, a2, a3 = 0.13518, -0.46245, -0.25171, 0.021349
    color = bp_mag - rp_mag
    return g_mag + a0 + a1*color + a2*color**2 + a3*color**3

def safe_masked_to_ndarray(col, fill=np.nan):
    if hasattr(col, "filled"):
        return np.array(col.filled(fill))
    return np.array(col)

def pick_col(tab, candidates, required=True, fill=np.nan):
    for name in candidates:
        if name in tab.colnames:
            return safe_masked_to_ndarray(tab[name], fill=fill)
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def abs_mag_from_parallax(m_app, parallax_mas):
    """
    M = m - DM, DM = 5*log10(d/10pc) = 5*log10(1000/pi_mas) - 5
    => M = m - (5*log10(1000/pi_mas) - 5) = m -10 + 5*log10(pi_mas)
    仅对 pi_mas > 0 有意义。
    """
    pi = parallax_mas
    M = np.full_like(m_app, np.nan, dtype=float)
    ok = np.isfinite(m_app) & np.isfinite(pi) & (pi > 0)
    M[ok] = m_app[ok] - 10.0 + 5.0 * np.log10(pi[ok])
    return M

# -------------------------
# KNN 插值（仅依赖 scipy）
# -------------------------
def build_knn_predictor(x1, x2, y, k=20):
    """
    在 (x1,x2) 空间对 y 做 KNN 距离加权插值。
    返回 predict(x1_new, x2_new) 或 None（训练样本不足）。
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    n = int(mask.sum())
    if n < max(10, min(k, 20)):
        return None

    X = np.vstack([x1[mask], x2[mask]]).T
    ytr = y[mask].astype(float)

    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    scale = np.where(mad > 0, mad, np.nanstd(X, axis=0))
    scale = np.where(scale > 0, scale, 1.0)

    Xs = (X - med) / scale
    tree = cKDTree(Xs)

    k_eff = min(k, len(ytr))

    def predict(x1_new, x2_new):
        x1_new = np.asarray(x1_new, dtype=float)
        x2_new = np.asarray(x2_new, dtype=float)
        Xn = np.vstack([x1_new, x2_new]).T
        Xns = (Xn - med) / scale

        # 兼容不同 scipy 版本：workers 参数不一定存在
        try:
            dist, idx = tree.query(Xns, k=k_eff, workers=-1)
        except TypeError:
            dist, idx = tree.query(Xns, k=k_eff)

        # k=1 时 dist/idx 可能是 1d
        if k_eff == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)

        w = 1.0 / (dist**2 + 1e-12)
        ypred = np.sum(w * ytr[idx], axis=1) / np.sum(w, axis=1)
        return ypred

    return predict

def fill_or_drop(y, predictor, x1, x2, allow_fallback, fallback_value, drop_mask):
    """
    Fill missing y using predictor(x1,x2) where query features are finite.
    If predictor cannot be applied:
      - allow_fallback=True  -> fill remaining missing with fallback_value
      - allow_fallback=False -> mark remaining missing for dropping
    """
    y = y.astype(float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    miss = ~np.isfinite(y)

    # Only predict where query features are finite
    if predictor is not None and np.any(miss):
        qmask = miss & np.isfinite(x1) & np.isfinite(x2)
        if np.any(qmask):
            y_pred = predictor(x1[qmask], x2[qmask])
            y[qmask] = y_pred

    miss2 = ~np.isfinite(y)
    if np.any(miss2):
        if allow_fallback:
            y[miss2] = fallback_value
        else:
            drop_mask |= miss2

    return y, drop_mask

# -------------------------
# HDF5 写入（继承现有写法）
# -------------------------
def ensure_dataset_append(group, name, data, dtype="f8"):
    if name not in group:
        group.create_dataset(name, data=data.astype(dtype), maxshape=(None,))
    else:
        ds = group[name]
        old = ds.shape[0]
        ds.resize((old + len(data),))
        ds[old:] = data.astype(dtype)

def write_h5_healpix(out_h5, hpix_ring, data_dict):
    uniq = np.unique(hpix_ring)
    
    # 修正 1: 使用 'a' (append) 模式，与 v1 (csv_to_c9h5.py) 保持一致
    # 这样既可以创建新文件，也可以向已有文件追加数据
    with h5py.File(out_h5, "a") as f:
        
        # 修正 2: 必须先进入或创建 'star_catalog' 组
        # v1 代码对应逻辑: root = h5f.require_group("star_catalog")
        root = f.require_group("star_catalog")

        for pid in uniq:
            m = (hpix_ring == pid)
            
            # 修正 3: 在 star_catalog 下创建/获取 HealpixID 组
            # v1 代码对应逻辑: grp = root.require_group(str(hid))
            grp = root.require_group(str(int(pid)))
            
            for key, arr in data_dict.items():
                # 修正 4: 使用脚本前面已定义的 ensure_dataset_append
                # 这样保证了数据类型 (dtype="f8") 和 存储结构 (maxshape) 与 v1 完全一致
                ensure_dataset_append(grp, key, arr[m])
                
    print(f"[OK] Wrote HDF5: {out_h5}  (groups={len(uniq)}) inside /star_catalog")
# -------------------------
# 主流程
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Gaia FITS -> CSST C9 healpix HDF5 (level 0/1/2)")
    p.add_argument("--in_fits", required=True, help="Input Gaia FITS table")
    p.add_argument("--out_h5", required=True, help="Output HDF5")
    p.add_argument("--nside", type=int, default=128, help="healpix NSIDE (default 128)")
    p.add_argument("--nest", action="store_true", help="use NEST indexing (default RING)")
    p.add_argument("--level", type=int, choices=[0, 1, 2], default=1,
                   help="0: keep all (impute/fallback), 1: require pm only (impute/fallback), 2: strict (parallax>0+pm+BP/RP; interpolate only)")
    p.add_argument("--k", type=int, default=20, help="KNN neighbors for interpolation (default 20)")
    return p.parse_args()

def main():
    args = parse_args()
    level = args.level
    allow_fallback = (level in (0, 1))  # level2 不允许“假值兜底”

    tab = Table.read(args.in_fits)
    n_in = len(tab)
    print(f"[INFO] Input rows: {n_in}")

    # ---------
    # 读入基础列（兼容不同下载脚本命名）
    # ---------
    ra  = pick_col(tab, ["ra", "RA"], required=True)
    dec = pick_col(tab, ["dec", "DEC"], required=True)

    parallax = pick_col(tab, ["parallax", "parallax_mas"], required=True)
    pmra     = pick_col(tab, ["pmra"], required=True)
    pmdec    = pick_col(tab, ["pmdec"], required=True)
    rv       = pick_col(tab, ["radial_velocity", "rv", "RV"], required=False)
    if rv is None:
        rv = np.full(n_in, np.nan)

    gmag  = pick_col(tab, ["phot_g_mean_mag", "gmag", "G"], required=True)
    bpmag = pick_col(tab, ["phot_bp_mean_mag", "bpmag", "BP"], required=False)
    rpmag = pick_col(tab, ["phot_rp_mean_mag", "rpmag", "RP"], required=False)

    teff_K = pick_col(tab, ["teff", "teff_gspphot"], required=False)
    logg   = pick_col(tab, ["logg", "logg_gspphot"], required=False)
    mh     = pick_col(tab, ["feh", "mh", "mh_gspphot"], required=False)

    # 新增（如果下载时 join 了 astrophysical_parameters）
    mass_flame   = pick_col(tab, ["mass_flame"], required=False)
    radius_flame = pick_col(tab, ["radius_flame"], required=False)
    radius_gsp   = pick_col(tab, ["radius_gspphot"], required=False)
    # ---------
    # level 过滤（只影响“保留哪些行”；不改变后续列的换算/插值逻辑）
    #
    # level 2：严格——要求 parallax>0、pmra/pmdec、BP/RP 全有效
    # level 1：宽松——仅要求 pmra/pmdec 有效（允许缺 BP/RP 或视差）
    # level 0：不筛选
    # ---------
    drop_ast = np.zeros(n_in, dtype=bool)

    if level == 2:
        # Level 2: 严格模式 (保持原样)
        need = (
            np.isfinite(parallax) & (parallax > 0) &
            np.isfinite(pmra) & np.isfinite(pmdec) &
            (bpmag is not None) & (rpmag is not None)
        )
        if bpmag is not None and rpmag is not None:
            need = need & np.isfinite(bpmag) & np.isfinite(rpmag)
        else:
            need = np.zeros(n_in, dtype=bool)
        drop_ast = ~need

    elif level == 1:
        # Level 1: 宽松模式 (只要求有自行)
        need = np.isfinite(pmra) & np.isfinite(pmdec)
        drop_ast = ~need
        
    # Level 0 保持不做任何筛选 (drop_ast 全为 False)

    n_after_ast = int((~drop_ast).sum())
    print(f"[INFO] After astrometry filter (level {level}): keep {n_after_ast}/{n_in} ({100*n_after_ast/n_in:.2f}%)")

    # 对 level0：允许缺失 parallax/pm/bp/rp 也先留下，但插值特征会缺
    keep = ~drop_ast

    # 先裁剪所有数组到 keep
    def cut(x):
        return x[keep]

    ra  = cut(ra);  dec = cut(dec)
    parallax = cut(parallax); pmra = cut(pmra); pmdec = cut(pmdec); rv = cut(rv)
    gmag = cut(gmag)
    bpmag = cut(bpmag) if bpmag is not None else None
    rpmag = cut(rpmag) if rpmag is not None else None
    teff_K = cut(teff_K) if teff_K is not None else None
    logg = cut(logg) if logg is not None else None
    mh = cut(mh) if mh is not None else None
    mass_flame = cut(mass_flame) if mass_flame is not None else None
    radius_flame = cut(radius_flame) if radius_flame is not None else None
    radius_gsp = cut(radius_gsp) if radius_gsp is not None else None

    n0 = len(ra)

    # ---------
    # 构造插值特征：color=BP-RP, 绝对星等 M_G（需要 parallax>0）
    # ---------
    color = np.full(n0, np.nan)
    if bpmag is not None and rpmag is not None:
        color = bpmag - rpmag
    MG = abs_mag_from_parallax(gmag, parallax)

    # 对 level0：允许 BP/RP 或 parallax 缺，MG/color 会 NaN，插值可能失败，后面用 fallback
    # 对 level1/2：这些已经被过滤为有效

    # ---------
    # app_sdss_g
    # level0/1：BP/RP 缺失允许退回 G（因为 level 1 允许缺颜色）
    # level2：BP/RP 必有
    # ---------
    app_sdss_g = np.full(n0, np.nan)
    if bpmag is not None and rpmag is not None:
        app_sdss_g = compute_sdss_g_from_gaia(gmag, bpmag, rpmag)
        
    if level <= 1:  # <--- 修改这里：将 level==0 改为 level<=1
        # 允许退回
        miss_g = ~np.isfinite(app_sdss_g)
        app_sdss_g[miss_g] = gmag[miss_g]

    # ---------
    # teff(log10K), logg, mh 先准备为 float 数组
    # ---------
    teff_log = np.full(n0, np.nan)
    if teff_K is not None:
        ok = np.isfinite(teff_K) & (teff_K > 0)
        teff_log[ok] = np.log10(teff_K[ok])

    grav = np.full(n0, np.nan)
    if logg is not None:
        grav = logg.astype(float)

    feh = np.full(n0, np.nan)
    if mh is not None:
        feh = mh.astype(float)

    # ---------
    # KNN predictor（在 HRD 空间插值）
    # ---------
    k = args.k
    pred_teff = build_knn_predictor(color, MG, teff_log, k=k)
    pred_logg = build_knn_predictor(color, MG, grav,     k=k)
    pred_feh  = build_knn_predictor(color, MG, feh,      k=k)

    drop_more = np.zeros(n0, dtype=bool)

    # fallback 值（与你原 pipeline 的精神一致，但更克制）
    teff_fb = np.nanmedian(teff_log[np.isfinite(teff_log)]) if np.isfinite(teff_log).any() else np.log10(5000.0)
    logg_fb = np.nanmedian(grav[np.isfinite(grav)])         if np.isfinite(grav).any()     else 4.5
    feh_fb  = np.nanmedian(feh[np.isfinite(feh)])           if np.isfinite(feh).any()      else -0.2

    teff_log, drop_more = fill_or_drop(teff_log, pred_teff, color, MG, allow_fallback, teff_fb, drop_more)
    grav,     drop_more = fill_or_drop(grav,     pred_logg, color, MG, allow_fallback, logg_fb, drop_more)
    feh,      drop_more = fill_or_drop(feh,      pred_feh,  color, MG, allow_fallback, feh_fb,  drop_more)

    # ---------
    # mass：优先 mass_flame；否则 (logg,radius) 物理估算；仍缺则 KNN(HRD) 插值
    # ---------
    mass = np.full(n0, np.nan)

    if mass_flame is not None:
        mass = mass_flame.astype(float)

    # 物理估算：M/Msun = 10^(logg - logg_sun) * (R/Rsun)^2, logg_sun≈4.438
    # radius 优先 flame，再 gsp
    radius = None
    if radius_flame is not None:
        radius = radius_flame.astype(float)
    elif radius_gsp is not None:
        radius = radius_gsp.astype(float)

    if radius is not None:
        ok = (~np.isfinite(mass)) & np.isfinite(grav) & np.isfinite(radius) & (radius > 0)
        mass[ok] = (10.0 ** (grav[ok] - 4.438)) * (radius[ok] ** 2)

    pred_mass = build_knn_predictor(color, MG, mass, k=k)
    mass_fb = np.nanmedian(mass[np.isfinite(mass)]) if np.isfinite(mass).any() else 1.0
    mass, drop_more = fill_or_drop(mass, pred_mass, color, MG, allow_fallback, mass_fb, drop_more)

    # 合理范围裁剪（避免极端插值把 SED 库搞崩）
    # 这不是“假值兜底”，而是数值稳定性保护
    teff_log = np.clip(teff_log, 3.2, 4.7)   # ~1600K - 50000K
    grav     = np.clip(grav,     0.0, 6.0)
    feh      = np.clip(feh,     -5.0, 1.0)
    mass     = np.clip(mass,     0.05, 50.0)

    # ---------
    # Z（z_met）：Z = Zsun * 10^[M/H]
    # ---------
    Z_sun = 0.02
    z_met = Z_sun * (10.0 ** feh)
    z_met = np.clip(z_met, 1e-5, 0.1)

    # ---------
    # AV, DM, RV：按你要求，全部填 0（不做）
    # ---------
    AV = np.zeros(n0, dtype=float)
    DM = np.zeros(n0, dtype=float)
    RV = np.zeros(n0, dtype=float)

    # ---------
    # level0/1：允许缺失量存在。为了输出字段数值稳定：
    #   - level 0：parallax/pm 可能缺失 -> 填 0
    #   - level 1：通常只可能缺 parallax（pm 已筛选为有效）-> 也填 0
    # level2：这些应当已保证有效；若仍有 NaN/非法则丢弃
    # ---------
    if level <= 1:  # <--- 修改这里：将 level==0 改为 level<=1
        for arr in (parallax, pmra, pmdec):
            bad = ~np.isfinite(arr)
            arr[bad] = 0.0
    else:
        # level2：若还有 NaN，视为无法接受 -> 丢弃
        bad_ast = (~np.isfinite(parallax)) | (parallax <= 0) | (~np.isfinite(pmra)) | (~np.isfinite(pmdec))
        drop_more |= bad_ast
    # app_sdss_g level2：若仍 NaN（理论不该），丢弃；level0/1 允许兜底（level0 已兜底）
    if level == 2:
        drop_more |= (~np.isfinite(app_sdss_g))

    # ---------
    # 应用 drop_more（主要发生在 level2：无法插值者丢弃）
    # ---------
    n_before_final = n0
    keep2 = ~drop_more
    n_final = int(keep2.sum())
    print(f"[INFO] Extra drop due to level rules: drop {n_before_final - n_final}/{n_before_final} ({100*(n_before_final-n_final)/max(1,n_before_final):.2f}%)")
    print(f"[INFO] Final keep: {n_final}/{n_in} ({100*n_final/max(1,n_in):.2f}%)")

    # 切到最终
    ra=ra[keep2]; dec=dec[keep2]
    parallax=parallax[keep2]; pmra=pmra[keep2]; pmdec=pmdec[keep2]
    app_sdss_g=app_sdss_g[keep2]
    teff_log=teff_log[keep2]; grav=grav[keep2]; z_met=z_met[keep2]
    mass=mass[keep2]
    AV=AV[keep2]; DM=DM[keep2]; RV=RV[keep2]

    # healpix 分组
    hpix = hp.ang2pix(args.nside, ra, dec, lonlat=True, nest=args.nest)

    data = {
        "RA": ra.astype(float),
        "DEC": dec.astype(float),
        "app_sdss_g": app_sdss_g.astype(float),
        "teff": teff_log.astype(float),
        "grav": grav.astype(float),
        "z_met": z_met.astype(float),
        "AV": AV.astype(float),
        "DM": DM.astype(float),
        "mass": mass.astype(float),
        "pmra": pmra.astype(float),
        "pmdec": pmdec.astype(float),
        "RV": RV.astype(float),
        "parallax": parallax.astype(float),
    }

    write_h5_healpix(args.out_h5, hpix, data)


if __name__ == "__main__":
    main()

