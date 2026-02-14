#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_constant_psfcube.py

功能：
1. 读取指定的输入 psfCube HDF5 文件。
2. 提取指定的某一个 psfMat (source_band, source_id) 作为“模板”。
3. 创建一个新的 HDF5 文件（复制原文件结构和元数据）。
4. 将新文件中所有的 psfMat 替换为上述“模板”数据。

依赖：h5py, numpy, shutil
"""

import argparse
import h5py
import numpy as np
import shutil
import os
import sys

def get_band_path(band_name):
    """处理波段名称前缀，如 'g' -> 'w_g'"""
    if not band_name.startswith("w_"):
        return f"w_{band_name}"
    return band_name

def main():
    parser = argparse.ArgumentParser(description="生成一个全场 PSF 恒定的 psfCube")
    parser.add_argument("--input", required=True, help="原始 psfCube HDF5 路径")
    parser.add_argument("--output", required=True, help="输出的新 HDF5 路径")
    parser.add_argument("--band", required=True, help="模板 PSF 所在的波段 (如 g, i, w_g)")
    parser.add_argument("--psf-id", required=True, type=int, help="模板 PSF 的 ID (如 0, 98)")
    
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    
    # 0. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"[Error] 输入文件不存在: {input_path}")
        sys.exit(1)

    # 1. 从源文件中读取“模板 PSF”
    target_band = get_band_path(args.band)
    template_path = f"/{target_band}/psf_{args.psf_id}/psfMat"

    print(f"[Info] 正在读取模板 PSF: {input_path} -> {template_path}")
    
    try:
        with h5py.File(input_path, 'r') as f_in:
            if template_path not in f_in:
                print(f"[Error] 找不到路径: {template_path}")
                print("可用波段示例:", list(f_in.keys()))
                sys.exit(1)
            
            # 读取数据到内存 (float64)
            template_psf = f_in[template_path][...].astype(np.float64)
            template_shape = template_psf.shape
            print(f"[Info] 模板 PSF 读取成功，尺寸: {template_shape}, 总通量: {np.sum(template_psf):.4f}")
    except Exception as e:
        print(f"[Error] 读取 HDF5 失败: {e}")
        sys.exit(1)

    # 2. 复制文件到输出路径 (保留所有元数据、坐标信息等)
    print(f"[Info] 正在复制文件结构到: {output_path} ...")
    shutil.copy2(input_path, output_path)

    # 3. 遍历新文件并覆写所有的 psfMat
    print("[Info] 正在覆写所有 PSF 数据...")
    
    count = 0
    
    def overwrite_visitor(name, node):
        nonlocal count
        # 仅处理名为 "psfMat" 的 Dataset
        if isinstance(node, h5py.Dataset) and name.endswith("psfMat"):
            # 检查尺寸是否匹配，避免报错
            if node.shape != template_shape:
                print(f"[Warning] 跳过 {name}: 尺寸不匹配 (原: {node.shape} vs 模: {template_shape})")
                return
            
            # 覆写数据
            node[...] = template_psf
            count += 1

    try:
        with h5py.File(output_path, 'r+') as f_out:
            f_out.visititems(overwrite_visitor)
            
        print("-" * 40)
        print(f"[Success] 处理完成！")
        print(f"[Info] 共替换了 {count} 个 psfMat 数据集。")
        print(f"[Info] 输出文件: {output_path}")
        
    except Exception as e:
        print(f"[Error] 写入 HDF5 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
