#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon数据结构调试脚本
=======================

分析coordinator中的数据结构以便正确测试模块功能
"""

import os
import sys
import numpy as np
from lammphonon import PhononCoordinator

# 设置数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def print_object_info(obj, name, max_depth=1, depth=0):
    """递归打印对象信息"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    print(f"{indent}{name}: {type(obj)}")
    
    if isinstance(obj, dict):
        print(f"{indent}  Keys: {list(obj.keys())}")
        for k, v in obj.items():
            print_object_info(v, f"{name}['{k}']", max_depth, depth+1)
    
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}  Length: {len(obj)}")
        if len(obj) > 0:
            print(f"{indent}  First element type: {type(obj[0])}")
            if isinstance(obj[0], (dict, list, tuple, np.ndarray)):
                print_object_info(obj[0], f"{name}[0]", max_depth, depth+1)
    
    elif isinstance(obj, np.ndarray):
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        if obj.size > 0:
            try:
                if obj.ndim == 1:
                    print(f"{indent}  First few elements: {obj[:min(3, obj.size)]}")
                else:
                    print(f"{indent}  First row: {obj[0]}")
            except:
                print(f"{indent}  [Error printing elements]")

def inspect_coordinator():
    """检查PhononCoordinator的数据结构"""
    print("初始化PhononCoordinator...")
    
    # 创建协调器
    coordinator = PhononCoordinator()
    
    try:
        # 读取轨迹文件
        traj_file = os.path.join(DATA_DIR, "test_dump.phonon")
        print(f"读取轨迹文件: {traj_file}")
        coordinator.read_trajectory(traj_file)
        
        # 读取能量文件
        energy_file = os.path.join(DATA_DIR, "test_energy.txt")
        print(f"读取能量文件: {energy_file}")
        coordinator.read_energy_data(energy_file)
        
        # 读取热流数据
        heatflux_file = os.path.join(DATA_DIR, "test_heatflux.dat")
        print(f"读取热流文件: {heatflux_file}")
        coordinator.read_heatflux_data(heatflux_file)
        
        # 读取极化数据
        polarization_file = os.path.join(DATA_DIR, "test_polarization.txt")
        print(f"读取极化文件: {polarization_file}")
        coordinator.read_polarization_data(polarization_file)
        
        # 检查数据结构
        print("\n===== 主要数据结构 =====")
        
        # 检查轨迹数据
        print("\n-- 轨迹数据 (trajectory_data) --")
        print_object_info(coordinator.trajectory_data, "trajectory_data", max_depth=2)
        
        # 检查能量数据
        print("\n-- 能量数据 (energy_data) --")
        print_object_info(coordinator.energy_data, "energy_data", max_depth=2)
        
        # 检查热流数据
        print("\n-- 热流数据 (heatflux_data) --")
        print_object_info(coordinator.heatflux_data, "heatflux_data", max_depth=2)
        
        # 检查极化数据
        print("\n-- 极化数据 (polarization_data) --")
        print_object_info(coordinator.polarization_data, "polarization_data", max_depth=2)
    
    except Exception as e:
        print(f"错误: {e}")
    
    return coordinator

if __name__ == "__main__":
    inspect_coordinator() 