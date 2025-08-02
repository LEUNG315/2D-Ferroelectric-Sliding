#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon简单测试脚本
=====================

测试LAMMPhonon的基本功能。

作者: 梁树铭 (Shuming Liang)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入LAMMPhonon模块
from lammphonon import PhononCoordinator, PhononAnalyzer, SlidingAnalyzer

# 设置数据和输出目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def test_basic_features():
    """测试基本功能"""
    print("测试LAMMPhonon基本功能...")
    
    # 创建协调器
    print("1. 创建PhononCoordinator...")
    coordinator = PhononCoordinator()
    coordinator.output_dir = OUTPUT_DIR
    
    # 读取轨迹文件
    traj_file = os.path.join(DATA_DIR, "test_dump.phonon")
    print(f"2. 读取轨迹文件: {traj_file}")
    coordinator.read_trajectory(traj_file)
    
    # 读取能量文件
    energy_file = os.path.join(DATA_DIR, "test_energy.txt")
    print(f"3. 读取能量文件: {energy_file}")
    coordinator.read_energy_data(energy_file)
    
    # 读取热流数据
    heatflux_file = os.path.join(DATA_DIR, "test_heatflux.dat")
    print(f"4. 读取热流文件: {heatflux_file}")
    coordinator.read_heatflux_data(heatflux_file)
    
    # 读取极化数据
    polarization_file = os.path.join(DATA_DIR, "test_polarization.txt")
    print(f"5. 读取极化文件: {polarization_file}")
    coordinator.read_polarization_data(polarization_file)
    
    # 创建声子分析器
    print("6. 创建PhononAnalyzer...")
    phonon_analyzer = PhononAnalyzer(coordinator)
    
    # 计算速度自相关函数
    print("7. 计算速度自相关函数...")
    velocities = np.array(coordinator.trajectory_data['velocities'])
    vacf = phonon_analyzer.calculate_velocity_autocorrelation(velocities)
    
    # 计算声子态密度
    print("8. 计算声子态密度...")
    freqs, dos = phonon_analyzer.calculate_dos(vacf)
    
    # 保存DOS结果
    print("9. 保存DOS计算结果...")
    dos_file = os.path.join(OUTPUT_DIR, "dos.txt")
    np.savetxt(dos_file, np.column_stack((freqs, dos)), 
               header="Frequency(THz) DOS", comments='# ')
    
    # 绘制DOS图像
    print("10. 绘制DOS图像...")
    plt.figure(figsize=(8, 6))
    plt.plot(freqs, dos)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS')
    plt.title('Phonon Density of States')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "dos.png"))
    plt.close()
    
    # 创建滑移分析器
    print("11. 创建SlidingAnalyzer...")
    sliding_analyzer = SlidingAnalyzer(coordinator)
    
    # 检测材料层
    print("12. 检测材料层...")
    positions = np.array(coordinator.trajectory_data['positions'][0])
    sliding_analyzer.detect_material_layers(positions)
    
    # 计算滑移距离
    print("13. 计算滑移距离...")
    all_positions = np.array(coordinator.trajectory_data['positions'])
    sliding_distance = sliding_analyzer.calculate_sliding_distance(all_positions)
    
    # 计算层间距离
    print("14. 计算层间距离...")
    interlayer_distance = sliding_analyzer.calculate_interlayer_distance(all_positions)
    
    # 绘制滑移距离图像
    print("15. 绘制滑移距离图像...")
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(sliding_distance)), sliding_distance)
    plt.xlabel('Frame')
    plt.ylabel('Sliding Distance (Å)')
    plt.title('Sliding Distance Evolution')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "sliding_distance.png"))
    plt.close()
    
    print("基本功能测试完成。")
    print(f"结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_basic_features() 