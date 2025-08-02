#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon - 基础声子态密度分析示例

本示例展示如何使用LAMMPhonon计算声子态密度(DOS)

作者: 梁树铭 (Shuming Liang)
邮箱: lsm315@mail.ustc.edu.cn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lammphonon import PhononCoordinator, PhononAnalyzer

def main():
    """基础声子态密度分析示例"""
    print("LAMMPhonon基础声子态密度分析示例")
    
    # 创建输出目录
    output_dir = "./phonon_dos_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 创建协调器和分析器
    coordinator = PhononCoordinator()
    coordinator.set_config("output_dir", output_dir)
    
    # 读取轨迹文件
    # 注意: 请修改为您实际的轨迹文件路径
    trajectory_file = "../data/dump.phonon"
    print(f"读取轨迹文件: {trajectory_file}")
    coordinator.read_trajectory(trajectory_file)
    
    # 创建声子分析器
    phonon_analyzer = PhononAnalyzer(coordinator)
    
    # 计算速度自相关函数(VACF)
    print("计算速度自相关函数(VACF)...")
    velocities = np.array(coordinator.trajectory_data['velocities'])
    vacf = phonon_analyzer.calculate_velocity_autocorrelation(velocities)
    
    # 保存VACF数据
    vacf_file = os.path.join(output_dir, "vacf.txt")
    np.savetxt(vacf_file, vacf)
    print(f"VACF已保存到: {vacf_file}")
    
    # 计算声子态密度(DOS)
    print("计算声子态密度(DOS)...")
    freqs, dos = phonon_analyzer.calculate_dos(vacf)
    
    # 保存DOS数据
    dos_file = os.path.join(output_dir, "dos.txt")
    np.savetxt(dos_file, np.column_stack((freqs, dos)), 
              header="Frequency (THz), DOS")
    print(f"DOS已保存到: {dos_file}")
    
    # 绘制DOS图
    print("生成DOS图...")
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, dos)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Density of States')
    plt.title('Phonon Density of States')
    plt.grid(True)
    
    # 保存图像
    dos_plot_file = os.path.join(output_dir, "dos.png")
    plt.savefig(dos_plot_file, dpi=300)
    plt.close()
    print(f"DOS图像已保存到: {dos_plot_file}")
    
    print("分析完成!")

if __name__ == "__main__":
    main() 