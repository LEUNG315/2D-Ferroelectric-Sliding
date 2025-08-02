#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon - 基础滑移分析示例

本示例展示如何使用LAMMPhonon分析材料滑移过程

作者: 梁树铭 (Shuming Liang)
邮箱: lsm315@mail.ustc.edu.cn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lammphonon import PhononCoordinator, SlidingAnalyzer

def main():
    """基础滑移分析示例"""
    print("LAMMPhonon基础滑移分析示例")
    
    # 创建输出目录
    output_dir = "./sliding_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 创建协调器
    coordinator = PhononCoordinator()
    coordinator.set_config("output_dir", output_dir)
    
    # 读取轨迹文件
    # 注意: 请修改为您实际的轨迹文件路径
    trajectory_file = "../data/dump.phonon"
    print(f"读取轨迹文件: {trajectory_file}")
    coordinator.read_trajectory(trajectory_file)
    
    # 读取力数据（如果有单独的力文件）
    # force_file = "../data/force.dat"
    # coordinator.read_force_data(force_file)
    
    # 创建滑移分析器
    sliding_analyzer = SlidingAnalyzer(coordinator)
    
    # 检测材料层
    print("检测材料层...")
    sliding_analyzer.detect_layers()
    
    # 计算滑移距离
    print("计算滑移距离...")
    distances = sliding_analyzer.calculate_sliding_distance()
    
    # 生成时间数组
    n_frames = len(distances)
    timestep = coordinator.timestep  # 默认为0.001 ps
    times = np.arange(n_frames) * timestep
    
    # 保存滑移距离数据
    dist_file = os.path.join(output_dir, "sliding_distance.txt")
    np.savetxt(dist_file, np.column_stack((times, distances)), 
              header="Time (ps), Sliding Distance (Å)")
    print(f"滑移距离已保存到: {dist_file}")
    
    # 绘制滑移距离图
    print("生成滑移距离图...")
    plt.figure(figsize=(10, 6))
    plt.plot(times, distances)
    plt.xlabel('Time (ps)')
    plt.ylabel('Sliding Distance (Å)')
    plt.title('Layer Sliding Distance vs. Time')
    plt.grid(True)
    
    # 保存图像
    dist_plot_file = os.path.join(output_dir, "sliding_distance.png")
    plt.savefig(dist_plot_file, dpi=300)
    plt.close()
    print(f"滑移距离图像已保存到: {dist_plot_file}")
    
    # 计算摩擦力
    print("计算摩擦力...")
    try:
        forces = sliding_analyzer.calculate_friction_force()
        
        # 保存摩擦力数据
        force_file = os.path.join(output_dir, "friction_force.txt")
        np.savetxt(force_file, np.column_stack((times, forces)), 
                  header="Time (ps), Friction Force (eV/Å)")
        print(f"摩擦力已保存到: {force_file}")
        
        # 绘制摩擦力图
        print("生成摩擦力图...")
        plt.figure(figsize=(10, 6))
        plt.plot(times, forces)
        plt.xlabel('Time (ps)')
        plt.ylabel('Friction Force (eV/Å)')
        plt.title('Friction Force vs. Time')
        plt.grid(True)
        
        # 保存图像
        force_plot_file = os.path.join(output_dir, "friction_force.png")
        plt.savefig(force_plot_file, dpi=300)
        plt.close()
        print(f"摩擦力图像已保存到: {force_plot_file}")
        
        # 计算摩擦系数
        print("计算摩擦系数...")
        # 注意：此处仅为示例，实际使用时需要提供正确的法向力
        normal_force = 1.0  # eV/Å
        friction_coef = sliding_analyzer.calculate_friction_coefficient(normal_force)
        print(f"摩擦系数: {friction_coef:.4f}")
        
        # 保存摩擦系数
        coef_file = os.path.join(output_dir, "friction_coefficient.txt")
        with open(coef_file, 'w') as f:
            f.write(f"Friction Coefficient: {friction_coef:.6f}\n")
        print(f"摩擦系数已保存到: {coef_file}")
        
    except Exception as e:
        print(f"计算摩擦力时出错: {str(e)}")
        print("可能需要轨迹文件中包含力数据，或提供单独的力文件")
    
    # 计算层间距离
    print("计算层间距离...")
    try:
        interlayer_distances = sliding_analyzer.calculate_interlayer_distance()
        
        # 保存层间距离数据
        int_dist_file = os.path.join(output_dir, "interlayer_distance.txt")
        np.savetxt(int_dist_file, np.column_stack((times, interlayer_distances)), 
                  header="Time (ps), Interlayer Distance (Å)")
        print(f"层间距离已保存到: {int_dist_file}")
        
        # 绘制层间距离图
        print("生成层间距离图...")
        plt.figure(figsize=(10, 6))
        plt.plot(times, interlayer_distances)
        plt.xlabel('Time (ps)')
        plt.ylabel('Interlayer Distance (Å)')
        plt.title('Interlayer Distance vs. Time')
        plt.grid(True)
        
        # 保存图像
        int_dist_plot_file = os.path.join(output_dir, "interlayer_distance.png")
        plt.savefig(int_dist_plot_file, dpi=300)
        plt.close()
        print(f"层间距离图像已保存到: {int_dist_plot_file}")
    except Exception as e:
        print(f"计算层间距离时出错: {str(e)}")
    
    print("分析完成!")

if __name__ == "__main__":
    main() 