#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon新增模块简单测试脚本
==========================

测试LAMMPhonon的新增分析模块：
1. ThermalAnalyzer - 热分析模块
2. AnharmonicAnalyzer - 非谐性分析模块
3. TemporalAnalyzer - 时间分析模块
4. EquilibrationAnalyzer - 平衡分析模块

作者: 梁树铭 (Shuming Liang)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入LAMMPhonon模块
from lammphonon import (
    PhononCoordinator, 
    PhononAnalyzer, 
    ThermalAnalyzer,
    AnharmonicAnalyzer,
    TemporalAnalyzer,
    EquilibrationAnalyzer
)

# 设置数据和输出目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/new_modules_simple")

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def print_section(title):
    """打印带分隔符的小节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def setup_coordinator():
    """初始化PhononCoordinator并加载测试数据"""
    print("初始化PhononCoordinator...")
    
    # 创建协调器
    coordinator = PhononCoordinator()
    coordinator.output_dir = OUTPUT_DIR
    
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
    
    return coordinator

def test_thermal_analyzer(coordinator):
    """测试ThermalAnalyzer热分析模块"""
    print_section("测试ThermalAnalyzer热分析模块")
    
    # 创建热分析器
    analyzer = ThermalAnalyzer(coordinator)
    
    # 测试热流自相关函数计算
    print("1. 计算热流自相关函数...")
    try:
        # 使用原始数据，而不是通过切片获取
        heatflux_data = np.array(coordinator.heatflux_data)[:, 1:4]
        
        # 计算热流自相关函数
        times, hfacf = analyzer.calculate_heatflux_autocorrelation(
            heatflux_data
        )
        
        # 绘制热流自相关函数
        plt.figure(figsize=(8, 6))
        plt.plot(times, hfacf)
        plt.xlabel('Time Lag (ps)')
        plt.ylabel('HFACF')
        plt.title('Heat Flux Autocorrelation Function')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "hfacf.png"))
        plt.close()
        print("  热流自相关函数计算成功，结果已保存")
    except Exception as e:
        print(f"  热流自相关函数计算出错: {e}")
    
    # 测试热导率计算
    print("2. 计算热导率...")
    try:
        # 设置体积
        box_size = [49.2, 42.6, 10.0]  # 从测试文件中获取
        volume = box_size[0] * box_size[1] * box_size[2]
        
        # 计算热导率
        thermal_conductivity, kappa_components = analyzer.calculate_thermal_conductivity(
            heatflux_data,
            temperature=300,  # 假设温度为300K
            volume=volume
        )
        
        print(f"  估计的热导率: {thermal_conductivity[-1]} W/mK")
        
        # 绘制热导率随时间的演化
        plt.figure(figsize=(8, 6))
        plt.plot(times, thermal_conductivity)
        plt.xlabel('Time (ps)')
        plt.ylabel('Thermal Conductivity (W/mK)')
        plt.title('Thermal Conductivity Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "thermal_conductivity.png"))
        plt.close()
    except Exception as e:
        print(f"  热导率计算出错: {e}")
    
    # 测试极化-热流相关性分析
    print("3. 分析极化-热流相关性...")
    try:
        # 使用原始数据
        polarization_data = np.array(coordinator.polarization_data)[:, 1:4]
        
        correlation = analyzer.analyze_polarization_heatflux_correlation(
            polarization_data,
            heatflux_data
        )
        
        print(f"  极化-热流相关系数: {correlation}")
    except Exception as e:
        print(f"  极化-热流相关性分析出错: {e}")

def test_anharmonic_analyzer(coordinator):
    """测试AnharmonicAnalyzer非谐性分析模块"""
    print_section("测试AnharmonicAnalyzer非谐性分析模块")
    
    # 创建非谐性分析器
    analyzer = AnharmonicAnalyzer(coordinator)
    
    # 计算玻色-爱因斯坦分布
    print("1. 计算玻色-爱因斯坦分布...")
    try:
        # 创建频率数组
        frequencies = np.linspace(0, 20, 100)  # 0-20 THz, 100点
        
        # 计算玻色-爱因斯坦分布
        be_occupations = analyzer.calculate_bose_einstein_distribution(
            frequencies,
            temperature=300  # 假设温度为300K
        )
        
        # 绘制玻色-爱因斯坦分布
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies, be_occupations)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Occupation Number')
        plt.title('Bose-Einstein Distribution (T=300K)')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "bose_einstein.png"))
        plt.close()
        
        print(f"  玻色-爱因斯坦分布计算成功，结果已保存")
    except Exception as e:
        print(f"  玻色-爱因斯坦分布计算出错: {e}")

def test_temporal_analyzer(coordinator):
    """测试TemporalAnalyzer时间分析模块"""
    print_section("测试TemporalAnalyzer时间分析模块")
    
    # 创建时间分析器
    analyzer = TemporalAnalyzer(coordinator)
    
    # 测试时间分辨DOS计算
    print("1. 计算时间分辨DOS...")
    try:
        # 转换数据类型
        velocities = np.array(coordinator.trajectory_data['velocities'])
        
        # 设置窗口大小
        window_size = 1  # 只有两帧，所以窗口大小设为1
        
        # 计算时间分辨DOS
        frequencies, dos_evolution, time_points = analyzer.calculate_time_resolved_dos(
            velocities,
            window_size=window_size
        )
        
        if frequencies is not None and dos_evolution is not None:
            # 绘制第一个时间点的DOS
            plt.figure(figsize=(8, 6))
            plt.plot(frequencies, dos_evolution[0])
            plt.xlabel('Frequency (THz)')
            plt.ylabel('DOS')
            plt.title(f'Phonon DOS at t={time_points[0]:.3f} ps')
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "time_resolved_dos.png"))
            plt.close()
            
            print(f"  时间分辨DOS计算成功，形状: {dos_evolution.shape}")
        else:
            print("  时间分辨DOS计算结果为空")
    except Exception as e:
        print(f"  计算时间分辨DOS时出错: {e}")

def test_equilibration_analyzer(coordinator):
    """测试EquilibrationAnalyzer平衡分析模块"""
    print_section("测试EquilibrationAnalyzer平衡分析模块")
    
    # 创建平衡分析器
    analyzer = EquilibrationAnalyzer(coordinator)
    
    # 测试系统平衡过程分析
    print("1. 分析系统平衡过程...")
    try:
        # 使用能量数据
        energy_data = np.array(coordinator.energy_data)[:, 1]  # 使用总能量列
        
        # 创建时间点数组
        time_points = np.array(coordinator.energy_data)[:, 0] * 0.001  # 假设时间单位是ps
        
        # 计算系统平衡时间
        equilibration_time, params = analyzer.calculate_system_equilibration(
            energy_data,
            time_points
        )
        
        if equilibration_time is not None:
            print(f"  估计的能量平衡时间: {equilibration_time:.3f} ps")
            print(f"  平衡拟合参数: A={params[0]:.3f}, tau={params[1]:.3f}, E_inf={params[2]:.3f}")
            
            # 绘制拟合曲线
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, energy_data, 'o', label='Energy Data')
            
            # 定义指数衰减函数
            def exp_decay(t, A, tau, E_inf):
                return A * np.exp(-t / tau) + E_inf
            
            # 计算拟合曲线
            fit_curve = exp_decay(time_points, *params)
            plt.plot(time_points, fit_curve, 'r-', label='Fitted Curve')
            
            plt.axvline(x=equilibration_time, color='g', linestyle='--', 
                        label=f'Equilibration Time: {equilibration_time:.3f} ps')
            
            plt.xlabel('Time (ps)')
            plt.ylabel('Total Energy (eV)')
            plt.title('System Energy Equilibration')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "energy_equilibration.png"))
            plt.close()
        else:
            print("  系统平衡时间计算失败")
    except Exception as e:
        print(f"  分析系统平衡过程时出错: {e}")

def run_all_tests():
    """运行所有测试"""
    # 初始化PhononCoordinator
    coordinator = setup_coordinator()
    
    # 测试ThermalAnalyzer
    test_thermal_analyzer(coordinator)
    
    # 测试AnharmonicAnalyzer
    test_anharmonic_analyzer(coordinator)
    
    # 测试TemporalAnalyzer
    test_temporal_analyzer(coordinator)
    
    # 测试EquilibrationAnalyzer
    test_equilibration_analyzer(coordinator)
    
    print("\n所有测试完成。")
    print(f"结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_all_tests() 