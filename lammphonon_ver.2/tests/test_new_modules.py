#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon新增模块测试脚本
======================

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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/new_modules")

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
        heatflux_data = np.array(coordinator.heatflux_data[:, 1:4])
        hfacf = analyzer.calculate_heatflux_autocorrelation(
            heatflux_data,
            max_correlation_time=len(heatflux_data)-1
        )
        
        # 绘制热流自相关函数
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(hfacf)), hfacf)
        plt.xlabel('Time Lag')
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
        box_size = [49.2, 42.6, 10.0]  # 从测试文件中获取
        volume = box_size[0] * box_size[1] * box_size[2]
        
        thermal_conductivity = analyzer.calculate_thermal_conductivity(
            heatflux_data,
            temperature=300,  # 假设温度为300K
            volume=volume
        )
        print(f"  估计的热导率: {thermal_conductivity} W/mK")
    except Exception as e:
        print(f"  热导率计算出错: {e}")
    
    # 测试极化-热流相关性分析
    print("3. 分析极化-热流相关性...")
    try:
        polarization_data = np.array(coordinator.polarization_data[:, 1:4])
        
        correlation = analyzer.analyze_polarization_heatflux_correlation(
            polarization_data,
            heatflux_data
        )
        
        print(f"  极化-热流相关系数: {correlation}")
    except Exception as e:
        print(f"  极化-热流相关性分析出错: {e}")
    
    # 测试层间热传导分析
    print("4. 分析层间热传导...")
    try:
        # 为测试目的创建假的温度梯度数据
        temp_gradient = np.array([0.1, 0.15, 0.2])
        
        thermal_boundary_conductance = analyzer.calculate_thermal_boundary_conductance(
            temp_gradient,
            interlayer_distance=3.0,  # 假设层间距为3.0埃
            heat_flux=10.0  # 假设热流为10.0 W/m^2
        )
        
        print(f"  层间热边界导率: {thermal_boundary_conductance} W/(m^2·K)")
    except Exception as e:
        print(f"  层间热传导分析出错: {e}")

def test_anharmonic_analyzer(coordinator):
    """测试AnharmonicAnalyzer非谐性分析模块"""
    print_section("测试AnharmonicAnalyzer非谐性分析模块")
    
    # 创建非谐性分析器
    analyzer = AnharmonicAnalyzer(coordinator)
    
    # 获取声子分析器计算简正模式
    phonon_analyzer = PhononAnalyzer(coordinator)
    
    # 计算简正模式
    print("1. 计算简正模式...")
    try:
        positions = np.array(coordinator.trajectory_data['positions'][0])
        masses = np.ones(positions.shape[0])  # 假设所有原子质量相同
        
        eigenvals, eigenvectors = phonon_analyzer.calculate_normal_modes(
            positions,
            masses
        )
        print(f"  模式频率计算成功，获得 {len(eigenvals)} 个特征值")
        
        # 计算频率
        freq_thz = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
        
        # 保存前几个模式频率
        mode_file = os.path.join(OUTPUT_DIR, "modes.txt")
        with open(mode_file, 'w') as f:
            f.write("# Mode Frequency(THz)\n")
            for i, freq in enumerate(freq_thz):
                if i < 10:  # 只保存前10个模式
                    f.write(f"{i} {freq}\n")
    except Exception as e:
        print(f"  简正模式计算出错: {e}")
        return
    
    # 测试模式投影
    print("2. 计算模式投影...")
    try:
        # 选择一个时间帧
        frame_idx = 0
        
        # 获取位置和速度数据
        pos_frame = np.array(coordinator.trajectory_data['positions'][frame_idx])
        vel_frame = np.array(coordinator.trajectory_data['velocities'][frame_idx])
        
        # 计算模式位移和速度
        mode_displacements, mode_velocities = analyzer.project_modes(
            pos_frame,
            vel_frame,
            eigenvectors,
            ref_positions=positions
        )
        
        print(f"  模式投影计算成功，获得 {len(mode_displacements)} 个模式位移和速度")
    except Exception as e:
        print(f"  模式投影计算出错: {e}")
        return
    
    # 测试模式占据数计算
    print("3. 计算模式占据数...")
    try:
        mode_occupations = analyzer.calculate_mode_occupation(
            mode_displacements, 
            mode_velocities, 
            eigenvals, 
            temperature=300  # 假设温度为300K
        )
        
        # 计算理论玻色-爱因斯坦分布
        be_distribution = analyzer.calculate_bose_einstein_distribution(
            freq_thz,
            temperature=300  # 假设温度为300K
        )
        
        # 保存结果
        occ_file = os.path.join(OUTPUT_DIR, "mode_occupation.txt")
        np.savetxt(occ_file, 
                  np.column_stack((freq_thz[:10], mode_occupations[:10], be_distribution[:10])),
                  header="Frequency(THz) Occupation BE_Distribution", 
                  comments='# ')
        
        # 绘制前10个模式的占据数对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(freq_thz[:10], mode_occupations[:10], label='Calculated')
        plt.plot(freq_thz[:10], be_distribution[:10], 'r-', label='Bose-Einstein')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Mode Occupation')
        plt.title('Mode Occupation vs. Bose-Einstein Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "mode_occupation.png"))
        plt.close()
        
        print(f"  模式占据数计算成功，结果已保存")
    except Exception as e:
        print(f"  模式占据数计算出错: {e}")
    
    # 测试模式有效温度计算
    print("4. 计算模式有效温度...")
    try:
        if 'mode_occupations' in locals():
            # 计算模式有效温度
            mode_temperatures = analyzer.calculate_mode_temperatures(
                mode_occupations, 
                freq_thz
            )
            
            # 保存结果
            temp_file = os.path.join(OUTPUT_DIR, "mode_temperatures.txt")
            np.savetxt(temp_file, 
                      np.column_stack((freq_thz[:10], mode_temperatures[:10])),
                      header="Frequency(THz) Temperature(K)", 
                      comments='# ')
            
            # 绘制前10个模式的有效温度
            plt.figure(figsize=(10, 6))
            plt.scatter(freq_thz[:10], mode_temperatures[:10])
            plt.axhline(y=300, color='r', linestyle='--', label='Equilibrium Temperature (300K)')
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Mode Temperature (K)')
            plt.title('Mode Effective Temperature')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "mode_temperatures.png"))
            plt.close()
            
            print(f"  模式有效温度计算成功，结果已保存")
        else:
            print("  缺少模式占据数数据，跳过模式温度计算")
    except Exception as e:
        print(f"  模式有效温度计算出错: {e}")
    
    # 测试格林艾森参数计算
    print("5. 计算格林艾森参数...")
    try:
        # 假设我们有两个不同体积下的特征值
        eigenvals_compressed = eigenvals * 1.1  # 模拟压缩状态下的特征值
        
        # 计算格林艾森参数
        gruneisen_params = analyzer.calculate_gruneisen_parameters(
            eigenvals, 
            eigenvals_compressed, 
            volume_change_ratio=0.1  # 体积变化率
        )
        
        # 保存结果
        gru_file = os.path.join(OUTPUT_DIR, "gruneisen_parameters.txt")
        np.savetxt(gru_file, 
                  np.column_stack((freq_thz[:10], gruneisen_params[:10])),
                  header="Frequency(THz) Gruneisen_Parameter", 
                  comments='# ')
        
        # 绘制前10个模式的格林艾森参数
        plt.figure(figsize=(10, 6))
        plt.scatter(freq_thz[:10], gruneisen_params[:10])
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Grüneisen Parameter')
        plt.title('Mode Grüneisen Parameters')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "gruneisen_parameters.png"))
        plt.close()
        
        print(f"  格林艾森参数计算成功，结果已保存")
    except Exception as e:
        print(f"  格林艾森参数计算出错: {e}")

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
        
        # 计算时间分辨DOS
        time_resolved_dos = analyzer.calculate_time_resolved_dos(
            velocities,
            window_size=1
        )
        
        if len(time_resolved_dos) > 0:
            # 绘制时间分辨DOS热图
            plt.figure(figsize=(10, 6))
            plt.imshow(
                time_resolved_dos.T,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, len(coordinator.trajectory_data['positions']), 0, 20]  # 假设频率范围为0-20 THz
            )
            plt.colorbar(label='DOS Intensity')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency (THz)')
            plt.title('Time-Resolved DOS')
            plt.savefig(os.path.join(OUTPUT_DIR, "time_resolved_dos.png"))
            plt.close()
            
            print(f"  时间分辨DOS计算成功，形状: {time_resolved_dos.shape}")
        else:
            print("  时间分辨DOS计算结果为空")
    except Exception as e:
        print(f"  计算时间分辨DOS时出错: {e}")
    
    # 测试频率带能量随时间演化分析
    print("2. 分析频率带能量随时间演化...")
    try:
        # 获取声子分析器计算简正模式
        phonon_analyzer = PhononAnalyzer(coordinator)
        positions = np.array(coordinator.trajectory_data['positions'][0])
        masses = np.ones(positions.shape[0])  # 假设所有原子质量相同
        
        eigenvals, eigenvectors = phonon_analyzer.calculate_normal_modes(
            positions,
            masses
        )
        
        # 定义频率带
        freq_thz = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
        freq_bands = [(0, 5), (5, 10), (10, 15)]  # 定义3个频率带：0-5, 5-10, 10-15 THz
        
        # 转换数据类型
        all_positions = np.array(coordinator.trajectory_data['positions'])
        all_velocities = np.array(coordinator.trajectory_data['velocities'])
        
        # 计算频率带能量随时间演化
        band_energies = analyzer.calculate_band_energy_evolution(
            all_positions,
            all_velocities,
            eigenvectors,
            eigenvals,
            freq_bands
        )
        
        if len(band_energies) > 0:
            # 绘制频率带能量随时间演化曲线
            plt.figure(figsize=(10, 6))
            for i, band in enumerate(freq_bands):
                plt.plot(range(len(coordinator.trajectory_data['positions'])), band_energies[:, i], 
                         label=f'{band[0]}-{band[1]} THz')
            
            plt.xlabel('Time Frame')
            plt.ylabel('Band Energy (eV)')
            plt.title('Frequency Band Energy Evolution')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "band_energy_evolution.png"))
            plt.close()
            
            print(f"  频率带能量演化计算成功，分析了 {len(freq_bands)} 个频率带")
        else:
            print("  频率带能量演化计算结果为空")
    except Exception as e:
        print(f"  分析频率带能量随时间演化时出错: {e}")

def test_equilibration_analyzer(coordinator):
    """测试EquilibrationAnalyzer平衡分析模块"""
    print_section("测试EquilibrationAnalyzer平衡分析模块")
    
    # 创建平衡分析器
    analyzer = EquilibrationAnalyzer(coordinator)
    
    # 测试系统平衡过程分析
    print("1. 分析系统平衡过程...")
    try:
        if hasattr(coordinator, 'energy_data') and len(coordinator.energy_data) > 0:
            # 分析能量平衡过程
            energy_data = np.array(coordinator.energy_data[:, 1])  # 使用总能量列
            
            equilibration_time, equilibration_metric = analyzer.analyze_energy_equilibration(
                energy_data,
                window_size=1
            )
            
            print(f"  估计的能量平衡时间: 第 {equilibration_time} 帧")
            print(f"  平衡度量值: {equilibration_metric}")
            
            # 绘制能量平衡过程
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(coordinator.energy_data)), coordinator.energy_data[:, 1])
            plt.axvline(x=equilibration_time, color='r', linestyle='--', 
                        label=f'Equilibration Time: Frame {equilibration_time}')
            plt.xlabel('Time Frame')
            plt.ylabel('Total Energy (eV)')
            plt.title('System Energy Equilibration')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "energy_equilibration.png"))
            plt.close()
        else:
            print("  无能量数据，跳过系统平衡过程分析")
    except Exception as e:
        print(f"  分析系统平衡过程时出错: {e}")
    
    # 测试模式平衡时间分析
    print("2. 分析模式平衡时间...")
    try:
        # 获取声子分析器计算简正模式
        phonon_analyzer = PhononAnalyzer(coordinator)
        positions = np.array(coordinator.trajectory_data['positions'][0])
        masses = np.ones(positions.shape[0])  # 假设所有原子质量相同
        
        eigenvals, eigenvectors = phonon_analyzer.calculate_normal_modes(
            positions,
            masses
        )
        
        # 转换数据类型
        all_positions = np.array(coordinator.trajectory_data['positions'])
        all_velocities = np.array(coordinator.trajectory_data['velocities'])
        
        # 计算模式平衡时间
        # 注意：由于我们的测试数据只有几帧，这里仅作为API测试，实际结果可能不准确
        mode_eq_times = analyzer.analyze_mode_equilibration(
            all_positions,
            all_velocities,
            eigenvectors,
            eigenvals,
            window_size=1
        )
        
        if isinstance(mode_eq_times, np.ndarray) and len(mode_eq_times) > 0:
            # 绘制前10个模式的平衡时间
            freq_thz = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
            plt.figure(figsize=(10, 6))
            plt.scatter(freq_thz[:10], mode_eq_times[:10])
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Equilibration Time (frames)')
            plt.title('Mode Equilibration Times')
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "mode_equilibration_times.png"))
            plt.close()
            
            print(f"  模式平衡时间分析成功，分析了 {len(mode_eq_times)} 个模式")
        else:
            print("  模式平衡时间分析结果为空")
    except Exception as e:
        print(f"  分析模式平衡时间时出错: {e}")
    
    # 测试热平衡分析
    print("3. 分析热平衡过程...")
    try:
        # 使用能量数据中的动能作为温度指标
        if hasattr(coordinator, 'energy_data') and len(coordinator.energy_data) > 0:
            # 从动能估算温度（简化）
            temp_data = coordinator.energy_data[:, 2] * 2 / (3 * positions.shape[0])
            
            # 分析温度平衡过程
            temp_eq_time, temp_eq_metric = analyzer.analyze_thermal_equilibration(
                temp_data,
                target_temperature=300,  # 假设目标温度为300K
                window_size=1
            )
            
            print(f"  估计的温度平衡时间: 第 {temp_eq_time} 帧")
            print(f"  温度平衡度量值: {temp_eq_metric}")
            
            # 绘制温度平衡过程
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(temp_data)), temp_data)
            plt.axvline(x=temp_eq_time, color='r', linestyle='--', 
                        label=f'Thermal Equilibration Time: Frame {temp_eq_time}')
            plt.axhline(y=300, color='g', linestyle='--', label='Target Temperature (300K)')
            plt.xlabel('Time Frame')
            plt.ylabel('Temperature (K)')
            plt.title('System Thermal Equilibration')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "thermal_equilibration.png"))
            plt.close()
        else:
            print("  无能量数据，跳过热平衡分析")
    except Exception as e:
        print(f"  分析热平衡过程时出错: {e}")
    
    # 测试平衡判断功能
    print("4. 检测系统是否达到平衡...")
    try:
        # 使用能量数据中的总能量
        if hasattr(coordinator, 'energy_data') and len(coordinator.energy_data) > 0:
            energy_data = np.array(coordinator.energy_data[:, 1])  # 使用总能量列
            
            is_equilibrated = analyzer.is_system_equilibrated(
                energy_data,
                tolerance=0.01,  # 相对波动1%
                window_size=1
            )
            
            print(f"  系统是否已平衡: {is_equilibrated}")
        else:
            print("  无能量数据，跳过平衡检测")
    except Exception as e:
        print(f"  检测系统平衡状态时出错: {e}")

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