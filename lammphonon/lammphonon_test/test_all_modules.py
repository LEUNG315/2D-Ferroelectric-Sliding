#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon全功能测试脚本
========================

测试LAMMPhonon的所有功能模块，确保其正常工作。

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
    SlidingAnalyzer,
    ThermalAnalyzer,
    AnharmonicAnalyzer,
    TemporalAnalyzer,
    EquilibrationAnalyzer
)

def ensure_dir(directory):
    """确保目录存在，如不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def print_section(title):
    """打印带分隔符的小节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_coordinator(data_dir, output_dir):
    """测试PhononCoordinator数据读取功能"""
    print_section("测试PhononCoordinator数据读取")
    
    # 创建协调器实例
    coordinator = PhononCoordinator()
    
    # 设置输出目录
    coordinator.output_dir = output_dir
    
    # 读取轨迹文件
    traj_file = os.path.join(data_dir, "test_dump.phonon")
    print(f"读取轨迹文件: {traj_file}")
    coordinator.read_trajectory(traj_file)
    
    # 读取能量文件
    energy_file = os.path.join(data_dir, "test_energy.txt")
    print(f"读取能量文件: {energy_file}")
    coordinator.read_energy_data(energy_file)
    
    # 读取热流数据
    heatflux_file = os.path.join(data_dir, "test_heatflux.dat")
    print(f"读取热流文件: {heatflux_file}")
    coordinator.read_heatflux_data(heatflux_file)
    
    # 读取极化数据
    polarization_file = os.path.join(data_dir, "test_polarization.txt")
    print(f"读取极化文件: {polarization_file}")
    coordinator.read_polarization_data(polarization_file)
    
    # 打印数据摘要
    print(f"轨迹数据帧数: {len(coordinator.trajectory_data['positions']) if hasattr(coordinator, 'trajectory_data') else 0}")
    print(f"原子数: {coordinator.trajectory_data['positions'][0].shape[0] if hasattr(coordinator, 'trajectory_data') and len(coordinator.trajectory_data['positions']) > 0 else 0}")
    print(f"能量数据点数: {len(coordinator.energy_data) if hasattr(coordinator, 'energy_data') else 0}")
    print(f"热流数据点数: {len(coordinator.heatflux_data) if hasattr(coordinator, 'heatflux_data') else 0}")
    print(f"极化数据点数: {len(coordinator.polarization_data) if hasattr(coordinator, 'polarization_data') else 0}")
    
    return coordinator

def test_phonon_analyzer(coordinator, output_dir):
    """测试PhononAnalyzer基础声子分析功能"""
    print_section("测试PhononAnalyzer基础声子分析")
    
    # 创建声子分析器
    analyzer = PhononAnalyzer(coordinator)
    
    # 计算速度自相关函数
    print("计算速度自相关函数...")
    velocities = np.array(coordinator.trajectory_data['velocities'])
    vacf = analyzer.calculate_velocity_autocorrelation(velocities)
    
    # 计算声子态密度
    print("计算声子态密度...")
    freqs, dos = analyzer.calculate_dos(vacf)
    
    # 保存DOS结果
    print("保存DOS计算结果...")
    dos_file = os.path.join(output_dir, "dos.txt")
    np.savetxt(dos_file, np.column_stack((freqs, dos)), 
               header="Frequency(THz) DOS", comments='# ')
    
    # 绘制DOS图像
    plt.figure(figsize=(8, 6))
    plt.plot(freqs, dos)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS')
    plt.title('Phonon Density of States')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dos.png"))
    plt.close()
    
    # 计算简正模式
    print("计算简正模式...")
    try:
        positions = np.array(coordinator.trajectory_data['positions'][0])
        masses = np.ones(positions.shape[0])  # 假设所有原子质量相同
        
        eigenvals, eigenvectors = analyzer.calculate_normal_modes(
            positions,
            masses
        )
        print(f"获得特征值数量: {len(eigenvals)}")
        
        # 保存前几个模式频率
        mode_file = os.path.join(output_dir, "modes.txt")
        with open(mode_file, 'w') as f:
            f.write("# Mode Frequency(THz)\n")
            for i, freq in enumerate(np.sqrt(np.abs(eigenvals)) / (2 * np.pi)):
                if i < 10:  # 只保存前10个模式
                    f.write(f"{i} {freq}\n")
        
    except Exception as e:
        print(f"计算简正模式时出错: {e}")
    
    return analyzer

def test_sliding_analyzer(coordinator, output_dir):
    """测试SlidingAnalyzer滑移分析功能"""
    print_section("测试SlidingAnalyzer滑移分析")
    
    # 创建滑移分析器
    analyzer = SlidingAnalyzer(coordinator)
    
    # 检测材料层
    print("检测材料层...")
    try:
        positions = np.array(coordinator.trajectory_data['positions'][0])
        analyzer.detect_material_layers(positions)
        print(f"检测到 {len(analyzer.layers)} 层，原子数分别为: {[len(layer) for layer in analyzer.layers]}")
        
        # 计算滑移距离
        print("计算滑移距离...")
        all_positions = np.array(coordinator.trajectory_data['positions'])
        sliding_distance = analyzer.calculate_sliding_distance(all_positions)
        print(f"滑移距离数据点数: {len(sliding_distance)}")
        
        # 计算层间距离
        print("计算层间距离...")
        interlayer_distance = analyzer.calculate_interlayer_distance(all_positions)
        print(f"层间距离数据点数: {len(interlayer_distance)}")
        
        # 保存结果
        sliding_file = os.path.join(output_dir, "sliding_distance.txt")
        np.savetxt(sliding_file, 
                   np.column_stack((range(len(sliding_distance)), sliding_distance)),
                   header="Frame Distance(Angstrom)", comments='# ')
        
        # 绘制滑移距离图像
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(sliding_distance)), sliding_distance)
        plt.xlabel('Frame')
        plt.ylabel('Sliding Distance (Å)')
        plt.title('Sliding Distance Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "sliding_distance.png"))
        plt.close()
        
    except Exception as e:
        print(f"滑移分析时出错: {e}")
    
    return analyzer

def test_thermal_analyzer(coordinator, output_dir):
    """测试ThermalAnalyzer热分析功能"""
    print_section("测试ThermalAnalyzer热分析")
    
    # 创建热分析器
    analyzer = ThermalAnalyzer(coordinator)
    
    # 测试热流相关性计算
    print("计算热流自相关函数...")
    try:
        if hasattr(coordinator, 'heatflux_data') and len(coordinator.heatflux_data) > 0:
            hfacf = analyzer.calculate_heatflux_autocorrelation(
                coordinator.heatflux_data, 
                max_correlation_time=len(coordinator.heatflux_data) - 1
            )
            print(f"热流自相关函数长度: {len(hfacf)}")
            
            # 绘制热流自相关函数
            plt.figure(figsize=(8, 6))
            plt.plot(range(len(hfacf)), hfacf)
            plt.xlabel('Time Lag')
            plt.ylabel('HFACF')
            plt.title('Heat Flux Autocorrelation Function')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "hfacf.png"))
            plt.close()
        else:
            print("无热流数据，跳过热流相关性计算")
    except Exception as e:
        print(f"计算热流自相关函数时出错: {e}")
    
    # 测试计算热导率
    print("尝试计算热导率...")
    try:
        if hasattr(coordinator, 'heatflux_data') and len(coordinator.heatflux_data) > 0:
            thermal_conductivity = analyzer.calculate_thermal_conductivity(
                coordinator.heatflux_data,
                temperature=300,  # 假设温度为300K
                volume=coordinator.box_size[0] * coordinator.box_size[1] * coordinator.box_size[2]
            )
            print(f"估计的热导率: {thermal_conductivity} W/mK")
        else:
            print("无热流数据，跳过热导率计算")
    except Exception as e:
        print(f"计算热导率时出错: {e}")
    
    # 测试极化-热流相关性分析
    print("分析极化-热流相关性...")
    try:
        if (hasattr(coordinator, 'polarization_data') and 
            hasattr(coordinator, 'heatflux_data') and
            len(coordinator.polarization_data) > 0 and
            len(coordinator.heatflux_data) > 0):
            
            correlation = analyzer.analyze_polarization_heatflux_correlation(
                coordinator.polarization_data,
                coordinator.heatflux_data
            )
            print(f"极化-热流相关系数: {correlation}")
        else:
            print("缺少极化或热流数据，跳过相关性分析")
    except Exception as e:
        print(f"分析极化-热流相关性时出错: {e}")
    
    return analyzer

def test_anharmonic_analyzer(coordinator, output_dir):
    """测试AnharmonicAnalyzer非谐性分析功能"""
    print_section("测试AnharmonicAnalyzer非谐性分析")
    
    # 创建非谐性分析器
    analyzer = AnharmonicAnalyzer(coordinator)
    
    # 获取声子分析器计算简正模式
    phonon_analyzer = PhononAnalyzer(coordinator)
    
    # 测试计算模式占据数
    print("计算模式占据数...")
    try:
        # 计算简正模式
        positions = np.array(coordinator.trajectory_data['positions'][0])
        masses = np.ones(positions.shape[0])  # 假设所有原子质量相同
        
        eigenvals, eigenvectors = phonon_analyzer.calculate_normal_modes(
            positions,
            masses
        )
        
        # 选择一个时间帧进行测试
        frame_idx = 0
        
        # 计算模式位移和速度
        pos_frame = np.array(coordinator.trajectory_data['positions'][frame_idx])
        vel_frame = np.array(coordinator.trajectory_data['velocities'][frame_idx])
        
        mode_displacements, mode_velocities = analyzer.project_to_normal_modes(
            pos_frame,
            vel_frame,
            eigenvectors
        )
        
        # 计算模式占据数
        mode_occupations = analyzer.calculate_mode_occupation(
            mode_displacements, 
            mode_velocities, 
            eigenvals, 
            temperature=300  # 假设温度为300K
        )
        
        # 计算理论玻色-爱因斯坦分布
        freq_thz = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
        be_distribution = analyzer.calculate_bose_einstein_distribution(
            freq_thz, 
            temperature=300  # 假设温度为300K
        )
        
        # 保存结果
        occ_file = os.path.join(output_dir, "mode_occupation.txt")
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
        plt.savefig(os.path.join(output_dir, "mode_occupation.png"))
        plt.close()
        
        print(f"计算了 {len(mode_occupations)} 个模式的占据数")
        
    except Exception as e:
        print(f"计算模式占据数时出错: {e}")
    
    # 测试计算模式有效温度
    print("计算模式有效温度...")
    try:
        if 'mode_occupations' in locals():
            # 计算模式有效温度
            mode_temperatures = analyzer.calculate_mode_temperatures(
                mode_occupations, 
                freq_thz
            )
            
            # 保存结果
            temp_file = os.path.join(output_dir, "mode_temperatures.txt")
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
            plt.savefig(os.path.join(output_dir, "mode_temperatures.png"))
            plt.close()
            
            print(f"计算了 {len(mode_temperatures)} 个模式的有效温度")
        else:
            print("缺少模式占据数数据，跳过模式温度计算")
    except Exception as e:
        print(f"计算模式有效温度时出错: {e}")
    
    # 测试计算格林艾森参数
    print("计算格林艾森参数...")
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
        gru_file = os.path.join(output_dir, "gruneisen_parameters.txt")
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
        plt.savefig(os.path.join(output_dir, "gruneisen_parameters.png"))
        plt.close()
        
        print(f"计算了 {len(gruneisen_params)} 个模式的格林艾森参数")
        
    except Exception as e:
        print(f"计算格林艾森参数时出错: {e}")
    
    return analyzer

def test_temporal_analyzer(coordinator, output_dir):
    """测试TemporalAnalyzer时间分析功能"""
    print_section("测试TemporalAnalyzer时间分析")
    
    # 创建时间分析器
    analyzer = TemporalAnalyzer(coordinator)
    
    # 测试时间分辨DOS计算
    print("计算时间分辨DOS...")
    try:
        # 选择一个子集的时间帧进行计算
        # 由于我们的测试数据只有3帧，这里我们使用所有帧
        
        # 创建声子分析器用于计算DOS
        phonon_analyzer = PhononAnalyzer(coordinator)
        
        # 转换数据类型
        velocities = np.array(coordinator.trajectory_data['velocities'])
        
        # 计算时间分辨DOS
        time_resolved_dos = analyzer.calculate_time_resolved_dos(
            velocities,
            window_size=1,
            step_size=1
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
            plt.savefig(os.path.join(output_dir, "time_resolved_dos.png"))
            plt.close()
            
            print(f"计算了时间分辨DOS，形状: {time_resolved_dos.shape}")
        else:
            print("时间分辨DOS计算结果为空")
    except Exception as e:
        print(f"计算时间分辨DOS时出错: {e}")
    
    # 测试频率带能量随时间演化分析
    print("分析频率带能量随时间演化...")
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
            plt.savefig(os.path.join(output_dir, "band_energy_evolution.png"))
            plt.close()
            
            print(f"计算了 {len(freq_bands)} 个频率带的能量演化，形状: {band_energies.shape}")
        else:
            print("频率带能量演化计算结果为空")
    except Exception as e:
        print(f"分析频率带能量随时间演化时出错: {e}")
    
    return analyzer

def test_equilibration_analyzer(coordinator, output_dir):
    """测试EquilibrationAnalyzer平衡分析功能"""
    print_section("测试EquilibrationAnalyzer平衡分析")
    
    # 创建平衡分析器
    analyzer = EquilibrationAnalyzer(coordinator)
    
    # 测试系统平衡过程分析
    print("分析系统平衡过程...")
    try:
        if hasattr(coordinator, 'energy_data') and len(coordinator.energy_data) > 0:
            # 分析能量平衡过程
            energy_data = np.array(coordinator.energy_data[:, 1])  # 使用总能量列
            
            equilibration_time, equilibration_metric = analyzer.analyze_energy_equilibration(
                energy_data,
                window_size=1
            )
            
            print(f"估计的能量平衡时间: 第 {equilibration_time} 帧")
            print(f"平衡度量值: {equilibration_metric}")
            
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
            plt.savefig(os.path.join(output_dir, "energy_equilibration.png"))
            plt.close()
        else:
            print("无能量数据，跳过系统平衡过程分析")
    except Exception as e:
        print(f"分析系统平衡过程时出错: {e}")
    
    # 测试模式相关平衡时间计算
    print("计算模式相关平衡时间...")
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
        
        # 计算模式相关平衡时间
        # 注意：由于我们的测试数据只有3帧，这里仅作为API测试，实际结果可能不准确
        mode_eq_times = analyzer.calculate_mode_equilibration_times(
            all_positions,
            all_velocities,
            eigenvectors,
            eigenvals,
            window_size=1
        )
        
        if len(mode_eq_times) > 0:
            # 绘制前10个模式的平衡时间
            freq_thz = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
            plt.figure(figsize=(10, 6))
            plt.scatter(freq_thz[:10], mode_eq_times[:10])
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Equilibration Time (frames)')
            plt.title('Mode Equilibration Times')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "mode_equilibration_times.png"))
            plt.close()
            
            print(f"计算了 {len(mode_eq_times)} 个模式的平衡时间")
        else:
            print("模式平衡时间计算结果为空")
    except Exception as e:
        print(f"计算模式相关平衡时间时出错: {e}")
    
    # 测试热平衡分析
    print("分析热平衡过程...")
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
            
            print(f"估计的温度平衡时间: 第 {temp_eq_time} 帧")
            print(f"温度平衡度量值: {temp_eq_metric}")
            
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
            plt.savefig(os.path.join(output_dir, "thermal_equilibration.png"))
            plt.close()
        else:
            print("无能量数据，跳过热平衡分析")
    except Exception as e:
        print(f"分析热平衡过程时出错: {e}")
    
    return analyzer

def run_all_tests():
    """运行所有测试"""
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "results")
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    print(f"测试数据目录: {data_dir}")
    print(f"输出结果目录: {output_dir}")
    
    # 测试PhononCoordinator
    coordinator = test_coordinator(data_dir, output_dir)
    
    # 测试PhononAnalyzer
    phonon_analyzer = test_phonon_analyzer(coordinator, output_dir)
    
    # 测试SlidingAnalyzer
    sliding_analyzer = test_sliding_analyzer(coordinator, output_dir)
    
    # 测试ThermalAnalyzer
    thermal_analyzer = test_thermal_analyzer(coordinator, output_dir)
    
    # 测试AnharmonicAnalyzer
    anharmonic_analyzer = test_anharmonic_analyzer(coordinator, output_dir)
    
    # 测试TemporalAnalyzer
    temporal_analyzer = test_temporal_analyzer(coordinator, output_dir)
    
    # 测试EquilibrationAnalyzer
    equilibration_analyzer = test_equilibration_analyzer(coordinator, output_dir)
    
    print_section("测试完成")
    print(f"所有测试结果已保存至: {output_dir}")

if __name__ == "__main__":
    run_all_tests() 