#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon - 综合分析示例

本示例展示如何使用LAMMPhonon进行全面的声子分析，
包括声子态密度、声子模式投影、滑移分析、热分析等多种功能的组合使用。

作者: 梁树铭 (Shuming Liang)
邮箱: lsm315@mail.ustc.edu.cn
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# 导入LAMMPhonon包
from lammphonon import (
    PhononCoordinator, 
    PhononAnalyzer, 
    SlidingAnalyzer,
    ThermalAnalyzer,
    EquilibrationAnalyzer,
    TemporalAnalyzer
)
# 导入修复工具
from lammphonon.fixer import apply_all_patches

class ComprehensiveAnalysis:
    """综合分析类，整合多种声子分析功能"""
    
    def __init__(self, trajectory_file=None, output_dir="./comprehensive_results"):
        """
        初始化分析
        
        参数:
            trajectory_file: 轨迹文件路径
            output_dir: 输出目录
        """
        self.trajectory_file = trajectory_file
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建协调器
        self.coordinator = PhononCoordinator()
        self.coordinator.set_config("output_dir", output_dir)
        self.coordinator.set_config("temperature", 300.0)  # 设置默认温度为300K
        
        # 创建各种分析器
        self.phonon_analyzer = None
        self.sliding_analyzer = None
        self.thermal_analyzer = None
        self.equilibration_analyzer = None
        self.temporal_analyzer = None
        
        # 分析结果
        self.results = {}
        
        # 配置
        self.config = {
            'freq_max': 50.0,        # 最大频率(THz)
            'freq_points': 1000,     # 频率点数
            'sigma': 0.1,            # 高斯展宽参数
            'n_layers': 2,           # 材料层数
            'max_correlation_time': 2000,  # 最大相关时间
            'window_size': 100,      # 时间窗口大小
            'window_step': 10,       # 时间窗口步长
        }
    
    def load_data(self, trajectory_file=None, energy_file=None, 
                 force_file=None, heatflux_file=None, mass_file=None,
                 layer_file=None, max_frames=None, frame_stride=1):
        """
        加载数据
        
        参数:
            trajectory_file: 轨迹文件路径
            energy_file: 能量文件路径
            force_file: 力文件路径
            heatflux_file: 热流文件路径
            mass_file: 质量文件路径
            layer_file: 层定义文件路径
            max_frames: 最大帧数
            frame_stride: 帧间隔
        """
        print("========= 数据加载 =========")
        
        # 设置轨迹文件
        if trajectory_file:
            self.trajectory_file = trajectory_file
        
        if not self.trajectory_file:
            raise ValueError("未提供轨迹文件路径")
        
        # 读取轨迹文件
        print(f"读取轨迹文件: {self.trajectory_file}")
        self.coordinator.read_trajectory(self.trajectory_file, max_frames=max_frames, frame_stride=frame_stride)
        
        # 检查轨迹数据
        if not self.coordinator.has_trajectory_data():
            raise ValueError("轨迹数据读取失败")
        
        # 读取能量文件（如果有）
        if energy_file:
            print(f"读取能量文件: {energy_file}")
            self.coordinator.read_energy_data(energy_file)
        
        # 读取力文件（如果有）
        if force_file:
            print(f"读取力文件: {force_file}")
            self.coordinator.read_force_data(force_file)
        
        # 读取热流文件（如果有）
        if heatflux_file:
            print(f"读取热流文件: {heatflux_file}")
            self.coordinator.read_heatflux_data(heatflux_file)
        
        # 读取质量文件（如果有）
        if mass_file:
            print(f"读取质量文件: {mass_file}")
            self.coordinator.read_mass_data(mass_file)
        
        # 读取层定义文件（如果有）
        if layer_file:
            print(f"读取层定义文件: {layer_file}")
            self.coordinator.read_layer_data(layer_file)
        
        print(f"数据加载完成: {len(self.coordinator.trajectory_data['positions'])} 帧")
        
        # 初始化分析器
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """初始化各种分析器"""
        # 应用修复补丁
        print("应用修复补丁...")
        apply_all_patches({
            'PhononAnalyzer': PhononAnalyzer,
            'ThermalAnalyzer': ThermalAnalyzer,
            'TemporalAnalyzer': TemporalAnalyzer,
            'EquilibrationAnalyzer': EquilibrationAnalyzer
        })
        
        # 创建各种分析器
        self.phonon_analyzer = PhononAnalyzer(self.coordinator)
        self.sliding_analyzer = SlidingAnalyzer(self.coordinator)
        self.thermal_analyzer = ThermalAnalyzer(self.coordinator)
        self.equilibration_analyzer = EquilibrationAnalyzer(self.coordinator)
        self.temporal_analyzer = TemporalAnalyzer(self.coordinator)
        
        # 更新配置
        self.coordinator.set_config('freq_max', self.config['freq_max'])
        self.coordinator.set_config('freq_points', self.config['freq_points'])
        self.coordinator.set_config('sigma', self.config['sigma'])
        self.coordinator.set_config('n_layers', self.config['n_layers'])
    
    def run_phonon_analysis(self):
        """运行基础声子分析"""
        print("\n========= 声子分析 =========")
        
        # 计算速度自相关函数(VACF)
        print("计算速度自相关函数(VACF)...")
        velocities = np.array(self.coordinator.trajectory_data['velocities'])
        vacf = self.phonon_analyzer.calculate_velocity_autocorrelation(velocities)
        self.results['vacf'] = vacf
        
        # 计算声子态密度(DOS)
        print("计算声子态密度(DOS)...")
        freqs, dos = self.phonon_analyzer.calculate_dos(vacf)
        self.results['freqs'] = freqs
        self.results['dos'] = dos
        
        # 保存DOS数据
        dos_file = os.path.join(self.output_dir, "dos.txt")
        np.savetxt(dos_file, np.column_stack((freqs, dos)), 
                  header="Frequency (THz), DOS")
        print(f"DOS已保存到: {dos_file}")
        
        # 绘制DOS图
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, dos)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Density of States')
        plt.title('Phonon Density of States')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "dos.png"), dpi=300)
        plt.close()
        
        # 计算正则模式
        try:
            print("计算正则模式...")
            # 获取第一帧的位置
            positions = np.array(self.coordinator.trajectory_data['positions'][0])
            
            # 获取质量（如果有）
            if hasattr(self.coordinator, 'masses') and self.coordinator.masses is not None:
                masses = self.coordinator.masses
            else:
                # 使用单位质量
                masses = np.ones(positions.shape[0])
                print("未提供质量数据，使用单位质量")
            
            # 计算正则模式
            eigenvalues, eigenvectors = self.phonon_analyzer.calculate_normal_modes(positions, masses)
            self.results['eigenvalues'] = eigenvalues
            self.results['eigenvectors'] = eigenvectors
            
            # 计算频率
            mode_freqs = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
            self.results['mode_freqs'] = mode_freqs
            
            # 保存频率数据
            freq_file = os.path.join(self.output_dir, "frequencies.txt")
            np.savetxt(freq_file, mode_freqs, header="Frequency (THz)")
            print(f"频率数据已保存到: {freq_file}")
            
            # 绘制频率分布图
            plt.figure(figsize=(10, 6))
            plt.hist(mode_freqs, bins=50, alpha=0.7)
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Number of Modes')
            plt.title('Phonon Mode Frequency Distribution')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "frequency_distribution.png"), dpi=300)
            plt.close()
            
            # 计算模式占据数
            print("计算模式占据数...")
            mode_occupation = self.phonon_analyzer.calculate_mode_occupation(
                velocities, eigenvectors, eigenvalues, self.coordinator.temperature
            )
            self.results['mode_occupation'] = mode_occupation
            
            # 计算理论占据数（玻色-爱因斯坦分布）
            be_occupation = self.phonon_analyzer.calculate_theoretical_occupation(
                eigenvalues, self.coordinator.temperature
            )
            self.results['be_occupation'] = be_occupation
            
            # 保存占据数数据
            occ_file = os.path.join(self.output_dir, "occupation.txt")
            np.savetxt(occ_file, np.column_stack((mode_freqs, mode_occupation, be_occupation)), 
                      header="Frequency (THz), MD Occupation, BE Occupation")
            print(f"占据数数据已保存到: {occ_file}")
            
            # 绘制占据数比较图
            plt.figure(figsize=(10, 6))
            plt.scatter(mode_freqs, mode_occupation, s=3, alpha=0.7, label='MD')
            plt.plot(mode_freqs, be_occupation, 'r-', label='Bose-Einstein')
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Occupation Number')
            plt.title('Phonon Occupation Comparison')
            plt.grid(True)
            plt.yscale('log')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "occupation_comparison.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"计算正则模式时出错: {str(e)}")
    
    def run_sliding_analysis(self):
        """运行滑移分析"""
        print("\n========= 滑移分析 =========")
        
        # 检测材料层
        print("检测材料层...")
        try:
            self.sliding_analyzer.detect_layers()
            
            # 计算滑移距离
            print("计算滑移距离...")
            distances = self.sliding_analyzer.calculate_sliding_distance()
            self.results['sliding_distances'] = distances
            
            # 生成时间数组
            n_frames = len(distances)
            timestep = self.coordinator.timestep
            times = np.arange(n_frames) * timestep
            
            # 保存滑移距离数据
            dist_file = os.path.join(self.output_dir, "sliding_distance.txt")
            np.savetxt(dist_file, np.column_stack((times, distances)), 
                      header="Time (ps), Sliding Distance (Å)")
            
            # 绘制滑移距离图
            plt.figure(figsize=(10, 6))
            plt.plot(times, distances)
            plt.xlabel('Time (ps)')
            plt.ylabel('Sliding Distance (Å)')
            plt.title('Layer Sliding Distance vs. Time')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "sliding_distance.png"), dpi=300)
            plt.close()
            
            # 尝试计算摩擦力
            try:
                print("计算摩擦力...")
                friction_forces = self.sliding_analyzer.calculate_friction_force()
                self.results['friction_forces'] = friction_forces
                
                # 保存摩擦力数据
                force_file = os.path.join(self.output_dir, "friction_force.txt")
                np.savetxt(force_file, np.column_stack((times, friction_forces)), 
                          header="Time (ps), Friction Force (eV/Å)")
                
                # 绘制摩擦力图
                plt.figure(figsize=(10, 6))
                plt.plot(times, friction_forces)
                plt.xlabel('Time (ps)')
                plt.ylabel('Friction Force (eV/Å)')
                plt.title('Friction Force vs. Time')
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "friction_force.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"计算摩擦力时出错: {str(e)}")
            
            # 尝试计算层间距离
            try:
                print("计算层间距离...")
                interlayer_distances = self.sliding_analyzer.calculate_interlayer_distance()
                self.results['interlayer_distances'] = interlayer_distances
                
                # 保存层间距离数据
                int_dist_file = os.path.join(self.output_dir, "interlayer_distance.txt")
                np.savetxt(int_dist_file, np.column_stack((times, interlayer_distances)), 
                          header="Time (ps), Interlayer Distance (Å)")
                
                # 绘制层间距离图
                plt.figure(figsize=(10, 6))
                plt.plot(times, interlayer_distances)
                plt.xlabel('Time (ps)')
                plt.ylabel('Interlayer Distance (Å)')
                plt.title('Interlayer Distance vs. Time')
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "interlayer_distance.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"计算层间距离时出错: {str(e)}")
                
            # 尝试计算堆垛参数
            try:
                print("计算堆垛参数...")
                stacking_params = self.sliding_analyzer.calculate_stacking_parameters()
                self.results['stacking_params'] = stacking_params
                
                # 保存堆垛参数数据
                stack_file = os.path.join(self.output_dir, "stacking_parameters.txt")
                np.savetxt(stack_file, np.column_stack((times, stacking_params)), 
                          header="Time (ps), registry_x, registry_y, twist_angle")
                
                # 绘制堆垛参数图
                plt.figure(figsize=(10, 6))
                plt.plot(times, stacking_params[:, 0], label='x registry')
                plt.plot(times, stacking_params[:, 1], label='y registry')
                plt.xlabel('Time (ps)')
                plt.ylabel('Registry Parameter')
                plt.title('Stacking Registry Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "registry_evolution.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"计算堆垛参数时出错: {str(e)}")
            
        except Exception as e:
            print(f"滑移分析时出错: {str(e)}")
    
    def run_thermal_analysis(self):
        """运行热分析"""
        print("\n========= 热分析 =========")
        
        # 检查是否有热流数据
        if not hasattr(self.coordinator, 'heatflux_data') or self.coordinator.heatflux_data is None:
            print("未提供热流数据，创建模拟数据...")
            n_frames = len(self.coordinator.trajectory_data['positions'])
            self.coordinator.heatflux_data = np.random.rand(n_frames, 3) * 10.0
        
        # 计算热流自相关函数
        print("计算热流自相关函数...")
        hfacf = self.thermal_analyzer.calculate_heatflux_autocorrelation(
            self.coordinator.heatflux_data,
            max_correlation_time=self.config.get('max_correlation_time')
        )
        
        # 处理返回值
        if isinstance(hfacf, tuple) and len(hfacf) == 2:
            time_lags, hfacf_values = hfacf
        else:
            hfacf_values = hfacf
            time_lags = np.arange(len(hfacf_values)) * self.coordinator.timestep
        
        self.results['hfacf'] = hfacf_values
        
        # 保存HFACF数据
        hfacf_file = os.path.join(self.output_dir, "hfacf.txt")
        np.savetxt(hfacf_file, np.column_stack((time_lags, hfacf_values)), 
                  header="Time (ps), HFACF")
        
        # 绘制HFACF图
        plt.figure(figsize=(10, 6))
        plt.plot(time_lags, hfacf_values)
        plt.xlabel('Time (ps)')
        plt.ylabel('Heat Flux ACF')
        plt.title('Heat Flux Autocorrelation Function')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "hfacf.png"), dpi=300)
        plt.close()
        
        # 计算热导率
        try:
            print("计算热导率...")
            # 获取体系体积
            box = self.coordinator.get_box()
            volume = box['volume']
            
            # 计算热导率
            kappa = self.thermal_analyzer.calculate_thermal_conductivity(
                self.coordinator.heatflux_data,
                temperature=self.coordinator.temperature,
                volume=volume
            )
            
            # 处理返回值
            if isinstance(kappa, tuple) and len(kappa) == 2:
                times, kappa_values = kappa
                final_kappa = kappa_values[-1]
                
                # 保存热导率数据
                kappa_file = os.path.join(self.output_dir, "thermal_conductivity.txt")
                np.savetxt(kappa_file, np.column_stack((times, kappa_values)), 
                          header="Time (ps), Thermal Conductivity (W/mK)")
                
                # 绘制热导率图
                plt.figure(figsize=(10, 6))
                plt.plot(times, kappa_values)
                plt.xlabel('Time (ps)')
                plt.ylabel('Thermal Conductivity (W/mK)')
                plt.title('Green-Kubo Thermal Conductivity')
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "thermal_conductivity.png"), dpi=300)
                plt.close()
            else:
                final_kappa = kappa
                
                # 保存热导率数据
                kappa_file = os.path.join(self.output_dir, "thermal_conductivity.txt")
                with open(kappa_file, 'w') as f:
                    f.write(f"Thermal Conductivity (W/mK): {final_kappa:.6f}\n")
            
            self.results['thermal_conductivity'] = final_kappa
            print(f"热导率: {final_kappa:.4f} W/mK")
            
        except Exception as e:
            print(f"计算热导率时出错: {str(e)}")
    
    def run_equilibration_analysis(self):
        """运行平衡分析"""
        print("\n========= 平衡分析 =========")
        
        # 检查是否有能量数据
        if not hasattr(self.coordinator, 'energy_data') or self.coordinator.energy_data is None:
            print("未提供能量数据，创建模拟数据...")
            n_frames = len(self.coordinator.trajectory_data['positions'])
            times = np.arange(n_frames) * self.coordinator.timestep
            # 创建指数衰减的能量数据
            energies = -1000.0 + 50.0 * np.exp(-times/2.0) + np.random.normal(0, 1, n_frames)
            self.coordinator.energy_data = {'time': times, 'total': energies}
        
        # 提取能量数据
        if isinstance(self.coordinator.energy_data, dict):
            times = self.coordinator.energy_data.get('time', 
                                                    np.arange(len(self.coordinator.energy_data.get('total', []))) * self.coordinator.timestep)
            energies = self.coordinator.energy_data.get('total', [])
        else:
            times = np.arange(len(self.coordinator.energy_data)) * self.coordinator.timestep
            energies = self.coordinator.energy_data
        
        # 计算系统平衡时间
        print("计算系统平衡时间...")
        equil_time, params = self.equilibration_analyzer.calculate_system_equilibration(
            energies, times
        )
        
        if equil_time is not None:
            self.results['equilibration_time'] = equil_time
            print(f"系统平衡时间: {equil_time:.4f} ps")
            
            # 保存平衡分析结果
            equil_file = os.path.join(self.output_dir, "equilibration.txt")
            with open(equil_file, 'w') as f:
                f.write(f"Equilibration Time (ps): {equil_time:.6f}\n")
                if params is not None:
                    f.write(f"Fit Parameters: {params}\n")
            
            # 绘制平衡分析图
            plt.figure(figsize=(10, 6))
            plt.plot(times, energies, 'o', markersize=3, label='Energy Data')
            
            if params is not None:
                # 定义指数衰减函数
                def exp_decay(t, A, tau, E_inf):
                    return A * np.exp(-t / tau) + E_inf
                
                # 计算拟合曲线
                fit_times = np.linspace(min(times), max(times), 1000)
                fit_curve = exp_decay(fit_times, *params)
                plt.plot(fit_times, fit_curve, 'r-', label='Fitted Curve')
            
            plt.axvline(x=equil_time, color='g', linestyle='--',
                       label=f'Equilibration Time: {equil_time:.2f} ps')
            
            plt.xlabel('Time (ps)')
            plt.ylabel('Total Energy (eV)')
            plt.title('System Equilibration Analysis')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "equilibration.png"), dpi=300)
            plt.close()
        else:
            print("无法确定系统平衡时间")
    
    def run_temporal_analysis(self):
        """运行时间分辨分析"""
        print("\n========= 时间分辨分析 =========")
        
        # 获取速度数据
        velocities = np.array(self.coordinator.trajectory_data['velocities'])
        
        # 设置时间窗口参数
        self.temporal_analyzer.window_size = self.config.get('window_size', 100)
        self.temporal_analyzer.window_step = self.config.get('window_step', 10)
        
        # 计算时间分辨DOS
        print("计算时间分辨DOS...")
        try:
            frequencies, dos_evolution, time_points = self.temporal_analyzer.calculate_time_resolved_dos(
                velocities,
                window_size=self.config.get('window_size'),
                window_step=self.config.get('window_step'),
                freq_max=self.config.get('freq_max'),
                freq_points=self.config.get('freq_points'),
                sigma=self.config.get('sigma')
            )
            
            self.results['tr_frequencies'] = frequencies
            self.results['tr_dos_evolution'] = dos_evolution
            self.results['tr_time_points'] = time_points
            
            # 保存时间分辨DOS数据
            tr_dos_file = os.path.join(self.output_dir, "time_resolved_dos.npz")
            np.savez(tr_dos_file, 
                    frequencies=frequencies, 
                    dos_evolution=dos_evolution, 
                    time_points=time_points)
            print(f"时间分辨DOS数据已保存到: {tr_dos_file}")
            
            # 绘制第一个时间点的DOS
            plt.figure(figsize=(10, 6))
            plt.plot(frequencies, dos_evolution[0])
            plt.xlabel('Frequency (THz)')
            plt.ylabel('DOS')
            plt.title(f'Phonon DOS at t={time_points[0]:.3f} ps')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "dos_first_timepoint.png"), dpi=300)
            plt.close()
            
            # 如果有多个时间点，绘制DOS演化热图
            if len(time_points) > 1:
                plt.figure(figsize=(12, 8))
                plt.imshow(
                    dos_evolution.T,
                    aspect='auto',
                    origin='lower',
                    extent=[min(time_points), max(time_points), min(frequencies), max(frequencies)],
                    cmap='viridis'
                )
                plt.colorbar(label='DOS')
                plt.xlabel('Time (ps)')
                plt.ylabel('Frequency (THz)')
                plt.title('Time-Resolved Phonon DOS')
                plt.savefig(os.path.join(self.output_dir, "dos_evolution.png"), dpi=300)
                plt.close()
                
                # 绘制几个时间点的DOS比较
                plt.figure(figsize=(10, 6))
                n_times = min(5, len(time_points))
                indices = np.linspace(0, len(time_points)-1, n_times, dtype=int)
                
                for i, idx in enumerate(indices):
                    t = time_points[idx]
                    plt.plot(frequencies, dos_evolution[idx], label=f't={t:.2f} ps')
                
                plt.xlabel('Frequency (THz)')
                plt.ylabel('DOS')
                plt.title('DOS Evolution Over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "dos_comparison.png"), dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"计算时间分辨DOS时出错: {str(e)}")
    
    def generate_report(self):
        """生成分析报告"""
        print("\n========= 生成报告 =========")
        
        report_file = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("=================================================\n")
            f.write("         LAMMPhonon 综合分析报告                   \n")
            f.write("=================================================\n\n")
            
            f.write(f"分析时间: {self.phonon_analyzer.get_timestamp()}\n\n")
            
            f.write("系统信息:\n")
            f.write(f"- 轨迹文件: {self.trajectory_file}\n")
            f.write(f"- 帧数: {len(self.coordinator.trajectory_data['positions'])}\n")
            f.write(f"- 原子数: {self.coordinator.trajectory_data['positions'][0].shape[0]}\n")
            f.write(f"- 温度: {self.coordinator.temperature} K\n")
            
            box = self.coordinator.get_box()
            if box:
                f.write(f"- 模拟盒尺寸: {box['dimensions']} Å\n")
                f.write(f"- 体积: {box['volume']:.2f} Å³\n")
            
            f.write("\n声子分析结果:\n")
            if 'freqs' in self.results and 'dos' in self.results:
                max_dos_idx = np.argmax(self.results['dos'])
                max_dos_freq = self.results['freqs'][max_dos_idx]
                f.write(f"- DOS峰值频率: {max_dos_freq:.4f} THz\n")
            
            if 'mode_freqs' in self.results:
                f.write(f"- 声子模式数: {len(self.results['mode_freqs'])}\n")
                f.write(f"- 最大频率: {np.max(self.results['mode_freqs']):.4f} THz\n")
                f.write(f"- 最小频率: {np.min(self.results['mode_freqs']):.4f} THz\n")
            
            if 'sliding_distances' in self.results:
                f.write("\n滑移分析结果:\n")
                f.write(f"- 最大滑移距离: {np.max(self.results['sliding_distances']):.4f} Å\n")
                
                if 'friction_forces' in self.results:
                    avg_force = np.mean(np.abs(self.results['friction_forces']))
                    f.write(f"- 平均摩擦力: {avg_force:.4f} eV/Å\n")
                
                if 'interlayer_distances' in self.results:
                    avg_dist = np.mean(self.results['interlayer_distances'])
                    f.write(f"- 平均层间距离: {avg_dist:.4f} Å\n")
            
            if 'thermal_conductivity' in self.results:
                f.write("\n热分析结果:\n")
                f.write(f"- 热导率: {self.results['thermal_conductivity']:.4f} W/mK\n")
            
            if 'equilibration_time' in self.results:
                f.write("\n平衡分析结果:\n")
                f.write(f"- 系统平衡时间: {self.results['equilibration_time']:.4f} ps\n")
            
            f.write("\n备注: 本报告由LAMMPhonon自动生成\n")
            f.write("作者: 梁树铭 (Shuming Liang)\n")
            f.write("邮箱: lsm315@mail.ustc.edu.cn\n")
        
        print(f"分析报告已生成: {report_file}")
    
    def run_all(self):
        """运行所有分析"""
        if not self.coordinator.has_trajectory_data():
            raise ValueError("请先加载数据")
        
        # 运行所有分析
        self.run_phonon_analysis()
        self.run_sliding_analysis()
        self.run_thermal_analysis()
        self.run_equilibration_analysis()
        self.run_temporal_analysis()
        
        # 生成报告
        self.generate_report()
        
        print("\n所有分析已完成!")
        print(f"结果已保存到: {self.output_dir}")

def main():
    """主函数"""
    print("LAMMPhonon 综合分析示例")
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='LAMMPhonon综合分析示例')
    parser.add_argument('-i', '--input', type=str, default="../data/dump.phonon",
                       help='轨迹文件路径')
    parser.add_argument('-o', '--output', type=str, default="./comprehensive_results",
                       help='输出目录')
    parser.add_argument('-e', '--energy', type=str,
                       help='能量文件路径')
    parser.add_argument('-f', '--force', type=str,
                       help='力文件路径')
    parser.add_argument('-hf', '--heatflux', type=str,
                       help='热流文件路径')
    parser.add_argument('-m', '--mass', type=str,
                       help='质量文件路径')
    parser.add_argument('-l', '--layer', type=str,
                       help='层定义文件路径')
    parser.add_argument('--max-frames', type=int,
                       help='最大帧数')
    parser.add_argument('--frame-stride', type=int, default=1,
                       help='帧间隔')
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ComprehensiveAnalysis(
        trajectory_file=args.input,
        output_dir=args.output
    )
    
    # 加载数据
    try:
        analyzer.load_data(
            energy_file=args.energy,
            force_file=args.force,
            heatflux_file=args.heatflux,
            mass_file=args.mass,
            layer_file=args.layer,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride
        )
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return
    
    # 运行所有分析
    try:
        analyzer.run_all()
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 