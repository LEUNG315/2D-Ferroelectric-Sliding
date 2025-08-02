#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main interactive menu for lammphonon
"""

import os
import sys
import time
import logging
import numpy as np
from ..core.coordinator import PhononCoordinator
from ..core.phonon_analyzer import PhononAnalyzer
from ..analysis.sliding_analyzer import SlidingAnalyzer
from ..utils.timing import Timer
import datetime

# Setup logging
logger = logging.getLogger(__name__)

class MainMenu:
    """Main interactive menu for lammphonon"""
    
    def __init__(self):
        """Initialize the main menu"""
        self.coordinator = PhononCoordinator()
        self.phonon_analyzer = PhononAnalyzer(self.coordinator)
        self.sliding_analyzer = SlidingAnalyzer(self.coordinator)
        
        # Set default output directory
        self.output_dir = os.path.expanduser("~/lammphonon_results")
        self.coordinator.set_config('output_dir', self.output_dir)
        
        # Menu state
        self.running = True
        self.current_menu = "main"
        self.timer = Timer("Menu")
        
        logger.debug("MainMenu initialized")
    
    def clear_screen(self):
        """Clear the terminal screen"""
        if os.name == 'nt':  # For Windows
            os.system('cls')
        else:  # For Linux/Mac
            os.system('clear')
    
    def print_header(self, title="LAMMPhonon 声子分析工具"):
        """Print menu header"""
        self.clear_screen()
        print("=" * 60)
        print(f"{title:^60}")
        print("=" * 60)
        print(f"版本: {self.coordinator.config.get('version', '1.0.0')}")
        print(f"作者: 梁树铭 (Shuming Liang)")
        print(f"邮箱: lsm315@mail.ustc.edu.cn")
        print("-" * 60)
    
    def print_footer(self):
        """Print menu footer"""
        print("-" * 60)
        print("输入选项编号或命令: ")
    
    def show_main_menu(self):
        """Show the main menu"""
        self.print_header()
        
        print("主菜单:")
        print("1. 数据导入与预处理")
        print("2. 声子分析")
        print("3. 滑移分析")
        print("4. 设置")
        print("5. 帮助")
        print("0. 退出")
        
        self.print_footer()
    
    def show_data_menu(self):
        """Show the data import menu"""
        self.print_header("数据导入与预处理")
        
        print("数据菜单:")
        print("1. 导入轨迹文件 (LAMMPS dump)")
        print("2. 导入能量文件")
        print("3. 导入力文件")
        print("4. 导入热流文件")
        print("5. 导入极化文件")
        print("6. 设置质量文件")
        print("7. 设置层定义文件")
        print("8. 显示数据信息")
        print("9. 返回主菜单")
        print("0. 退出")
        
        self.print_footer()
    
    def show_phonon_menu(self):
        """Show the phonon analysis menu"""
        self.print_header("声子分析")
        
        print("声子分析菜单:")
        print("1. 计算速度自相关函数(VACF)")
        print("2. 计算声子态密度(DOS)")
        print("3. 计算投影态密度(PDOS)")
        print("4. 计算方向投影态密度")
        print("5. 计算正则模式")
        print("6. 计算模式占据数")
        print("7. 计算模式寿命")
        print("8. 计算热导率")
        print("9. 返回主菜单")
        print("0. 退出")
        
        self.print_footer()
    
    def show_sliding_menu(self):
        """Show the sliding analysis menu"""
        self.print_header("滑移分析")
        
        print("滑移分析菜单:")
        print("1. 检测材料层")
        print("2. 计算滑移距离")
        print("3. 计算层间距离")
        print("4. 计算摩擦力")
        print("5. 计算摩擦系数")
        print("6. 分析黏滑行为")
        print("7. 计算能量耗散率")
        print("8. 计算堆垛构型参数")
        print("9. 返回主菜单")
        print("0. 退出")
        
        self.print_footer()
    
    def show_settings_menu(self):
        """Show the settings menu"""
        self.print_header("设置")
        
        print("设置菜单:")
        print("1. 设置输出目录")
        print("2. 设置温度")
        print("3. 设置最大频率")
        print("4. 设置频率点数")
        print("5. 设置平滑参数")
        print("6. 设置时间步长")
        print("7. 设置帧数限制")
        print("8. 设置层数")
        print("9. 返回主菜单")
        print("0. 退出")
        
        self.print_footer()
    
    def handle_main_menu(self, choice):
        """Handle main menu selection"""
        if choice == '1':
            self.current_menu = "data"
        elif choice == '2':
            self.current_menu = "phonon"
        elif choice == '3':
            self.current_menu = "sliding"
        elif choice == '4':
            self.current_menu = "settings"
        elif choice == '5':
            self.show_help()
        elif choice == '0':
            self.running = False
        else:
            print("无效选项，请重试")
            time.sleep(1)
    
    def handle_data_menu(self, choice):
        """Handle data menu selection"""
        if choice == '1':
            self.import_trajectory()
        elif choice == '2':
            self.import_energy()
        elif choice == '3':
            self.import_force()
        elif choice == '4':
            self.import_heatflux()
        elif choice == '5':
            self.import_polarization()
        elif choice == '6':
            self.set_mass_file()
        elif choice == '7':
            self.set_layer_file()
        elif choice == '8':
            self.show_data_info()
        elif choice == '9':
            self.current_menu = "main"
        elif choice == '0':
            self.running = False
        else:
            print("无效选项，请重试")
            time.sleep(1)
    
    def handle_phonon_menu(self, choice):
        """Handle phonon menu selection"""
        if choice == '1':
            self.calculate_vacf()
        elif choice == '2':
            self.calculate_dos()
        elif choice == '3':
            self.calculate_pdos()
        elif choice == '4':
            self.calculate_directional_dos()
        elif choice == '5':
            self.calculate_normal_modes()
        elif choice == '6':
            self.calculate_mode_occupation()
        elif choice == '7':
            self.calculate_mode_lifetime()
        elif choice == '8':
            self.calculate_thermal_conductivity()
        elif choice == '9':
            self.current_menu = "main"
        elif choice == '0':
            self.running = False
        else:
            print("无效选项，请重试")
            time.sleep(1)
    
    def handle_sliding_menu(self, choice):
        """Handle sliding menu selection"""
        if choice == '1':
            self.detect_layers()
        elif choice == '2':
            self.calculate_sliding_distance()
        elif choice == '3':
            self.calculate_interlayer_distance()
        elif choice == '4':
            self.calculate_friction_force()
        elif choice == '5':
            self.calculate_friction_coefficient()
        elif choice == '6':
            self.analyze_stick_slip()
        elif choice == '7':
            self.calculate_energy_dissipation()
        elif choice == '8':
            self.calculate_stacking_order()
        elif choice == '9':
            self.current_menu = "main"
        elif choice == '0':
            self.running = False
        else:
            print("无效选项，请重试")
            time.sleep(1)
    
    def handle_settings_menu(self, choice):
        """Handle settings menu selection"""
        if choice == '1':
            self.set_output_dir()
        elif choice == '2':
            self.set_temperature()
        elif choice == '3':
            self.set_freq_max()
        elif choice == '4':
            self.set_freq_points()
        elif choice == '5':
            self.set_sigma()
        elif choice == '6':
            self.set_timestep()
        elif choice == '7':
            self.set_max_frames()
        elif choice == '8':
            self.set_n_layers()
        elif choice == '9':
            self.current_menu = "main"
        elif choice == '0':
            self.running = False
        else:
            print("无效选项，请重试")
            time.sleep(1)
    
    def show_help(self):
        """Show help information"""
        self.print_header("帮助信息")
        
        print("LAMMPhonon 是一个用于LAMMPS分子动力学模拟的声子分析工具。")
        print("它提供以下功能：")
        print("- 分析LAMMPS轨迹文件中的声子属性")
        print("- 计算声子态密度(DOS)和投影态密度(PDOS)")
        print("- 分析材料滑移过程中的摩擦特性")
        print("- 可视化声子和滑移分析结果")
        print()
        print("使用建议：")
        print("1. 首先在'数据导入'菜单中导入需要分析的文件")
        print("2. 然后在'声子分析'或'滑移分析'菜单中进行相应的分析")
        print("3. 分析结果将保存在输出目录中")
        print()
        print("按任意键返回主菜单...")
        input()
        self.current_menu = "main"
    
    # 数据导入与预处理方法
    def import_trajectory(self):
        """Import trajectory file"""
        self.print_header("导入轨迹文件")
        
        print("请输入轨迹文件路径 (LAMMPS dump/lammpstrj格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在读取轨迹文件...")
        
        # 设置最大帧数和跳过帧数
        max_frames = self.coordinator.config.get('max_frames', 0)
        skip_frames = self.coordinator.config.get('skip_frames', 0)
        
        print(f"最大帧数: {max_frames if max_frames > 0 else '全部'}")
        print(f"跳过帧数: {skip_frames}")
        
        # 开始计时
        self.timer.start()
        
        # 读取轨迹文件
        success = self.coordinator.read_trajectory(file_path, max_frames, skip_frames)
        
        # 停止计时
        self.timer.stop()
        
        if success:
            print(f"成功导入轨迹文件: {file_path}")
            print(f"帧数: {self.coordinator.trajectory_data['n_frames']}")
            print(f"原子数: {self.coordinator.trajectory_data['n_atoms']}")
            print(f"用时: {self.timer.elapsed_str()}")
        else:
            print(f"导入轨迹文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def import_energy(self):
        """Import energy file"""
        self.print_header("导入能量文件")
        
        print("请输入能量文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在读取能量文件...")
        
        # 开始计时
        self.timer.start()
        
        # 读取能量文件
        success = self.coordinator.read_energy_data(file_path)
        
        # 停止计时
        self.timer.stop()
        
        if success:
            print(f"成功导入能量文件: {file_path}")
            print(f"数据点数: {len(self.coordinator.energy_data.get('time', []))}")
            print(f"用时: {self.timer.elapsed_str()}")
        else:
            print(f"导入能量文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def import_force(self):
        """Import force file"""
        self.print_header("导入力文件")
        
        print("请输入力文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在读取力文件...")
        
        # 开始计时
        self.timer.start()
        
        # 设置力文件
        success = self.coordinator.set_input_file('force', file_path)
        
        # 停止计时
        self.timer.stop()
        
        if success:
            print(f"成功设置力文件: {file_path}")
            print(f"用时: {self.timer.elapsed_str()}")
        else:
            print(f"设置力文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def import_heatflux(self):
        """Import heat flux file"""
        self.print_header("导入热流文件")
        
        print("请输入热流文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在读取热流文件...")
        
        # 开始计时
        self.timer.start()
        
        # 读取热流文件
        success = self.coordinator.read_heatflux_data(file_path)
        
        # 停止计时
        self.timer.stop()
        
        if success:
            print(f"成功导入热流文件: {file_path}")
            print(f"数据点数: {len(self.coordinator.heatflux_data.get('time', []))}")
            print(f"用时: {self.timer.elapsed_str()}")
        else:
            print(f"导入热流文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def import_polarization(self):
        """Import polarization file"""
        self.print_header("导入极化文件")
        
        print("请输入极化文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在读取极化文件...")
        
        # 开始计时
        self.timer.start()
        
        # 读取极化文件
        success = self.coordinator.read_polarization_data(file_path)
        
        # 停止计时
        self.timer.stop()
        
        if success:
            print(f"成功导入极化文件: {file_path}")
            print(f"数据点数: {len(self.coordinator.polarization_data.get('time', []))}")
            print(f"用时: {self.timer.elapsed_str()}")
        else:
            print(f"导入极化文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def set_mass_file(self):
        """Set mass file"""
        self.print_header("设置质量文件")
        
        print("请输入质量文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在设置质量文件...")
        
        # 设置质量文件
        success = self.coordinator.set_input_file('masses', file_path)
        
        if success:
            print(f"成功设置质量文件: {file_path}")
        else:
            print(f"设置质量文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def set_layer_file(self):
        """Set layer definition file"""
        self.print_header("设置层定义文件")
        
        print("请输入层定义文件路径 (文本格式):")
        file_path = input().strip()
        
        if not file_path:
            print("未提供文件路径，操作取消")
            time.sleep(1)
            return
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            time.sleep(1)
            return
        
        print("正在设置层定义文件...")
        
        # 设置层定义文件
        success = self.coordinator.set_input_file('layer_definition', file_path)
        
        if success:
            print(f"成功设置层定义文件: {file_path}")
        else:
            print(f"设置层定义文件失败: {file_path}")
        
        print("按任意键继续...")
        input()
    
    def show_data_info(self):
        """Show data information"""
        self.print_header("数据信息")
        
        # 轨迹文件信息
        traj_file = self.coordinator.input_files.get('trajectory')
        if traj_file and self.coordinator.trajectory_data:
            print("轨迹文件信息:")
            print(f"文件路径: {traj_file}")
            print(f"帧数: {self.coordinator.trajectory_data['n_frames']}")
            print(f"原子数: {self.coordinator.trajectory_data['n_atoms']}")
            print()
        else:
            print("未导入轨迹文件")
            print()
        
        # 能量文件信息
        energy_file = self.coordinator.input_files.get('energy')
        if energy_file and self.coordinator.energy_data:
            print("能量文件信息:")
            print(f"文件路径: {energy_file}")
            print(f"数据点数: {len(self.coordinator.energy_data.get('time', []))}")
            print()
        else:
            print("未导入能量文件")
            print()
        
        # 热流文件信息
        heatflux_file = self.coordinator.input_files.get('heatflux')
        if heatflux_file and self.coordinator.heatflux_data:
            print("热流文件信息:")
            print(f"文件路径: {heatflux_file}")
            print(f"数据点数: {len(self.coordinator.heatflux_data.get('time', []))}")
            print()
        else:
            print("未导入热流文件")
            print()
        
        # 极化文件信息
        polarization_file = self.coordinator.input_files.get('polarization')
        if polarization_file and self.coordinator.polarization_data:
            print("极化文件信息:")
            print(f"文件路径: {polarization_file}")
            print(f"数据点数: {len(self.coordinator.polarization_data.get('time', []))}")
            print()
        else:
            print("未导入极化文件")
            print()
        
        # 其他输入文件信息
        for file_type in ['force', 'masses', 'layer_definition']:
            file_path = self.coordinator.input_files.get(file_type)
            if file_path:
                print(f"{file_type}文件: {file_path}")
        
        print()
        print("按任意键继续...")
        input()
    
    def set_output_dir(self):
        """Set output directory"""
        self.print_header("设置输出目录")
        
        print(f"当前输出目录: {self.coordinator.output_dir}")
        print("请输入新的输出目录路径:")
        dir_path = input().strip()
        
        if not dir_path:
            print("未提供目录路径，操作取消")
            time.sleep(1)
            return
        
        # 展开用户路径 (如 ~)
        dir_path = os.path.expanduser(dir_path)
        
        # 检查并创建目录
        try:
            os.makedirs(dir_path, exist_ok=True)
            self.coordinator.set_config('output_dir', dir_path)
            self.coordinator.output_dir = dir_path
            print(f"输出目录已设置为: {dir_path}")
        except Exception as e:
            print(f"创建目录失败: {e}")
        
        print("按任意键继续...")
        input()
    
    def set_temperature(self):
        """Set temperature"""
        self.print_header("设置温度")
        
        current_temp = self.coordinator.config.get('temperature', 300.0)
        print(f"当前温度: {current_temp} K")
        print("请输入新的温度值 (K):")
        
        try:
            temp = float(input().strip())
            if temp <= 0:
                print("温度必须大于零")
            else:
                self.coordinator.set_config('temperature', temp)
                self.phonon_analyzer.temperature = temp
                print(f"温度已设置为: {temp} K")
        except ValueError:
            print("无效的温度值")
        
        print("按任意键继续...")
        input()
    
    def set_freq_max(self):
        """Set maximum frequency"""
        self.print_header("设置最大频率")
        
        current_freq = self.coordinator.config.get('freq_max', 30.0)
        print(f"当前最大频率: {current_freq} THz")
        print("请输入新的最大频率值 (THz):")
        
        try:
            freq = float(input().strip())
            if freq <= 0:
                print("频率必须大于零")
            else:
                self.coordinator.set_config('freq_max', freq)
                self.phonon_analyzer.freq_max = freq
                print(f"最大频率已设置为: {freq} THz")
        except ValueError:
            print("无效的频率值")
        
        print("按任意键继续...")
        input()
    
    def set_freq_points(self):
        """Set frequency points"""
        self.print_header("设置频率点数")
        
        current_points = self.coordinator.config.get('freq_points', 1000)
        print(f"当前频率点数: {current_points}")
        print("请输入新的频率点数:")
        
        try:
            points = int(input().strip())
            if points <= 0:
                print("点数必须大于零")
            else:
                self.coordinator.set_config('freq_points', points)
                self.phonon_analyzer.freq_points = points
                print(f"频率点数已设置为: {points}")
        except ValueError:
            print("无效的点数值")
        
        print("按任意键继续...")
        input()
    
    def set_sigma(self):
        """Set smoothing parameter"""
        self.print_header("设置平滑参数")
        
        current_sigma = self.coordinator.config.get('sigma', 0.1)
        print(f"当前平滑参数: {current_sigma} THz")
        print("请输入新的平滑参数值 (THz):")
        
        try:
            sigma = float(input().strip())
            if sigma < 0:
                print("平滑参数不能为负值")
            else:
                self.coordinator.set_config('sigma', sigma)
                self.phonon_analyzer.sigma = sigma
                print(f"平滑参数已设置为: {sigma} THz")
        except ValueError:
            print("无效的参数值")
        
        print("按任意键继续...")
        input()
    
    def set_timestep(self):
        """Set time step"""
        self.print_header("设置时间步长")
        
        current_timestep = self.coordinator.config.get('timestep', 0.001)
        print(f"当前时间步长: {current_timestep} ps")
        print("请输入新的时间步长值 (ps):")
        
        try:
            timestep = float(input().strip())
            if timestep <= 0:
                print("时间步长必须大于零")
            else:
                self.coordinator.set_config('timestep', timestep)
                self.phonon_analyzer.timestep = timestep
                print(f"时间步长已设置为: {timestep} ps")
        except ValueError:
            print("无效的时间步长值")
        
        print("按任意键继续...")
        input()
    
    def set_max_frames(self):
        """Set maximum frames"""
        self.print_header("设置帧数限制")
        
        current_frames = self.coordinator.config.get('max_frames', 0)
        print(f"当前帧数限制: {current_frames if current_frames > 0 else '全部'}")
        print("请输入新的帧数限制 (0表示全部):")
        
        try:
            frames = int(input().strip())
            if frames < 0:
                print("帧数不能为负值")
            else:
                self.coordinator.set_config('max_frames', frames)
                print(f"帧数限制已设置为: {frames if frames > 0 else '全部'}")
        except ValueError:
            print("无效的帧数值")
        
        print("按任意键继续...")
        input()
    
    def set_n_layers(self):
        """Set number of layers"""
        self.print_header("设置层数")
        
        current_layers = self.coordinator.config.get('n_layers', 2)
        print(f"当前层数: {current_layers}")
        print("请输入新的层数:")
        
        try:
            layers = int(input().strip())
            if layers <= 0:
                print("层数必须大于零")
            else:
                self.coordinator.set_config('n_layers', layers)
                self.sliding_analyzer.n_layers = layers
                print(f"层数已设置为: {layers}")
        except ValueError:
            print("无效的层数值")
        
        print("按任意键继续...")
        input()
    
    def calculate_vacf(self):
        """Calculate velocity autocorrelation function"""
        self.print_header("计算速度自相关函数(VACF)")
        
        # 检查轨迹数据
        if not self.coordinator.trajectory_data or 'velocities' not in self.coordinator.trajectory_data:
            print("错误: 未找到速度数据。请先导入轨迹文件。")
            print("按任意键继续...")
            input()
            return
        
        print("计算速度自相关函数(VACF)...")
        print(f"帧数: {self.coordinator.trajectory_data['n_frames']}")
        print(f"原子数: {self.coordinator.trajectory_data['n_atoms']}")
        
        # 获取设置
        max_lag = int(self.coordinator.trajectory_data['n_frames'] / 2)
        print(f"最大时滞: {max_lag} 帧")
        
        # 开始计时
        self.timer.start()
        
        # 计算VACF
        try:
            vacf = self.phonon_analyzer.calculate_velocity_autocorrelation(
                self.coordinator.trajectory_data['velocities'],
                max_lag=max_lag,
                normalize=True
            )
            
            # 停止计时
            self.timer.stop()
            
            print(f"VACF计算完成，时间: {self.timer.elapsed_str()}")
            print(f"VACF长度: {len(vacf)}")
            
            # 保存结果
            self.coordinator.save_results(
                'vacf', vacf, 
                'Velocity Autocorrelation Function'
            )
            
            # 计算结果文件路径
            output_file = os.path.join(
                self.coordinator.results_dir, 
                f"vacf_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            # 创建时间数组
            timestep = self.coordinator.config.get('timestep', 0.001)
            time_data = np.arange(len(vacf)) * timestep
            
            # 保存为文本文件
            np.savetxt(
                output_file, 
                np.column_stack((time_data, vacf)), 
                header="Time(ps)  VACF", 
                fmt='%.6f'
            )
            
            print(f"VACF已保存至: {output_file}")
        except Exception as e:
            print(f"计算VACF时出错: {str(e)}")
        
        print("按任意键继续...")
        input()
    
    def calculate_dos(self):
        """Calculate phonon density of states"""
        self.print_header("计算声子态密度(DOS)")
        
        # 检查VACF数据
        if self.phonon_analyzer.vacf is None:
            print("错误: 未找到VACF数据。请先计算速度自相关函数。")
            print("按任意键继续...")
            input()
            return
        
        print("计算声子态密度(DOS)...")
        
        # 获取设置
        freq_max = self.coordinator.config.get('freq_max', 30.0)
        freq_points = self.coordinator.config.get('freq_points', 1000)
        sigma = self.coordinator.config.get('sigma', 0.1)
        timestep = self.coordinator.config.get('timestep', 0.001)
        
        print(f"最大频率: {freq_max} THz")
        print(f"频率点数: {freq_points}")
        print(f"平滑参数: {sigma} THz")
        print(f"时间步长: {timestep} ps")
        
        # 开始计时
        self.timer.start()
        
        # 计算DOS
        try:
            freqs, dos = self.phonon_analyzer.calculate_dos(
                vacf=self.phonon_analyzer.vacf,
                timestep=timestep,
                freq_max=freq_max,
                freq_points=freq_points,
                sigma=sigma
            )
            
            # 停止计时
            self.timer.stop()
            
            print(f"DOS计算完成，时间: {self.timer.elapsed_str()}")
            
            # 保存结果
            self.coordinator.save_results(
                'dos', {'frequencies': freqs, 'dos': dos}, 
                'Phonon Density of States'
            )
            
            # 计算结果文件路径
            output_file = os.path.join(
                self.coordinator.results_dir, 
                f"dos_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            # 保存为文本文件
            np.savetxt(
                output_file, 
                np.column_stack((freqs, dos)), 
                header="Frequency(THz)  DOS", 
                fmt='%.6f'
            )
            
            print(f"DOS已保存至: {output_file}")
        except Exception as e:
            print(f"计算DOS时出错: {str(e)}")
        
        print("按任意键继续...")
        input()
    
    def detect_layers(self):
        """Detect material layers"""
        self.print_header("检测材料层")
        
        # 检查轨迹数据
        if not self.coordinator.trajectory_data or 'positions' not in self.coordinator.trajectory_data:
            print("错误: 未找到位置数据。请先导入轨迹文件。")
            print("按任意键继续...")
            input()
            return
        
        print("检测材料层...")
        
        # 获取设置
        n_layers = self.coordinator.config.get('n_layers', 2)
        method = self.coordinator.config.get('layer_detection_method', 'kmeans')
        layer_direction = self.coordinator.config.get('layer_direction', 'z')
        
        print(f"层数: {n_layers}")
        print(f"检测方法: {method}")
        print(f"层方向: {layer_direction}")
        
        # 获取第一帧的位置数据
        positions = self.coordinator.trajectory_data['positions'][0]
        
        # 开始计时
        self.timer.start()
        
        # 检测层
        try:
            layer_indices = self.sliding_analyzer.detect_material_layers(
                positions,
                method=method,
                n_layers=n_layers,
                layer_direction=layer_direction
            )
            
            # 停止计时
            self.timer.stop()
            
            print(f"层检测完成，时间: {self.timer.elapsed_str()}")
            
            # 打印每层的原子数
            for layer, indices in layer_indices.items():
                print(f"层 {layer+1}: {len(indices)} 个原子")
            
            # 保存结果
            self.coordinator.save_results(
                'layer_indices', layer_indices, 
                'Material Layer Indices'
            )
        except Exception as e:
            print(f"检测层时出错: {str(e)}")
        
        print("按任意键继续...")
        input()
    
    def calculate_sliding_distance(self):
        """Calculate sliding distance"""
        self.print_header("计算滑移距离")
        
        # 检查轨迹数据
        if not self.coordinator.trajectory_data or 'positions' not in self.coordinator.trajectory_data:
            print("错误: 未找到位置数据。请先导入轨迹文件。")
            print("按任意键继续...")
            input()
            return
        
        # 检查层索引
        if self.sliding_analyzer.layer_indices is None:
            print("未检测到层索引。正在自动检测...")
            # 获取第一帧的位置数据
            positions = self.coordinator.trajectory_data['positions'][0]
            
            # 获取设置
            n_layers = self.coordinator.config.get('n_layers', 2)
            method = self.coordinator.config.get('layer_detection_method', 'kmeans')
            layer_direction = self.coordinator.config.get('layer_direction', 'z')
            
            # 检测层
            try:
                self.sliding_analyzer.detect_material_layers(
                    positions,
                    method=method,
                    n_layers=n_layers,
                    layer_direction=layer_direction
                )
                print("层检测完成")
            except Exception as e:
                print(f"检测层时出错: {str(e)}")
                print("按任意键继续...")
                input()
                return
        
        print("计算滑移距离...")
        
        # 获取设置
        sliding_direction = self.coordinator.config.get('sliding_direction', 'x')
        
        print(f"滑移方向: {sliding_direction}")
        print(f"帧数: {self.coordinator.trajectory_data['n_frames']}")
        
        # 开始计时
        self.timer.start()
        
        # 计算滑移距离
        try:
            sliding_distance = self.sliding_analyzer.calculate_sliding_distance(
                self.coordinator.trajectory_data['positions'],
                sliding_direction=sliding_direction
            )
            
            # 停止计时
            self.timer.stop()
            
            print(f"滑移距离计算完成，时间: {self.timer.elapsed_str()}")
            print(f"最大滑移距离: {np.max(np.abs(sliding_distance)):.4f} Å")
            print(f"最终滑移距离: {sliding_distance[-1]:.4f} Å")
            
            # 保存结果
            self.coordinator.save_results(
                'sliding_distance', sliding_distance, 
                'Sliding Distance'
            )
            
            # 计算结果文件路径
            output_file = os.path.join(
                self.coordinator.results_dir, 
                f"sliding_distance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            # 创建时间数组
            timestep = self.coordinator.config.get('timestep', 0.001)
            dump_interval = self.coordinator.config.get('dump_interval', 1)
            time_data = np.arange(len(sliding_distance)) * timestep * dump_interval
            
            # 保存为文本文件
            np.savetxt(
                output_file, 
                np.column_stack((time_data, sliding_distance)), 
                header="Time(ps)  Distance(Å)", 
                fmt='%.6f'
            )
            
            print(f"滑移距离已保存至: {output_file}")
        except Exception as e:
            print(f"计算滑移距离时出错: {str(e)}")
        
        print("按任意键继续...")
        input()
    
    def run(self):
        """Run the main menu loop"""
        self.running = True
        
        while self.running:
            # Show current menu
            if self.current_menu == "main":
                self.show_main_menu()
            elif self.current_menu == "data":
                self.show_data_menu()
            elif self.current_menu == "phonon":
                self.show_phonon_menu()
            elif self.current_menu == "sliding":
                self.show_sliding_menu()
            elif self.current_menu == "settings":
                self.show_settings_menu()
            
            # Get user input
            choice = input().strip()
            
            # Handle menu selection
            if self.current_menu == "main":
                self.handle_main_menu(choice)
            elif self.current_menu == "data":
                self.handle_data_menu(choice)
            elif self.current_menu == "phonon":
                self.handle_phonon_menu(choice)
            elif self.current_menu == "sliding":
                self.handle_sliding_menu(choice)
            elif self.current_menu == "settings":
                self.handle_settings_menu(choice)
        
        print("感谢使用LAMMPhonon！")
        print("再见！")
        time.sleep(1)
        self.clear_screen()


def main():
    """Main function to run the menu"""
    # Setup logging
    log_file = f"lammphonon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run menu
    menu = MainMenu()
    try:
        menu.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        logging.exception("程序异常")
    
    print("程序已退出")


if __name__ == "__main__":
    main() 