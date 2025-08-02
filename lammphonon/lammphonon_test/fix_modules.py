#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon模块修复脚本
====================

修复LAMMPhonon的关键功能模块问题。

作者: 梁树铭 (Shuming Liang)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import inspect

# 导入LAMMPhonon模块
from lammphonon import (
    PhononCoordinator, 
    PhononAnalyzer, 
    ThermalAnalyzer,
    AnharmonicAnalyzer,
    TemporalAnalyzer,
    EquilibrationAnalyzer
)

# 设置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/fixes")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def patch_thermal_analyzer():
    """修复ThermalAnalyzer中的问题"""
    print("修复ThermalAnalyzer中的问题...")
    
    # 获取ThermalAnalyzer类定义所在文件
    file_path = inspect.getfile(ThermalAnalyzer)
    print(f"ThermalAnalyzer定义在: {file_path}")
    
    try:
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查calculate_heatflux_autocorrelation方法中的问题
        # 主要问题是热流向量形状不匹配
        if "calculate_heatflux_autocorrelation" in content:
            print("找到calculate_heatflux_autocorrelation方法")
            
            # 在方法中添加形状检查和适配代码
            lines = content.split('\n')
            method_start = -1
            for i, line in enumerate(lines):
                if "def calculate_heatflux_autocorrelation" in line:
                    method_start = i
                    break
                    
            if method_start >= 0:
                # 找到方法体开始的位置
                for i in range(method_start + 1, len(lines)):
                    if "self.profiler.start" in lines[i]:
                        # 在方法开始处添加形状检查和调整代码
                        shape_check_code = """
        # Ensure heatflux data has correct shape [frames, 3]
        if heatflux_data is None:
            logger.error("No heat flux data provided")
            self.profiler.stop()
            return None
            
        # Check shape and convert if necessary
        if len(heatflux_data.shape) != 2:
            logger.warning(f"Heat flux data has unexpected shape: {heatflux_data.shape}")
            # Try to reshape if possible
            if len(heatflux_data.shape) == 1:
                # Single vector, reshape to [1, 3]
                if len(heatflux_data) >= 3:
                    heatflux_data = heatflux_data[:3].reshape(1, 3)
                    logger.info(f"Reshaped heat flux data to: {heatflux_data.shape}")
                else:
                    logger.error("Heat flux data should have shape [frames, 3], got {heatflux_data.shape}")
                    self.profiler.stop()
                    return None
            else:
                logger.error(f"Cannot reshape heat flux data with shape {heatflux_data.shape}")
                self.profiler.stop()
                return None
                
        if heatflux_data.shape[1] != 3:
            logger.error(f"Heat flux data should have shape [frames, 3], got {heatflux_data.shape}")
            # Try to extract 3 components if possible
            if heatflux_data.shape[1] > 3:
                heatflux_data = heatflux_data[:, :3]
                logger.info(f"Using first 3 columns of heat flux data, new shape: {heatflux_data.shape}")
            else:
                self.profiler.stop()
                return None"""
                        
                        # 插入代码到方法开始处
                        lines.insert(i, shape_check_code)
                        content = '\n'.join(lines)
                        
                        # 写回文件
                        with open(file_path, 'w') as f:
                            f.write(content)
                        
                        print("已修复ThermalAnalyzer.calculate_heatflux_autocorrelation方法中的形状检查问题")
                        break
            else:
                print("未找到calculate_heatflux_autocorrelation方法的开始位置")
    except Exception as e:
        print(f"修复ThermalAnalyzer时出错: {str(e)}")

def patch_temporal_analyzer():
    """修复TemporalAnalyzer中的问题"""
    print("修复TemporalAnalyzer中的问题...")
    
    # 获取TemporalAnalyzer类定义所在文件
    file_path = inspect.getfile(TemporalAnalyzer)
    print(f"TemporalAnalyzer定义在: {file_path}")
    
    try:
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查calculate_time_resolved_dos方法中的问题
        # 主要问题是窗口大小和数据帧数不匹配
        if "calculate_time_resolved_dos" in content:
            print("找到calculate_time_resolved_dos方法")
            
            # 在方法中添加帧数检查和窗口调整代码
            lines = content.split('\n')
            method_start = -1
            found_window_calc = False
            for i, line in enumerate(lines):
                if "def calculate_time_resolved_dos" in line:
                    method_start = i
                
                if method_start >= 0 and "n_windows = " in line:
                    # 在计算窗口数之前添加窗口尺寸调整代码
                    window_adjust_code = """
        # Adjust window size if necessary
        if window_size > n_frames:
            logger.warning(f"Window size ({window_size}) larger than available frames ({n_frames})")
            logger.warning(f"Adjusting window size to {n_frames}")
            window_size = n_frames
            
        # Ensure we have at least one window
        if window_size < 1:
            window_size = 1
            logger.warning("Window size adjusted to 1")
            
        # Ensure window step is valid
        if window_step < 1:
            window_step = 1
            logger.warning("Window step adjusted to 1")
        
        # Minimum window size for meaningful VACF
        if window_size < 2:
            logger.warning("Window size too small for meaningful VACF, results may be unreliable")
            """
                    
                    # 插入代码到指定位置
                    lines.insert(i, window_adjust_code)
                    content = '\n'.join(lines)
                    
                    # 写回文件
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    print("已修复TemporalAnalyzer.calculate_time_resolved_dos方法中的窗口尺寸调整问题")
                    found_window_calc = True
                    break
            
            if not found_window_calc:
                print("未找到calculate_time_resolved_dos方法中的窗口计算部分")
    except Exception as e:
        print(f"修复TemporalAnalyzer时出错: {str(e)}")
        
def patch_equilibration_analyzer():
    """修复EquilibrationAnalyzer中的问题"""
    print("修复EquilibrationAnalyzer中的问题...")
    
    # 获取EquilibrationAnalyzer类定义所在文件
    file_path = inspect.getfile(EquilibrationAnalyzer)
    print(f"EquilibrationAnalyzer定义在: {file_path}")
    
    try:
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查calculate_system_equilibration方法中的问题
        # 主要问题是数据点不足导致拟合失败
        if "calculate_system_equilibration" in content:
            print("找到calculate_system_equilibration方法")
            
            # 在方法中添加数据点检查和处理代码
            lines = content.split('\n')
            method_start = -1
            found_fitting_code = False
            for i, line in enumerate(lines):
                if "def calculate_system_equilibration" in line:
                    method_start = i
                
                if method_start >= 0 and "try:" in line and i+5 < len(lines) and "curve_fit" in lines[i+5]:
                    # 在拟合曲线之前添加数据点检查代码
                    data_check_code = """
            # Check if we have enough data points for fitting
            if len(energy_values) < 4:
                logger.warning(f"Not enough data points for curve fitting (need at least 4, got {len(energy_values)})")
                # Return a simple estimate if we can't do proper fitting
                if len(energy_values) > 1:
                    # Simple decay rate estimate
                    E_start = energy_values[0]
                    E_end = energy_values[-1]
                    decay_fraction = abs(E_end - E_start) / (abs(E_start) if abs(E_start) > 1e-10 else 1.0)
                    
                    # If significant decay, estimate equilibration at 95% of change
                    if decay_fraction > 0.01:  # 1% change threshold
                        t_start = time_points[0]
                        t_end = time_points[-1]
                        # Estimate equilibration time as 95% of time range
                        equilibration_time = t_start + 0.95 * (t_end - t_start)
                        # Simplified parameters [A, tau, E_inf]
                        params = [E_start - E_end, (t_end - t_start) / 3.0, E_end]
                        logger.warning(f"Using simplified estimate: equilibration time = {equilibration_time:.3f}")
                        return equilibration_time, params
                
                logger.error("System equilibration curve fitting failed: not enough data points")
                return None, None"""
                    
                    # 插入代码到拟合操作之前
                    lines.insert(i+1, data_check_code)
                    content = '\n'.join(lines)
                    
                    # 写回文件
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    print("已修复EquilibrationAnalyzer.calculate_system_equilibration方法中的数据点检查问题")
                    found_fitting_code = True
                    break
                    
            if not found_fitting_code:
                print("未找到calculate_system_equilibration方法中的拟合代码部分")
    except Exception as e:
        print(f"修复EquilibrationAnalyzer时出错: {str(e)}")

def add_missing_methods():
    """添加缺失的方法"""
    print("添加缺失的方法...")
    
    # 添加analyze_energy_equilibration方法到EquilibrationAnalyzer
    if not hasattr(EquilibrationAnalyzer, 'analyze_energy_equilibration'):
        print("添加EquilibrationAnalyzer.analyze_energy_equilibration方法")
        
        file_path = inspect.getfile(EquilibrationAnalyzer)
        
        try:
            # 读取文件内容
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 找到类定义的末尾
            lines = content.split('\n')
            last_method_end = 0
            for i in range(len(lines)-1, 0, -1):
                if lines[i].strip() == '':
                    continue
                if lines[i].startswith('    def '):
                    # 找到最后一个方法定义
                    for j in range(i+1, len(lines)):
                        if j+1 >= len(lines) or (lines[j].strip() == '' and lines[j+1].strip() != '' and not lines[j+1].startswith(' ')):
                            last_method_end = j
                            break
                    break
            
            if last_method_end > 0:
                # 添加新方法
                new_method = """
    def analyze_energy_equilibration(self, energy_values, window_size=10):
        \"\"\"
        分析能量平衡过程，返回平衡帧索引和衡量指标
        
        参数:
            energy_values: 能量数据
            window_size: 计算均值和方差的窗口大小
            
        返回:
            equilibration_frame: 平衡帧索引
            metric: 衡量指标
        \"\"\"
        self.profiler.start("analyze_energy_equilibration")
        
        # 检查数据
        if energy_values is None or len(energy_values) < window_size + 1:
            logger.warning(f"数据点不足，无法分析能量平衡")
            self.profiler.stop()
            return 0, 0.0
        
        # 计算移动平均和方差
        n_frames = len(energy_values)
        means = np.zeros(n_frames - window_size + 1)
        vars = np.zeros(n_frames - window_size + 1)
        
        for i in range(len(means)):
            window = energy_values[i:i+window_size]
            means[i] = np.mean(window)
            vars[i] = np.var(window)
        
        # 方差稳定的点视为平衡点
        # 简化：取方差下降到最大方差10%以下的点
        threshold = 0.1 * np.max(vars)
        stable_indices = np.where(vars < threshold)[0]
        
        if len(stable_indices) > 0:
            equilibration_frame = stable_indices[0] + window_size // 2
        else:
            # 如果找不到稳定点，取中点
            equilibration_frame = n_frames // 2
        
        # 衡量指标：方差/均值的比率
        metric = vars[min(equilibration_frame, len(vars)-1)] / abs(means[min(equilibration_frame, len(means)-1)]) if abs(means[min(equilibration_frame, len(means)-1)]) > 1e-10 else 0.0
        
        self.profiler.stop()
        return equilibration_frame, metric"""
                
                # 插入新方法到类定义末尾
                lines.insert(last_method_end + 1, new_method)
                content = '\n'.join(lines)
                
                # 写回文件
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print("已添加EquilibrationAnalyzer.analyze_energy_equilibration方法")
            else:
                print("未找到合适的位置添加新方法")
        except Exception as e:
            print(f"添加EquilibrationAnalyzer.analyze_energy_equilibration方法时出错: {str(e)}")
    else:
        print("EquilibrationAnalyzer.analyze_energy_equilibration方法已存在")

def apply_patches():
    """应用所有修复补丁"""
    print("开始应用修复补丁...")
    
    # 1. 修复ThermalAnalyzer
    patch_thermal_analyzer()
    
    # 2. 修复TemporalAnalyzer
    patch_temporal_analyzer()
    
    # 3. 修复EquilibrationAnalyzer
    patch_equilibration_analyzer()
    
    # 4. 添加缺失的方法
    add_missing_methods()
    
    print("所有修复补丁应用完成!")

if __name__ == "__main__":
    apply_patches() 