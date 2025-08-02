#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon - 热分析高级示例

本示例展示如何使用LAMMPhonon进行热传导相关分析

作者: 梁树铭 (Shuming Liang)
邮箱: lsm315@mail.ustc.edu.cn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lammphonon import PhononCoordinator, ThermalAnalyzer
from lammphonon.fixer import apply_all_patches

def main():
    """热分析高级示例"""
    print("LAMMPhonon热分析高级示例")
    
    # 创建输出目录
    output_dir = "./thermal_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 创建协调器
    coordinator = PhononCoordinator()
    coordinator.set_config("output_dir", output_dir)
    coordinator.set_config("temperature", 300.0)  # 300K
    
    # 读取轨迹文件
    # 注意: 请修改为您实际的轨迹文件路径
    trajectory_file = "../data/dump.phonon"
    print(f"读取轨迹文件: {trajectory_file}")
    coordinator.read_trajectory(trajectory_file)
    
    # 读取热流数据
    # 注意: 请修改为您实际的热流文件路径
    heatflux_file = "../data/heatflux.dat"
    print(f"读取热流文件: {heatflux_file}")
    try:
        coordinator.read_heatflux_data(heatflux_file)
    except Exception as e:
        print(f"读取热流文件出错: {str(e)}")
        print("将使用模拟数据进行演示")
        # 创建模拟热流数据
        n_frames = len(coordinator.trajectory_data['positions'])
        heatflux_data = np.random.rand(n_frames, 3) * 10.0  # 模拟三个方向的热流
        coordinator.heatflux_data = heatflux_data
    
    # 应用补丁修复潜在问题
    print("应用热分析修复补丁...")
    apply_all_patches({'ThermalAnalyzer': ThermalAnalyzer})
    
    # 创建热分析器
    thermal_analyzer = ThermalAnalyzer(coordinator)
    
    # 计算热流自相关函数(HFACF)
    print("计算热流自相关函数(HFACF)...")
    hfacf = thermal_analyzer.calculate_heatflux_autocorrelation(coordinator.heatflux_data)
    
    # 检查返回值类型
    if isinstance(hfacf, tuple) and len(hfacf) == 2:
        time_lags, hfacf_values = hfacf
    else:
        hfacf_values = hfacf
        time_lags = np.arange(len(hfacf_values)) * coordinator.timestep
    
    # 保存HFACF数据
    hfacf_file = os.path.join(output_dir, "hfacf.txt")
    np.savetxt(hfacf_file, np.column_stack((time_lags, hfacf_values)), 
              header="Time (ps), HFACF")
    print(f"HFACF已保存到: {hfacf_file}")
    
    # 绘制HFACF图
    print("生成HFACF图...")
    plt.figure(figsize=(10, 6))
    plt.plot(time_lags, hfacf_values)
    plt.xlabel('Time (ps)')
    plt.ylabel('Heat Flux ACF')
    plt.title('Heat Flux Autocorrelation Function')
    plt.grid(True)
    
    # 保存图像
    hfacf_plot_file = os.path.join(output_dir, "hfacf.png")
    plt.savefig(hfacf_plot_file, dpi=300)
    plt.close()
    print(f"HFACF图像已保存到: {hfacf_plot_file}")
    
    # 计算热导率
    print("计算热导率...")
    # 获取模拟体系的体积
    box = coordinator.get_box()
    volume = box['volume']
    print(f"模拟体系体积: {volume:.2f} Å³")
    
    kappa = thermal_analyzer.calculate_thermal_conductivity(
        coordinator.heatflux_data,
        temperature=coordinator.temperature,
        volume=volume
    )
    
    # 处理热导率结果
    if isinstance(kappa, tuple) and len(kappa) == 2:
        times, kappa_values = kappa
        
        # 保存热导率数据
        kappa_file = os.path.join(output_dir, "thermal_conductivity.txt")
        np.savetxt(kappa_file, np.column_stack((times, kappa_values)), 
                  header="Time (ps), Thermal Conductivity (W/mK)")
        print(f"热导率数据已保存到: {kappa_file}")
        
        # 绘制热导率图
        print("生成热导率图...")
        plt.figure(figsize=(10, 6))
        plt.plot(times, kappa_values)
        plt.xlabel('Time (ps)')
        plt.ylabel('Thermal Conductivity (W/mK)')
        plt.title('Green-Kubo Thermal Conductivity')
        plt.grid(True)
        
        # 保存图像
        kappa_plot_file = os.path.join(output_dir, "thermal_conductivity.png")
        plt.savefig(kappa_plot_file, dpi=300)
        plt.close()
        print(f"热导率图像已保存到: {kappa_plot_file}")
        
        # 最终热导率（最后一个值）
        final_kappa = kappa_values[-1]
    else:
        final_kappa = kappa
        
        # 保存热导率数据
        kappa_file = os.path.join(output_dir, "thermal_conductivity.txt")
        with open(kappa_file, 'w') as f:
            f.write(f"Thermal Conductivity (W/mK): {final_kappa:.6f}\n")
        print(f"热导率数据已保存到: {kappa_file}")
    
    print(f"热导率: {final_kappa:.4f} W/mK")
    
    # 分析各方向的热导率贡献
    print("分析各方向的热导率贡献...")
    try:
        kappa_xyz = thermal_analyzer.calculate_directional_thermal_conductivity(
            coordinator.heatflux_data,
            temperature=coordinator.temperature,
            volume=volume
        )
        
        # 处理各方向热导率结果
        if isinstance(kappa_xyz, dict):
            # 保存各方向热导率数据
            kappa_dir_file = os.path.join(output_dir, "directional_thermal_conductivity.txt")
            with open(kappa_dir_file, 'w') as f:
                for direction, value in kappa_xyz.items():
                    f.write(f"{direction}-direction: {value:.6f} W/mK\n")
            print(f"各方向热导率已保存到: {kappa_dir_file}")
            
            # 绘制各方向热导率条形图
            print("生成各方向热导率条形图...")
            plt.figure(figsize=(8, 6))
            directions = list(kappa_xyz.keys())
            values = [kappa_xyz[d] for d in directions]
            
            plt.bar(directions, values)
            plt.ylabel('Thermal Conductivity (W/mK)')
            plt.title('Directional Thermal Conductivity')
            
            # 保存图像
            kappa_dir_plot_file = os.path.join(output_dir, "directional_thermal_conductivity.png")
            plt.savefig(kappa_dir_plot_file, dpi=300)
            plt.close()
            print(f"各方向热导率图像已保存到: {kappa_dir_plot_file}")
        
    except Exception as e:
        print(f"计算各方向热导率时出错: {str(e)}")
    
    # 创建热传导分析报告
    print("生成热传导分析报告...")
    report_file = os.path.join(output_dir, "thermal_analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("=============================================\n")
        f.write("           热传导分析报告                    \n")
        f.write("=============================================\n\n")
        f.write(f"分析时间: {thermal_analyzer.get_timestamp()}\n\n")
        f.write("系统参数:\n")
        f.write(f"- 温度: {coordinator.temperature} K\n")
        f.write(f"- 体积: {volume:.2f} Å³\n")
        f.write(f"- 轨迹帧数: {len(coordinator.trajectory_data['positions'])}\n\n")
        f.write("热导率结果:\n")
        f.write(f"- 总热导率: {final_kappa:.4f} W/mK\n")
        if 'kappa_xyz' in locals():
            f.write("- 各方向热导率:\n")
            for direction, value in kappa_xyz.items():
                f.write(f"  * {direction}-方向: {value:.4f} W/mK\n")
        f.write("\n备注: 以上结果基于格林-库博方法计算\n")
    
    print(f"热传导分析报告已保存到: {report_file}")
    print("分析完成!")

if __name__ == "__main__":
    main() 