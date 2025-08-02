#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon修复和测试集成程序
=========================

整合所有修复和测试功能，提供统一的命令行界面。

作者: 梁树铭 (Shuming Liang)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 导入LAMMPhonon模块
    from lammphonon import (
        PhononCoordinator, 
        PhononAnalyzer, 
        ThermalAnalyzer,
        AnharmonicAnalyzer,
        TemporalAnalyzer,
        EquilibrationAnalyzer
    )
except ImportError:
    logger.error("无法导入LAMMPhonon模块，请确保已正确安装")
    sys.exit(1)

# 创建lammphonon_fixer目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXER_DIR = os.path.join(SCRIPT_DIR, "lammphonon_fixer")
Path(FIXER_DIR).mkdir(parents=True, exist_ok=True)

# 设置其他目录
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/integrated")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 导入自定义修复和测试模块
sys.path.insert(0, SCRIPT_DIR)
try:
    from lammphonon_fixer import (
        # 补丁功能
        patch_thermal_analyzer,
        patch_temporal_analyzer,
        patch_equilibration_analyzer,
        apply_all_patches,
        # 修复功能
        fix_thermal_analyzer,
        fix_temporal_analyzer,
        fix_equilibration_analyzer,
        add_missing_methods,
        fix_all_modules,
        # 测试功能
        test_phonon_analyzer,
        test_thermal_analyzer,
        test_temporal_analyzer,
        test_equilibration_analyzer,
        run_all_tests
    )
except ImportError:
    logger.error("无法导入修复和测试模块，请确保lammphonon_fixer目录存在并包含相应模块")
    sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LAMMPhonon修复和测试集成程序')
    
    # 运行模式
    parser.add_argument('--mode', choices=['patch', 'fix', 'test', 'all'], default='all',
                       help='运行模式: patch(运行时补丁), fix(直接修改源码), test(仅测试), all(全部)')
    
    # 需要处理的模块
    parser.add_argument('--modules', nargs='+', 
                       choices=['all', 'thermal', 'temporal', 'equilibration', 'phonon'],
                       default=['all'],
                       help='需要修复或测试的模块')
    
    # 数据目录
    parser.add_argument('--data', type=str, default=DATA_DIR,
                       help='测试数据目录路径')
    
    # 输出目录
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                       help='输出目录路径')
    
    # 详细模式
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    
    return parser.parse_args()

def load_coordinator(data_dir, load_all=True):
    """
    加载PhononCoordinator并读取测试数据
    
    参数:
        data_dir: 测试数据目录
        load_all: 是否加载所有类型的数据
        
    返回:
        coordinator: 加载数据后的PhononCoordinator实例
    """
    logger.info(f"从 {data_dir} 加载测试数据")
    
    # 创建协调器
    coordinator = PhononCoordinator()
    
    # 设置输出目录
    coordinator.output_dir = OUTPUT_DIR
    
    # 读取轨迹文件
    traj_file = os.path.join(data_dir, "test_dump.phonon")
    if os.path.exists(traj_file):
        logger.info(f"读取轨迹文件: {traj_file}")
        coordinator.read_trajectory(traj_file)
    else:
        logger.error(f"轨迹文件不存在: {traj_file}")
        sys.exit(1)
    
    if load_all:
        # 读取能量文件
        energy_file = os.path.join(data_dir, "test_energy.txt")
        if os.path.exists(energy_file):
            logger.info(f"读取能量文件: {energy_file}")
            coordinator.read_energy_data(energy_file)
        else:
            logger.warning(f"能量文件不存在: {energy_file}")
        
        # 读取热流数据
        heatflux_file = os.path.join(data_dir, "test_heatflux.dat")
        if os.path.exists(heatflux_file):
            logger.info(f"读取热流文件: {heatflux_file}")
            coordinator.read_heatflux_data(heatflux_file)
        else:
            logger.warning(f"热流文件不存在: {heatflux_file}")
        
        # 读取极化数据
        polarization_file = os.path.join(data_dir, "test_polarization.txt")
        if os.path.exists(polarization_file):
            logger.info(f"读取极化文件: {polarization_file}")
            coordinator.read_polarization_data(polarization_file)
        else:
            logger.warning(f"极化文件不存在: {polarization_file}")
    
    # 打印数据结构
    logger.info(f"\n数据加载情况:")
    logger.info(f"轨迹数据类型: {type(coordinator.trajectory_data)}")
    
    if hasattr(coordinator, 'trajectory_data') and isinstance(coordinator.trajectory_data, dict):
        logger.info(f"轨迹数据包含键: {coordinator.trajectory_data.keys()}")
        logger.info(f"轨迹数据帧数: {len(coordinator.trajectory_data['positions']) if 'positions' in coordinator.trajectory_data else 0}")
    
    if hasattr(coordinator, 'energy_data'):
        logger.info(f"能量数据形状: {coordinator.energy_data.shape if hasattr(coordinator.energy_data, 'shape') else type(coordinator.energy_data)}")
    
    if hasattr(coordinator, 'heatflux_data'):
        logger.info(f"热流数据形状: {coordinator.heatflux_data.shape if hasattr(coordinator.heatflux_data, 'shape') else type(coordinator.heatflux_data)}")
    
    if hasattr(coordinator, 'polarization_data'):
        logger.info(f"极化数据形状: {coordinator.polarization_data.shape if hasattr(coordinator.polarization_data, 'shape') else type(coordinator.polarization_data)}")
    
    return coordinator

def prepare_modules():
    """准备模块字典，包含各种分析器类"""
    return {
        'PhononAnalyzer': PhononAnalyzer,
        'ThermalAnalyzer': ThermalAnalyzer,
        'AnharmonicAnalyzer': AnharmonicAnalyzer,
        'TemporalAnalyzer': TemporalAnalyzer,
        'EquilibrationAnalyzer': EquilibrationAnalyzer
    }

def filter_modules(module_dict, selected_modules):
    """
    过滤需要处理的模块
    
    参数:
        module_dict: 包含所有模块的字典
        selected_modules: 选择的模块列表
        
    返回:
        filtered_dict: 过滤后的模块字典
    """
    if 'all' in selected_modules:
        return module_dict
    
    filtered_dict = {}
    mapping = {
        'phonon': 'PhononAnalyzer',
        'thermal': 'ThermalAnalyzer',
        'temporal': 'TemporalAnalyzer',
        'equilibration': 'EquilibrationAnalyzer'
    }
    
    for module_name in selected_modules:
        if module_name in mapping and mapping[module_name] in module_dict:
            filtered_dict[mapping[module_name]] = module_dict[mapping[module_name]]
    
    return filtered_dict

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 准备输出目录
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    coordinator = load_coordinator(args.data)
    
    # 准备并过滤模块
    all_modules = prepare_modules()
    modules_to_process = filter_modules(all_modules, args.modules)
    
    # 根据运行模式执行相应功能
    if args.mode in ['patch', 'all']:
        logger.info("\n===== 运行时补丁模式 =====")
        apply_all_patches(modules_to_process)
    
    if args.mode in ['fix', 'all']:
        logger.info("\n===== 直接修改源码模式 =====")
        fix_all_modules(modules_to_process)
    
    if args.mode in ['test', 'all']:
        logger.info("\n===== 测试模式 =====")
        test_result = run_all_tests(coordinator, output_dir)
        if test_result:
            logger.info("所有测试通过！")
        else:
            logger.warning("部分测试失败！")
    
    logger.info("\n程序执行完毕")

if __name__ == "__main__":
    main() 