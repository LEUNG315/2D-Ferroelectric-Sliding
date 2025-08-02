# LAMMPhonon - 声子分析工具包

LAMMPhonon是一个用于LAMMPS分子动力学模拟的声子分析工具包，提供了声子态密度计算、模式投影、滑移分析等多种功能。

作者: 梁树铭 (Shuming Liang)  
邮箱: lsm315@mail.ustc.edu.cn  
电话: 18189209026

## 功能特点

- 读取并分析LAMMPS轨迹文件（dump/lammpstrj格式）
- 计算速度自相关函数(VACF)和声子态密度(DOS)
- 支持原子投影和方向投影的声子态密度计算
- 正则模式分析和模式占据数统计
- 双层材料滑移过程分析，包括：
  - 滑移距离和层间距离计算
  - 摩擦力和摩擦系数计算
  - 黏滑行为和能量耗散分析
  - 堆垛构型参数计算
- 热传导相关分析：
  - 热流自相关函数计算
  - 格林-库博公式热导率计算
- 新增温度平衡分析功能：
  - 系统能量平衡过程分析
  - 平衡时间自动计算
- 新增时间分辨分析功能：
  - 时间窗口内声子态密度演化
  - 非平衡态声子分布随时间变化
- 交互式中文菜单界面和英文命令行接口：
  - 全中文互动操作菜单
  - 全英文命令行参数化分析
  - 所有图例和标签为英文，支持科学论文使用
- 新增自动修复工具：
  - 运行时补丁修复常见问题
  - 自动错误处理和数据验证
- 模块化设计，支持脚本和API调用

## 安装方法

### 从源代码安装

```bash
git clone https://github.com/LEUNG315/2D-Ferroelectric-Sliding.git
cd lammphonon
pip install -e .
```

### 通过pip安装

```bash
pip install lammphonon
```

## 使用方法

### 交互式菜单

启动交互式中文菜单：

```bash
lammphonon_menu
```

或者：

```bash
python -m lammphonon
```

### 命令行分析工具

新增的命令行分析工具，支持参数化分析：

```bash
lammphonon_analyze -i trajectory.lammpstrj -o results/ -dos
```

常用参数：
- `-i, --input`: 输入轨迹文件
- `-o, --output`: 输出目录
- `--energy, --force, --heatflux`: 额外数据文件
- `-dos`: 计算声子态密度
- `-po`: 计算声子占据数
- `-sliding`: 进行滑移分析
- `-thermal`: 进行热分析
- `-equil`: 进行平衡时间分析
- `-all`: 执行所有分析
- `--apply-patches`: 应用自动修复补丁

完整参数列表请使用：
```bash
lammphonon_analyze --help
```

### Python API调用

基础API用法：

```python
import numpy as np
from lammphonon import PhononCoordinator, PhononAnalyzer, SlidingAnalyzer

# 创建协调器
coordinator = PhononCoordinator()

# 读取轨迹文件
coordinator.read_trajectory("trajectory.lammpstrj")

# 声子分析
phonon_analyzer = PhononAnalyzer(coordinator)
vacf = phonon_analyzer.calculate_velocity_autocorrelation(
    coordinator.trajectory_data['velocities']
)
freqs, dos = phonon_analyzer.calculate_dos(vacf)

# 滑移分析
sliding_analyzer = SlidingAnalyzer(coordinator)
sliding_analyzer.detect_layers()
sliding_distance = sliding_analyzer.calculate_sliding_distance()
```

新增分析器用法：

```python
from lammphonon import ThermalAnalyzer, EquilibrationAnalyzer, TemporalAnalyzer

# 热分析
thermal_analyzer = ThermalAnalyzer(coordinator)
kappa = thermal_analyzer.calculate_thermal_conductivity(
    coordinator.heatflux_data,
    temperature=300.0,
    volume=1000.0
)

# 平衡分析
equilibration_analyzer = EquilibrationAnalyzer(coordinator)
equil_time, params = equilibration_analyzer.calculate_system_equilibration(
    coordinator.energy_data['total']
)

# 时间分辨分析
temporal_analyzer = TemporalAnalyzer(coordinator)
freqs, dos_evolution, times = temporal_analyzer.calculate_time_resolved_dos(
    coordinator.trajectory_data['velocities']
)
```

## 输入文件要求

### 轨迹文件

支持LAMMPS dump/lammpstrj格式的轨迹文件，需要包含以下列：

- `id`, `type`: 原子ID和类型
- `x`, `y`, `z`: 原子坐标
- `vx`, `vy`, `vz`: 原子速度（用于声子分析）
- `fx`, `fy`, `fz`: 原子受力（可选，用于摩擦分析）

### 其他数据文件

工具还支持读取以下格式的数据文件：

- 能量文件：包含总能量、动能、势能等数据的文本文件
- 热流文件：包含热流矢量数据的文本文件
- 极化文件：包含极化矢量数据的文本文件
- 层定义文件：定义材料层的文本文件
- 质量文件：定义原子质量的文本文件

## 输出格式

分析结果会保存在输出目录（默认为`~/lammphonon_results`）中，包括：

- 文本数据文件（.txt/.dat）：包含计算的数值结果
- 图像文件（.png/.pdf）：可视化的分析图表
- 数据库文件（.npz）：存储中间计算结果，便于后续分析
- 报告文件：总结分析结果

## 版本历史

### Version 2.0.0
- 集成了多种新分析功能
- 添加了自动修复和错误处理机制
- 新增命令行接口和参数化分析
- 增强了可视化效果和数据输出
- 改进了模块化结构和API设计
- 解决了多个已知问题

### Version 1.0.0
- 初始版本，包含基础声子分析功能
- 添加了滑移分析和摩擦计算功能
- 实现了交互式中文菜单界面

## 示例

请参考`examples`目录中的示例脚本和数据文件。

## 引用

如果您在研究中使用了LAMMPhonon，请引用：

```
Liang, SM. (2025). LAMMPhonon: A toolkit for phonon analysis in LAMMPS simulations.
```

## 许可证

MIT 许可证 
