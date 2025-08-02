# LAMMPhonon - 声子分析工具包

LAMMPhonon是一个用于LAMMPS分子动力学模拟的声子分析工具包，提供了声子态密度计算、模式投影、滑移分析等多种功能。

作者: 梁树铭 (Shuming Liang)  
邮箱: lsm315@mail.ustc.edu.cn  
电话: 18256949203

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
- 交互式中文菜单界面
- 模块化设计，支持脚本和API调用

## 安装方法

### 从源代码安装

```bash
git clone https://github.com/liangshuming/lammphonon.git
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

### Python API调用

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
sliding_analyzer.detect_material_layers(
    coordinator.trajectory_data['positions'][0]
)
sliding_distance = sliding_analyzer.calculate_sliding_distance(
    coordinator.trajectory_data['positions']
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

## 输出格式

分析结果会保存在输出目录（默认为`~/lammphonon_results`）中，包括：

- 文本数据文件（.txt/.dat）：包含计算的数值结果
- 图像文件（.png/.pdf）：可视化的分析图表
- 报告文件：总结分析结果

## 示例

请参考`examples`目录中的示例脚本和数据文件。

## 引用

如果您在研究中使用了LAMMPhonon，请引用：

```
Liang, S. (2023). LAMMPhonon: A toolkit for phonon analysis in LAMMPS simulations.
```

## 许可证

MIT 许可证 