# LAMMPhonon - 二维材料声子分析工具包

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LAMMPS](https://img.shields.io/badge/LAMMPS-Compatible-orange.svg)](https://lammps.sandia.gov/)

**LAMMPhonon** 是一个专为LAMMPS分子动力学模拟设计的声子分析工具包，主要用于研究二维材料（如石墨烯）的滑移铁电效应和声子能量耗散机制。该工具包是我毕业论文《二维材料层间滑移动力学与声子能量耗散机制研究》的核心分析工具。

##  研究背景

近年来，二维范德瓦尔斯材料中涌现出一类新型铁电性——**滑移铁电**，其极化翻转源于独特的层间滑移机制。当双层结构失去上下层原子堆垛的镜面对称性时，净层间的垂直电荷转移会诱导净极化，而层间滑移可实现极化方向的可控翻转。理论研究表明，此类体系的滑移势垒较传统铁电体降低2-3个数量级，为开发超低能耗铁电器件提供了可能。

##  核心功能

### 滑移动力学分析
- **滑移距离与速度演化**: 分析滑移过程中的位移、速度变化规律
- **摩擦系数计算**: 基于摩擦力"锯齿"振荡特性计算摩擦系数
- **能量耗散分析**: 研究势能/动能耦合和耗散速率

### 声子动力学分析
- **声子态密度计算**: 基于速度自相关函数(VACF)的声子态密度分析
- **声子寿命分析**: 计算不同声子模式的寿命及其与频率的关系
- **能量分配研究**: 分析能量在不同声子模式间的分配特性
- **热平衡过程**: 研究声子占据数与玻色-爱因斯坦分布的偏差

### 非谐性效应研究
- **声子非谐性**: 分析滑移过程中的非谐效应及其对能量耗散的影响
- **模式耦合**: 研究不同声子模式间的耦合机制
- **界面效应**: 分析界面散射对声子输运的影响

##  安装

```bash
# 从源代码安装
git clone https://github.com/liangshuming/lammphonon.git
cd lammphonon
pip install -e .

# 或通过pip安装
pip install lammphonon
```

##  快速使用

### 交互式分析
```bash
lammphonon_menu
```

### 命令行分析
```bash
# 基础声子分析
lammphonon_analyze -i trajectory.lammpstrj -o results/ -dos

# 完整分析（包含非谐性和摩擦耦合）
lammphonon_analyze -i trajectory.lammpstrj -o results/ -all
```

### Python API
```python
from lammphonon import PhononCoordinator, PhononAnalyzer, SlidingAnalyzer

# 创建分析器
coordinator = PhononCoordinator()
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

##  主要研究

- **滑移动力学特性**
- **声子能量耗散机制**
- **非平衡热力学过程**


##  应用案例

项目包含完整的石墨烯双层滑移模拟和分析示例：
- **LAMMPS输入文件**: `in.graphene_phonon_analysis` - 双层石墨烯滑移模型构建
- **非谐性分析**: `anharmonic_analysis_*.py` - 声子寿命和非谐性效应分析
- **摩擦耦合分析**: `phonon_friction_analysis_*.py` - 声子-摩擦耦合机制研究

##  输入要求

支持LAMMPS dump/lammpstrj格式，需要包含：
- 原子坐标 (`x`, `y`, `z`)
- 原子速度 (`vx`, `vy`, `vz`)
- 原子受力 (`fx`, `fy`, `fz`) - 可选

##  引用

如果您在研究中使用了LAMMPhonon，请引用：

```bibtex
@software{liang2023lammphonon,
  title={LAMMPhonon: A toolkit for phonon analysis in LAMMPS simulations},
  author={Liang, Shuming},
  year={2023},
  url={https://github.com/liangshuming/lammphonon}
}
```

##  作者

**梁树铭 (Shuming Liang)**  
-  邮箱: lsm315@mail.ustc.edu.cn  
-  中国科学技术大学 凝聚态物理专业
-  论文: 《二维材料层间滑移动力学与声子能量耗散机制研究》
-  导师: 赵瑾 教授

##  许可证

MIT 许可证

---
