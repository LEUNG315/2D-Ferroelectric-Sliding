#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LammPhonon - Phonon Analysis Toolkit for LAMMPS
==============================================

A comprehensive phonon analysis toolkit for molecular dynamics simulations.

Author: Shuming Liang (梁树铭)
Email: lsm315@mail.ustc.edu.cn
Phone: 18256949203

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Shuming Liang (梁树铭)"
__email__ = "lsm315@mail.ustc.edu.cn"

from lammphonon.core.coordinator import PhononCoordinator
from lammphonon.core.phonon_analyzer import PhononAnalyzer
from lammphonon.analysis.sliding_analyzer import SlidingAnalyzer
from lammphonon.analysis.thermal_analyzer import ThermalAnalyzer
from lammphonon.analysis.anharmonic_analyzer import AnharmonicAnalyzer
from lammphonon.analysis.equilibration_analyzer import EquilibrationAnalyzer
from lammphonon.analysis.temporal_analyzer import TemporalAnalyzer
from lammphonon.menu.main_menu import MainMenu
from lammphonon.fixer import patch_modules, fix_modules, test_modules

__all__ = [
    "PhononCoordinator", 
    "PhononAnalyzer", 
    "SlidingAnalyzer",
    "ThermalAnalyzer",
    "AnharmonicAnalyzer",
    "EquilibrationAnalyzer",
    "TemporalAnalyzer",
    "MainMenu",
    "patch_modules",
    "fix_modules",
    "test_modules"
] 