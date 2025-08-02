#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon修复和测试工具包
=======================

整合各种修复方法和测试功能，提供统一的接口。

作者: 梁树铭 (Shuming Liang)
"""

from .patch_modules import (
    patch_thermal_analyzer,
    patch_temporal_analyzer, 
    patch_equilibration_analyzer,
    apply_all_patches
)

from .fix_modules import (
    fix_thermal_analyzer,
    fix_temporal_analyzer,
    fix_equilibration_analyzer,
    add_missing_methods,
    fix_all_modules
)

from .test_modules import (
    test_phonon_analyzer,
    test_thermal_analyzer,
    test_temporal_analyzer,
    test_equilibration_analyzer,
    run_all_tests
)

__all__ = [
    'patch_thermal_analyzer',
    'patch_temporal_analyzer', 
    'patch_equilibration_analyzer',
    'apply_all_patches',
    'fix_thermal_analyzer',
    'fix_temporal_analyzer',
    'fix_equilibration_analyzer',
    'add_missing_methods',
    'fix_all_modules',
    'test_phonon_analyzer',
    'test_thermal_analyzer',
    'test_temporal_analyzer',
    'test_equilibration_analyzer',
    'run_all_tests'
] 