#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAMMPhonon - Fixer Module
========================

Provides runtime patches, fixes, and testing utilities for LAMMPhonon.

Author: Shuming Liang (梁树铭)
Email: lsm315@mail.ustc.edu.cn
Phone: 18256949203
"""

from lammphonon.fixer.patch_modules import (
    patch_thermal_analyzer,
    patch_temporal_analyzer, 
    patch_equilibration_analyzer,
    apply_all_patches
)

from lammphonon.fixer.fix_modules import (
    fix_thermal_analyzer,
    fix_temporal_analyzer,
    fix_equilibration_analyzer,
    add_missing_methods,
    fix_all_modules
)

from lammphonon.fixer.test_modules import (
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