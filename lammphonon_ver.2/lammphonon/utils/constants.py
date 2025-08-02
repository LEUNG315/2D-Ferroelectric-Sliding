#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physical constants and unit conversions for phonon analysis
"""

import numpy as np

# Physical constants
KB = 1.380649e-23  # Boltzmann constant (J/K)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
EV_TO_J = 1.602176634e-19  # Conversion from eV to J
AVOGADRO = 6.02214076e23  # Avogadro constant
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg
ANGSTROM_TO_M = 1e-10  # Angstrom to meter
FS_TO_S = 1e-15  # Femtosecond to second
PS_TO_S = 1e-12  # Picosecond to second

# LAMMPS commonly used units
UNIT_SYSTEMS = {
    'metal': {
        'mass': 'g/mol',
        'distance': 'Å',
        'time': 'ps',
        'energy': 'eV',
        'velocity': 'Å/ps',
        'force': 'eV/Å',
        'temperature': 'K',
        'pressure': 'bar',
        'kb': 8.617333262e-5,  # eV/K
        'hbar': 6.582119569e-16  # eV·ps
    },
    'real': {
        'mass': 'g/mol',
        'distance': 'Å',
        'time': 'fs',
        'energy': 'kcal/mol',
        'velocity': 'Å/fs',
        'force': 'kcal/(mol·Å)',
        'temperature': 'K',
        'pressure': 'atm',
        'kb': 0.0019872067,  # kcal/(mol·K)
        'hbar': 9.5414e-4  # kcal·fs/mol
    },
    'si': {
        'mass': 'kg',
        'distance': 'm',
        'time': 's',
        'energy': 'J',
        'velocity': 'm/s',
        'force': 'N',
        'temperature': 'K',
        'pressure': 'Pa',
        'kb': 1.380649e-23,  # J/K
        'hbar': 1.054571817e-34  # J·s
    }
}

# Default atom masses (g/mol)
DEFAULT_MASSES = {
    'H': 1.008,
    'He': 4.003,
    'Li': 6.941,
    'Be': 9.012,
    'B': 10.811,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.086,
    'P': 30.974,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'K': 39.098,
    'Ca': 40.078,
    'Ti': 47.867,
    'Fe': 55.845,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.64,
    'As': 74.922,
    'Mo': 95.96,
    'Pd': 106.42,
    'Ag': 107.868,
    'Au': 196.967
}

# Conversion functions
def temp_to_energy(temperature, units='metal'):
    """Convert temperature to energy in the given unit system"""
    return temperature * UNIT_SYSTEMS[units]['kb']

def energy_to_temp(energy, units='metal'):
    """Convert energy to temperature in the given unit system"""
    return energy / UNIT_SYSTEMS[units]['kb']

def freq_to_energy(frequency, units='metal'):
    """Convert frequency (THz) to energy in the given unit system"""
    # Convert to angular frequency (rad/s)
    omega = 2 * np.pi * frequency * 1e12
    # Calculate energy using E = ħω
    if units == 'metal':
        return UNIT_SYSTEMS[units]['hbar'] * omega / 1e12  # Convert to eV
    elif units == 'real':
        return UNIT_SYSTEMS[units]['hbar'] * omega * 1e-15  # Convert to kcal/mol
    else:  # SI
        return UNIT_SYSTEMS[units]['hbar'] * omega  # J

def energy_to_freq(energy, units='metal'):
    """Convert energy to frequency (THz) in the given unit system"""
    if units == 'metal':
        omega = energy / UNIT_SYSTEMS[units]['hbar'] * 1e12
    elif units == 'real':
        omega = energy / UNIT_SYSTEMS[units]['hbar'] / 1e-15
    else:  # SI
        omega = energy / UNIT_SYSTEMS[units]['hbar']
    
    return omega / (2 * np.pi) / 1e12  # Return in THz 