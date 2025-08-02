#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anharmonic phonon analysis module
"""

import numpy as np
import logging
import scipy.optimize
from ..utils.timing import Timer, TimeProfiler
from ..utils.constants import KB, HBAR, temp_to_energy, energy_to_temp, freq_to_energy, energy_to_freq

# Setup logging
logger = logging.getLogger(__name__)

class AnharmonicAnalyzer:
    """Anharmonic phonon analysis for materials"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the anharmonic analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("AnharmonicAnalyzer")
        
        # Default parameters
        self.temperature = 300.0  # K
        self.freq_max = 30.0  # THz
        self.freq_points = 1000
        self.mode_bin_size = 10  # Number of modes per bin for statistics
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.temperature = coordinator.get_config('temperature', self.temperature)
            self.freq_max = coordinator.get_config('freq_max', self.freq_max)
            self.freq_points = coordinator.get_config('freq_points', self.freq_points)
        
        # Analysis results
        self.mode_frequencies = None
        self.mode_energies = None
        self.mode_occupations = None
        self.bose_einstein_occupations = None
        self.occupation_deviations = None
        self.mode_effective_temps = None
        self.gruneisen_parameters = None
        
        logger.debug("AnharmonicAnalyzer initialized")
    
    def calculate_bose_einstein_distribution(self, frequencies, temperature=None):
        """
        Calculate Bose-Einstein distribution for given frequencies
        
        Parameters:
            frequencies: Array of frequencies in THz
            temperature: Temperature in K
            
        Returns:
            n: Bose-Einstein occupation numbers
        """
        self.profiler.start("calculate_bose_einstein")
        
        # Use provided temperature or default
        if temperature is None:
            temperature = self.temperature
        
        # Convert frequency to angular frequency (2πf)
        omega = 2 * np.pi * frequencies * 1e12  # convert to Hz
        
        # Calculate Bose-Einstein occupation: n = 1/(exp(ℏω/kT) - 1)
        # KB = 1.380649e-23  # J/K
        # HBAR = 1.054571817e-34  # J.s
        
        boltzmann = KB
        hbar = HBAR
        
        # Avoid division by zero for zero frequency
        n = np.zeros_like(frequencies)
        nonzero = frequencies > 0
        
        # Calculate occupation
        x = hbar * omega[nonzero] / (boltzmann * temperature)
        n[nonzero] = 1.0 / (np.exp(x) - 1.0)
        
        # Store results
        self.bose_einstein_occupations = n
        
        self.profiler.stop()
        return n
    
    def compare_occupation_with_bose_einstein(self, mode_energies, frequencies, temperature=None):
        """
        Compare mode occupation with Bose-Einstein distribution
        
        Parameters:
            mode_energies: Modal energies
            frequencies: Mode frequencies in THz
            temperature: Temperature in K
            
        Returns:
            occupation: Actual mode occupation
            be_occupation: Bose-Einstein occupation
            deviation: Deviation factor
        """
        self.profiler.start("compare_occupation")
        
        # Use provided temperature or default
        if temperature is None:
            temperature = self.temperature
        
        # Calculate actual occupation from mode energies
        # E = ℏω(n + 1/2)
        # n = E/(ℏω) - 1/2
        
        # Avoid division by zero for zero frequency
        occupation = np.zeros_like(frequencies)
        nonzero = frequencies > 0
        
        # Calculate occupation
        # Convert frequency to angular frequency (2πf)
        omega = 2 * np.pi * frequencies[nonzero] * 1e12  # convert to Hz
        
        # Calculate ℏω in J
        hbar_omega = HBAR * omega
        
        # Convert mode energies from eV to J
        # 1 eV = 1.602176634e-19 J
        energies_j = mode_energies[nonzero] * 1.602176634e-19
        
        # Calculate occupation
        occupation[nonzero] = energies_j / hbar_omega - 0.5
        
        # Calculate Bose-Einstein occupation
        be_occupation = self.calculate_bose_einstein_distribution(frequencies, temperature)
        
        # Calculate deviation factor (actual / BE)
        deviation = np.zeros_like(frequencies)
        nonzero_be = be_occupation > 0
        deviation[nonzero_be] = occupation[nonzero_be] / be_occupation[nonzero_be]
        
        # Store results
        self.mode_occupations = occupation
        self.bose_einstein_occupations = be_occupation
        self.occupation_deviations = deviation
        
        self.profiler.stop()
        return occupation, be_occupation, deviation
    
    def calculate_mode_effective_temperatures(self, mode_energies, frequencies):
        """
        Calculate effective temperature for each mode
        
        Parameters:
            mode_energies: Modal energies
            frequencies: Mode frequencies in THz
            
        Returns:
            effective_temps: Effective temperature for each mode
        """
        self.profiler.start("calculate_effective_temps")
        
        # Calculate effective temperature from mode energy
        # E = ℏω(n + 1/2)
        # n = 1/(exp(ℏω/kT) - 1)
        # Solve for T
        
        # Avoid processing zero frequency modes
        effective_temps = np.zeros_like(frequencies)
        nonzero = frequencies > 0
        
        # Convert frequency to angular frequency (2πf)
        omega = 2 * np.pi * frequencies[nonzero] * 1e12  # convert to Hz
        
        # Calculate ℏω in J
        hbar_omega = HBAR * omega
        
        # Convert mode energies from eV to J
        # 1 eV = 1.602176634e-19 J
        energies_j = mode_energies[nonzero] * 1.602176634e-19
        
        # Calculate occupation
        occupation = energies_j / hbar_omega - 0.5
        
        # Calculate effective temperature
        # n = 1/(exp(ℏω/kT) - 1)
        # exp(ℏω/kT) = 1 + 1/n
        # ℏω/kT = ln(1 + 1/n)
        # T = ℏω/(k*ln(1 + 1/n))
        
        # Handle special cases and numerical stability
        valid = (occupation > 0)
        
        if np.sum(valid) > 0:
            # Calculate temperature for valid modes
            hbar_omega_valid = hbar_omega[valid]
            occupation_valid = occupation[valid]
            
            # Avoid very small occupation values
            occupation_valid = np.maximum(occupation_valid, 1e-10)
            
            # Calculate T
            log_term = np.log(1 + 1/occupation_valid)
            effective_temps[nonzero][valid] = hbar_omega_valid / (KB * log_term)
        
        # Store results
        self.mode_effective_temps = effective_temps
        
        self.profiler.stop()
        return effective_temps
    
    def calculate_anharmonic_factor(self, mode_energies, frequencies, temperature_range=None):
        """
        Calculate anharmonic factor based on temperature dependence
        
        Parameters:
            mode_energies: Dictionary of modal energies at different temperatures
            frequencies: Mode frequencies in THz
            temperature_range: List of temperatures for analysis
            
        Returns:
            anharmonic_factor: Anharmonicity factor for each mode
        """
        self.profiler.start("calculate_anharmonic_factor")
        
        # Default temperature range if not provided
        if temperature_range is None:
            temperature_range = [100, 200, 300, 400, 500]
        
        # Check if we have energies for all temperatures
        for temp in temperature_range:
            if str(temp) not in mode_energies:
                logger.error(f"Mode energies for temperature {temp}K not provided")
                self.profiler.stop()
                return None
        
        # Calculate anharmonic factor for each mode
        n_modes = len(frequencies)
        anharmonic_factor = np.zeros(n_modes)
        
        # We'll fit energy vs temperature to get anharmonic factor
        for i in range(n_modes):
            if frequencies[i] <= 0:
                continue
                
            # Collect energies for this mode at different temperatures
            temps = np.array(temperature_range)
            energies = np.array([mode_energies[str(temp)][i] for temp in temperature_range])
            
            try:
                # Fit linear model: E = a + b*T
                # b/a is the anharmonic factor
                coeffs = np.polyfit(temps, energies, 1)
                if abs(coeffs[1]) > 1e-10:  # avoid division by zero
                    anharmonic_factor[i] = coeffs[0] / coeffs[1]
                else:
                    anharmonic_factor[i] = 0
            except:
                anharmonic_factor[i] = 0
        
        # Store results
        self.anharmonic_factor = anharmonic_factor
        
        self.profiler.stop()
        return anharmonic_factor
    
    def calculate_gruneisen_parameter(self, frequencies, volume, volume_derivative):
        """
        Calculate mode Grüneisen parameters
        
        Parameters:
            frequencies: Mode frequencies in THz
            volume: System volume
            volume_derivative: Derivative of frequencies with respect to volume
            
        Returns:
            gruneisen: Grüneisen parameters for each mode
        """
        self.profiler.start("calculate_gruneisen")
        
        # Compute Grüneisen parameter: γ = -(V/ω)(∂ω/∂V)
        gruneisen = np.zeros_like(frequencies)
        
        # Avoid division by zero
        nonzero = frequencies > 0
        gruneisen[nonzero] = -(volume / frequencies[nonzero]) * volume_derivative[nonzero]
        
        # Store results
        self.gruneisen_parameters = gruneisen
        
        self.profiler.stop()
        return gruneisen
    
    def analyze_occupation_equilibration(self, mode_energies_time_series, frequencies, timestep=None):
        """
        Analyze equilibration of phonon occupation over time
        
        Parameters:
            mode_energies_time_series: Time series of mode energies [times, modes]
            frequencies: Mode frequencies in THz
            timestep: Time step in ps
            
        Returns:
            equilibration_times: Equilibration time for each mode
            occupation_time_series: Time series of occupation numbers
        """
        self.profiler.start("analyze_occupation_equilibration")
        
        # Default time step
        if timestep is None:
            timestep = self.coordinator.get_config('timestep', 0.001) if self.coordinator else 0.001
        
        # Get dimensions
        n_times, n_modes = mode_energies_time_series.shape
        
        # Calculate occupation time series
        occupation_time_series = np.zeros_like(mode_energies_time_series)
        
        # Avoid zero frequency modes
        nonzero = frequencies > 0
        
        # Convert frequency to angular frequency (2πf)
        omega = 2 * np.pi * frequencies[nonzero] * 1e12  # convert to Hz
        
        # Calculate ℏω in J
        hbar_omega = HBAR * omega
        
        # Convert mode energies from eV to J
        # 1 eV = 1.602176634e-19 J
        for t in range(n_times):
            energies_j = mode_energies_time_series[t, nonzero] * 1.602176634e-19
            occupation_time_series[t, nonzero] = energies_j / hbar_omega - 0.5
        
        # Calculate equilibration time for each mode
        equilibration_times = np.zeros(n_modes)
        equilibration_params = np.zeros((n_modes, 3))  # A, tau, n_inf
        
        # Create time array
        times = np.arange(n_times) * timestep
        
        # Define exponential decay function for fitting
        def exp_decay(t, A, tau, n_inf):
            return A * np.exp(-t / tau) + n_inf
        
        # Fit each mode
        for i in range(n_modes):
            if not nonzero[i] or np.any(np.isnan(occupation_time_series[:, i])):
                continue
                
            try:
                # Initial guesses
                n_data = occupation_time_series[:, i]
                n_inf_guess = np.mean(n_data[-int(n_times/5):])  # average of last 20%
                A_guess = n_data[0] - n_inf_guess
                tau_guess = n_times * timestep / 5
                
                # Fit exponential decay
                popt, _ = scipy.optimize.curve_fit(
                    exp_decay, times, n_data,
                    p0=[A_guess, tau_guess, n_inf_guess],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    maxfev=5000
                )
                
                # Store parameters
                equilibration_params[i] = popt
                equilibration_times[i] = popt[1]  # tau
            except:
                # If fitting fails, use time to reach 1/e of initial deviation
                try:
                    n_data = occupation_time_series[:, i]
                    n_final = np.mean(n_data[-int(n_times/5):])
                    initial_dev = n_data[0] - n_final
                    target = n_final + initial_dev / np.e
                    
                    # Find first time point below target
                    if initial_dev > 0:
                        idx = np.argmax(n_data <= target)
                    else:
                        idx = np.argmax(n_data >= target)
                    
                    if idx > 0:
                        equilibration_times[i] = times[idx]
                    else:
                        equilibration_times[i] = times[-1]
                except:
                    equilibration_times[i] = 0
        
        # Store results
        self.equilibration_times = equilibration_times
        self.equilibration_params = equilibration_params
        self.occupation_time_series = occupation_time_series
        
        self.profiler.stop()
        return equilibration_times, occupation_time_series
    
    def get_anharmonic_analysis_summary(self):
        """
        Get a summary of anharmonic analysis results
        
        Returns:
            summary: Dictionary with analysis results
        """
        summary = {
            "mode_occupations": self.mode_occupations,
            "bose_einstein_occupations": self.bose_einstein_occupations,
            "occupation_deviations": self.occupation_deviations,
            "mode_effective_temps": self.mode_effective_temps,
            "equilibration_times": self.equilibration_times if hasattr(self, 'equilibration_times') else None,
            "anharmonic_factor": self.anharmonic_factor if hasattr(self, 'anharmonic_factor') else None,
            "gruneisen_parameters": self.gruneisen_parameters if hasattr(self, 'gruneisen_parameters') else None,
            "temperature": self.temperature
        }
        
        return summary 