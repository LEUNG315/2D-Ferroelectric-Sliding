#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Equilibration process analysis module
"""

import numpy as np
import logging
import scipy.optimize
from ..utils.timing import Timer, TimeProfiler
from ..utils.constants import KB, HBAR
from ..utils.helpers import smooth_data

# Setup logging
logger = logging.getLogger(__name__)

class EquilibrationAnalyzer:
    """Equilibration process analysis for non-equilibrium systems"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the equilibration analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("EquilibrationAnalyzer")
        
        # Default parameters
        self.temperature = 300.0  # K
        self.timestep = 0.001  # ps
        self.n_time_windows = 10
        self.freq_bins = 10
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.temperature = coordinator.get_config('temperature', self.temperature)
            self.timestep = coordinator.get_config('timestep', self.timestep)
            self.n_time_windows = coordinator.get_config('n_time_windows', self.n_time_windows)
            self.freq_bins = coordinator.get_config('freq_bins', self.freq_bins)
        
        # Analysis results
        self.equilibration_times = None
        self.energy_evolution = None
        self.time_constants = None
        self.equilibrium_values = None
        
        logger.debug("EquilibrationAnalyzer initialized")
    
    def calculate_system_equilibration(self, energy_time_series, time_points=None):
        """
        Calculate system equilibration from total energy time series
        
        Parameters:
            energy_time_series: Total energy over time
            time_points: Time points corresponding to energy values
            
        Returns:
            equilibration_time: System equilibration time
            equilibration_params: Fitted parameters
        """
        self.profiler.start("calculate_system_equilibration")
        
        # Create time array if not provided
        if time_points is None:
            time_points = np.arange(len(energy_time_series)) * self.timestep
        
        # Define exponential decay function for fitting
        def exp_decay(t, A, tau, E_inf):
            return A * np.exp(-t / tau) + E_inf
        
        # Fit exponential decay
        try:
            # Initial guesses
            E_inf_guess = np.mean(energy_time_series[-int(len(energy_time_series)/5):])  # avg of last 20%
            A_guess = energy_time_series[0] - E_inf_guess
            tau_guess = len(energy_time_series) * self.timestep / 5
            
            # Fit curve
            popt, pcov = scipy.optimize.curve_fit(
                exp_decay, time_points, energy_time_series,
                p0=[A_guess, tau_guess, E_inf_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=5000
            )
            
            # Calculate equilibration time (tau)
            equilibration_time = popt[1]
            
            # Calculate fitted curve
            fitted_curve = exp_decay(time_points, *popt)
            
            # Calculate R-squared
            residuals = energy_time_series - fitted_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((energy_time_series - np.mean(energy_time_series))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store results
            self.system_equilibration_time = equilibration_time
            self.system_equilibration_params = popt
            self.system_equilibration_r_squared = r_squared
            self.system_fitted_curve = fitted_curve
            
            logger.info(f"System equilibration time: {equilibration_time:.3f} ps, R² = {r_squared:.4f}")
            
            self.profiler.stop()
            return equilibration_time, popt
            
        except Exception as e:
            logger.error(f"System equilibration curve fitting failed: {str(e)}")
            self.profiler.stop()
            return None, None
    
    def analyze_mode_equilibration(self, mode_energies_time_series, frequencies, 
                               time_points=None, freq_bins=None):
        """
        Analyze equilibration of different frequency modes
        
        Parameters:
            mode_energies_time_series: Time series of mode energies [times, modes]
            frequencies: Mode frequencies in THz
            time_points: Time points corresponding to energy values
            freq_bins: Number of frequency bins for analysis
            
        Returns:
            bin_equilibration_times: Equilibration time for each frequency bin
            bin_frequencies: Center frequency of each bin
        """
        self.profiler.start("analyze_mode_equilibration")
        
        # Create time array if not provided
        if time_points is None:
            time_points = np.arange(mode_energies_time_series.shape[0]) * self.timestep
        
        # Set frequency bins if not provided
        if freq_bins is None:
            freq_bins = self.freq_bins
        
        # Get dimensions
        n_times, n_modes = mode_energies_time_series.shape
        
        # Create frequency bins
        freq_max = np.max(frequencies)
        freq_edges = np.linspace(0, freq_max, freq_bins + 1)
        bin_frequencies = (freq_edges[:-1] + freq_edges[1:]) / 2
        
        # Calculate bin indices
        bin_indices = [None] * freq_bins
        for i in range(freq_bins):
            if i < freq_bins - 1:
                bin_indices[i] = (frequencies >= freq_edges[i]) & (frequencies < freq_edges[i+1])
            else:
                bin_indices[i] = (frequencies >= freq_edges[i]) & (frequencies <= freq_edges[i+1])
        
        # Calculate average energy for each bin
        bin_energies = np.zeros((n_times, freq_bins))
        bin_mode_counts = np.zeros(freq_bins, dtype=int)
        
        for b in range(freq_bins):
            bin_mode_counts[b] = np.sum(bin_indices[b])
            if bin_mode_counts[b] > 0:
                bin_energies[:, b] = np.mean(mode_energies_time_series[:, bin_indices[b]], axis=1)
        
        # Fit exponential decay for each bin
        bin_equilibration_times = np.zeros(freq_bins)
        bin_equilibration_params = np.zeros((freq_bins, 3))  # A, tau, E_inf
        bin_r_squared = np.zeros(freq_bins)
        bin_fitted_curves = np.zeros((n_times, freq_bins))
        
        # Define exponential decay function for fitting
        def exp_decay(t, A, tau, E_inf):
            return A * np.exp(-t / tau) + E_inf
        
        # Fit each bin
        for b in range(freq_bins):
            if bin_mode_counts[b] == 0:
                continue
                
            try:
                # Get energy data for this bin
                e_data = bin_energies[:, b]
                
                # Initial guesses
                E_inf_guess = np.mean(e_data[-int(n_times/5):])  # average of last 20%
                A_guess = e_data[0] - E_inf_guess
                tau_guess = n_times * self.timestep / 5
                
                # Fit exponential decay
                popt, _ = scipy.optimize.curve_fit(
                    exp_decay, time_points, e_data,
                    p0=[A_guess, tau_guess, E_inf_guess],
                    bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=5000
                )
                
                # Store parameters
                bin_equilibration_params[b] = popt
                bin_equilibration_times[b] = popt[1]  # tau
                
                # Calculate fitted curve
                bin_fitted_curves[:, b] = exp_decay(time_points, *popt)
                
                # Calculate R-squared
                residuals = e_data - bin_fitted_curves[:, b]
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((e_data - np.mean(e_data))**2)
                bin_r_squared[b] = 1 - (ss_res / ss_tot)
                
            except Exception as e:
                logger.warning(f"Fitting failed for bin {b} ({bin_frequencies[b]:.2f} THz): {str(e)}")
                
                # If fitting fails, use time to reach 1/e of initial deviation
                try:
                    e_data = bin_energies[:, b]
                    e_final = np.mean(e_data[-int(n_times/5):])
                    initial_dev = e_data[0] - e_final
                    target = e_final + initial_dev / np.e
                    
                    # Find first time point below target
                    if initial_dev > 0:
                        idx = np.argmax(e_data <= target)
                    else:
                        idx = np.argmax(e_data >= target)
                    
                    if idx > 0:
                        bin_equilibration_times[b] = time_points[idx]
                    else:
                        bin_equilibration_times[b] = time_points[-1]
                except:
                    bin_equilibration_times[b] = 0
        
        # Store results
        self.bin_frequencies = bin_frequencies
        self.bin_mode_counts = bin_mode_counts
        self.bin_energies = bin_energies
        self.bin_equilibration_times = bin_equilibration_times
        self.bin_equilibration_params = bin_equilibration_params
        self.bin_r_squared = bin_r_squared
        self.bin_fitted_curves = bin_fitted_curves
        
        self.profiler.stop()
        return bin_equilibration_times, bin_frequencies
    
    def calculate_time_window_distributions(self, mode_energies_time_series, frequencies, 
                                         n_windows=None, temperature=None):
        """
        Calculate energy distribution in different time windows
        
        Parameters:
            mode_energies_time_series: Time series of mode energies [times, modes]
            frequencies: Mode frequencies in THz
            n_windows: Number of time windows
            temperature: System temperature in K
            
        Returns:
            window_distributions: Energy distributions for each window
            window_times: Center time for each window
        """
        self.profiler.start("calculate_time_window_distributions")
        
        # Use provided parameters or defaults
        if n_windows is None:
            n_windows = self.n_time_windows
        
        if temperature is None:
            temperature = self.temperature
        
        # Get dimensions
        n_times, n_modes = mode_energies_time_series.shape
        
        # Create time windows
        window_size = n_times // n_windows
        window_times = np.zeros(n_windows)
        
        # Calculate Bose-Einstein distribution for reference
        from ..utils.constants import calc_bose_einstein
        be_occupation = calc_bose_einstein(frequencies, temperature)
        
        # Calculate energy normalized by hbar*omega
        # E/(hbar*omega) = n + 1/2
        normalized_energies = np.zeros_like(mode_energies_time_series)
        
        # Avoid zero frequency modes
        nonzero = frequencies > 0
        
        # Convert frequency to angular frequency (2πf)
        omega = 2 * np.pi * frequencies[nonzero] * 1e12  # convert to Hz
        
        # Calculate ℏω in J
        hbar_omega = HBAR * omega
        
        # Convert to eV
        # 1 J = 6.241509e18 eV
        hbar_omega_ev = hbar_omega / 1.602176634e-19
        
        # Normalize energies
        for t in range(n_times):
            normalized_energies[t, nonzero] = mode_energies_time_series[t, nonzero] / hbar_omega_ev
        
        # Calculate occupation (n = E/(hbar*omega) - 1/2)
        occupation = normalized_energies - 0.5
        
        # Calculate distributions for each window
        window_distributions = np.zeros((n_windows, n_modes))
        window_occupations = np.zeros((n_windows, n_modes))
        window_deviations = np.zeros((n_windows, n_modes))
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, n_times)
            
            # Calculate center time
            window_times[w] = (start_idx + (end_idx - start_idx) / 2) * self.timestep
            
            # Average energy in this window
            window_distributions[w] = np.mean(mode_energies_time_series[start_idx:end_idx], axis=0)
            
            # Average occupation
            window_occupations[w] = np.mean(occupation[start_idx:end_idx], axis=0)
            
            # Calculate deviation from BE distribution
            nonzero_be = be_occupation > 0
            window_deviations[w, nonzero_be] = window_occupations[w, nonzero_be] / be_occupation[nonzero_be]
        
        # Store results
        self.window_times = window_times
        self.window_distributions = window_distributions
        self.window_occupations = window_occupations
        self.window_deviations = window_deviations
        self.bose_einstein_reference = be_occupation
        
        self.profiler.stop()
        return window_distributions, window_times
    
    def calculate_energy_relaxation_spectrum(self, mode_energies_time_series, frequencies, 
                                         time_points=None):
        """
        Calculate energy relaxation spectrum (time constant vs frequency)
        
        Parameters:
            mode_energies_time_series: Time series of mode energies [times, modes]
            frequencies: Mode frequencies in THz
            time_points: Time points corresponding to energy values
            
        Returns:
            time_constants: Time constants for each mode
            r_squared: R-squared values for each fit
        """
        self.profiler.start("calculate_energy_relaxation_spectrum")
        
        # Create time array if not provided
        if time_points is None:
            time_points = np.arange(mode_energies_time_series.shape[0]) * self.timestep
        
        # Get dimensions
        n_times, n_modes = mode_energies_time_series.shape
        
        # Initialize arrays for results
        time_constants = np.zeros(n_modes)
        amplitudes = np.zeros(n_modes)
        equilibrium_values = np.zeros(n_modes)
        r_squared = np.zeros(n_modes)
        
        # Define exponential decay function for fitting
        def exp_decay(t, A, tau, E_inf):
            return A * np.exp(-t / tau) + E_inf
        
        # Fit each mode
        valid_fits = 0
        for i in range(n_modes):
            # Skip zero frequency modes
            if frequencies[i] <= 0:
                continue
                
            try:
                # Get energy data for this mode
                e_data = mode_energies_time_series[:, i]
                
                # Check if there's variation in the data
                if np.std(e_data) < 1e-10:
                    continue
                
                # Initial guesses
                E_inf_guess = np.mean(e_data[-int(n_times/5):])  # average of last 20%
                A_guess = e_data[0] - E_inf_guess
                tau_guess = n_times * self.timestep / 5
                
                # Fit exponential decay
                popt, _ = scipy.optimize.curve_fit(
                    exp_decay, time_points, e_data,
                    p0=[A_guess, tau_guess, E_inf_guess],
                    bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=5000
                )
                
                # Store parameters
                amplitudes[i] = popt[0]
                time_constants[i] = popt[1]  # tau
                equilibrium_values[i] = popt[2]  # E_inf
                
                # Calculate fitted curve
                fitted = exp_decay(time_points, *popt)
                
                # Calculate R-squared
                residuals = e_data - fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((e_data - np.mean(e_data))**2)
                r_squared[i] = 1 - (ss_res / ss_tot)
                
                valid_fits += 1
                
            except Exception as e:
                # If fitting fails, set to 0
                time_constants[i] = 0
                r_squared[i] = 0
        
        # Store results
        self.time_constants = time_constants
        self.amplitudes = amplitudes
        self.equilibrium_values = equilibrium_values
        self.fit_r_squared = r_squared
        
        logger.info(f"Successfully fitted {valid_fits}/{n_modes} modes")
        
        self.profiler.stop()
        return time_constants, r_squared
    
    def get_equilibration_analysis_summary(self):
        """
        Get a summary of equilibration analysis results
        
        Returns:
            summary: Dictionary with analysis results
        """
        summary = {
            "system_equilibration_time": self.system_equilibration_time if hasattr(self, 'system_equilibration_time') else None,
            "system_equilibration_params": self.system_equilibration_params if hasattr(self, 'system_equilibration_params') else None,
            "bin_equilibration_times": self.bin_equilibration_times if hasattr(self, 'bin_equilibration_times') else None,
            "bin_frequencies": self.bin_frequencies if hasattr(self, 'bin_frequencies') else None,
            "window_times": self.window_times if hasattr(self, 'window_times') else None,
            "window_deviations": self.window_deviations if hasattr(self, 'window_deviations') else None,
            "time_constants": self.time_constants if hasattr(self, 'time_constants') else None,
            "fit_r_squared": self.fit_r_squared if hasattr(self, 'fit_r_squared') else None,
            "temperature": self.temperature
        }
        
        return summary 