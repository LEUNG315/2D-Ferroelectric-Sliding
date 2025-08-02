#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal phonon analysis module
"""

import numpy as np
import logging
import scipy.signal
from ..utils.timing import Timer, TimeProfiler
from ..utils.constants import KB, HBAR
from ..utils.helpers import smooth_data

# Setup logging
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Temporal phonon analysis for time-resolved studies"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the temporal analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("TemporalAnalyzer")
        
        # Default parameters
        self.freq_max = 30.0  # THz
        self.freq_points = 1000
        self.sigma = 0.1  # THz
        self.window_size = 500  # frames
        self.window_step = 100  # frames
        self.timestep = 0.001  # ps
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.freq_max = coordinator.get_config('freq_max', self.freq_max)
            self.freq_points = coordinator.get_config('freq_points', self.freq_points)
            self.sigma = coordinator.get_config('sigma', self.sigma)
            self.window_size = coordinator.get_config('window_size', self.window_size)
            self.window_step = coordinator.get_config('window_step', self.window_step)
            self.timestep = coordinator.get_config('timestep', self.timestep)
        
        # Analysis results
        self.time_resolved_dos = None
        self.dos_evolution = None
        self.time_points = None
        self.frequencies = None
        
        logger.debug("TemporalAnalyzer initialized")
    
    def calculate_time_resolved_dos(self, velocities, window_size=None, window_step=None, 
                                  freq_max=None, freq_points=None, sigma=None):
        """
        Calculate time-resolved phonon density of states
        
        Parameters:
            velocities: Velocity data [frames, atoms, 3]
            window_size: Analysis window size in frames
            window_step: Step size between windows in frames
            freq_max: Maximum frequency in THz
            freq_points: Number of frequency points
            sigma: Gaussian smoothing width in THz
            
        Returns:
            frequencies: Frequency array in THz
            dos_evolution: Time-resolved DOS [windows, freq_points]
            time_points: Center time for each window in ps
        """
        self.profiler.start("calculate_time_resolved_dos")
        
        # Use provided parameters or defaults
        if window_size is None:
            window_size = self.window_size
        
        if window_step is None:
            window_step = self.window_step
        
        if freq_max is None:
            freq_max = self.freq_max
        
        if freq_points is None:
            freq_points = self.freq_points
        
        if sigma is None:
            sigma = self.sigma
        
        # Check input dimensions
        if len(velocities.shape) == 3:
            n_frames, n_atoms, n_dims = velocities.shape
        else:
            # Assume [frames, atoms*3]
            n_frames = velocities.shape[0]
            n_atoms = velocities.shape[1] // 3
            n_dims = 3
            velocities = velocities.reshape(n_frames, n_atoms, n_dims)
        
        # Calculate number of windows
        n_windows = 1 + (n_frames - window_size) // window_step
        
        if n_windows <= 0:
            logger.error(f"Window size ({window_size}) larger than available frames ({n_frames})")
            self.profiler.stop()
            return None, None, None
        
        # Create frequency array
        frequencies = np.linspace(0, freq_max, freq_points)
        
        # Allocate arrays for results
        dos_evolution = np.zeros((n_windows, freq_points))
        time_points = np.zeros(n_windows)
        
        # Process each window
        for w in range(n_windows):
            # Get window start and end
            start_idx = w * window_step
            end_idx = start_idx + window_size
            
            # Calculate center time
            center_time = (start_idx + window_size / 2) * self.timestep
            time_points[w] = center_time
            
            # Get velocities for this window
            v_window = velocities[start_idx:end_idx]
            
            # Calculate VACF
            vacf = np.zeros(window_size // 2)
            max_lag = len(vacf)
            
            # Compute autocorrelation
            for i in range(max_lag):
                # Sum over all atoms and dimensions
                vacf[i] = np.sum(v_window[:window_size-i] * v_window[i:], axis=(0, 1, 2))
                
                # Normalize by number of frames
                vacf[i] /= (window_size - i)
            
            # Normalize VACF
            if vacf[0] > 0:
                vacf /= vacf[0]
            
            # Calculate DOS from VACF using FFT
            fft_result = np.fft.rfft(vacf)
            fft_freq = np.fft.rfftfreq(window_size // 2, d=self.timestep)
            
            # Convert frequency to THz
            fft_freq_thz = fft_freq * 1000  # ps^-1 to THz
            
            # Interpolate to regular frequency grid
            import scipy.interpolate
            dos = np.zeros_like(frequencies)
            valid_indices = fft_freq_thz <= freq_max
            
            if np.sum(valid_indices) > 1:
                try:
                    interp = scipy.interpolate.interp1d(
                        fft_freq_thz[valid_indices], 
                        np.abs(fft_result[valid_indices]),
                        bounds_error=False, 
                        fill_value=0.0
                    )
                    dos = interp(frequencies)
                except:
                    logger.warning(f"Interpolation failed for window {w}")
            
            # Apply Gaussian smoothing if sigma > 0
            if sigma > 0:
                # Convert sigma to number of points
                sigma_points = sigma / (freq_max / freq_points)
                
                # Ensure odd window size for gaussian filter
                window_length = int(6 * sigma_points)
                if window_length % 2 == 0:
                    window_length += 1
                
                if window_length > 3:
                    dos = scipy.ndimage.gaussian_filter1d(dos, sigma=sigma_points)
            
            # Normalize DOS
            if np.max(dos) > 0:
                dos /= np.max(dos)
            
            # Store in evolution array
            dos_evolution[w] = dos
        
        # Store results
        self.frequencies = frequencies
        self.dos_evolution = dos_evolution
        self.time_points = time_points
        
        self.profiler.stop()
        return frequencies, dos_evolution, time_points
    
    def calculate_dos_difference(self, reference_idx=0):
        """
        Calculate difference between DOS at each time point and a reference DOS
        
        Parameters:
            reference_idx: Index of reference DOS (default: first window)
            
        Returns:
            dos_diff: Difference in DOS [windows, freq_points]
        """
        self.profiler.start("calculate_dos_difference")
        
        # Check if we have calculated time-resolved DOS
        if self.dos_evolution is None:
            logger.error("Time-resolved DOS not calculated. Call calculate_time_resolved_dos first.")
            self.profiler.stop()
            return None
        
        # Get reference DOS
        reference_dos = self.dos_evolution[reference_idx]
        
        # Calculate difference
        dos_diff = self.dos_evolution - reference_dos[np.newaxis, :]
        
        # Store results
        self.dos_difference = dos_diff
        
        self.profiler.stop()
        return dos_diff
    
    def calculate_dos_evolution_metrics(self):
        """
        Calculate metrics to characterize DOS evolution
        
        Returns:
            metrics: Dictionary of evolution metrics
        """
        self.profiler.start("calculate_dos_evolution_metrics")
        
        # Check if we have calculated time-resolved DOS
        if self.dos_evolution is None:
            logger.error("Time-resolved DOS not calculated. Call calculate_time_resolved_dos first.")
            self.profiler.stop()
            return None
        
        # Get dimensions
        n_windows, n_freq = self.dos_evolution.shape
        
        # Calculate metrics
        metrics = {
            'peak_heights': np.zeros((n_windows, 3)),  # Track top 3 peaks
            'peak_freqs': np.zeros((n_windows, 3)),
            'low_freq_intensity': np.zeros(n_windows),   # 0-10 THz
            'mid_freq_intensity': np.zeros(n_windows),   # 10-20 THz
            'high_freq_intensity': np.zeros(n_windows),  # 20+ THz
            'total_intensity': np.zeros(n_windows),
            'mean_frequency': np.zeros(n_windows),
            'std_frequency': np.zeros(n_windows)
        }
        
        # Define frequency bands
        low_idx = (self.frequencies <= 10.0)
        mid_idx = (self.frequencies > 10.0) & (self.frequencies <= 20.0)
        high_idx = (self.frequencies > 20.0)
        
        # Calculate metrics for each window
        for w in range(n_windows):
            dos = self.dos_evolution[w]
            
            # Find peaks
            from scipy.signal import find_peaks
            peak_indices, _ = find_peaks(dos, height=0.1)
            
            # Sort peaks by height
            peak_heights = dos[peak_indices]
            sorted_indices = np.argsort(peak_heights)[::-1]  # descending
            
            # Store top 3 peaks (if available)
            for i in range(min(3, len(peak_indices))):
                if i < len(sorted_indices):
                    idx = peak_indices[sorted_indices[i]]
                    metrics['peak_heights'][w, i] = dos[idx]
                    metrics['peak_freqs'][w, i] = self.frequencies[idx]
            
            # Calculate intensities
            metrics['low_freq_intensity'][w] = np.sum(dos[low_idx])
            metrics['mid_freq_intensity'][w] = np.sum(dos[mid_idx])
            metrics['high_freq_intensity'][w] = np.sum(dos[high_idx])
            metrics['total_intensity'][w] = np.sum(dos)
            
            # Calculate mean and std frequency (weighted by DOS)
            if np.sum(dos) > 0:
                metrics['mean_frequency'][w] = np.sum(self.frequencies * dos) / np.sum(dos)
                metrics['std_frequency'][w] = np.sqrt(
                    np.sum(dos * (self.frequencies - metrics['mean_frequency'][w])**2) / np.sum(dos)
                )
        
        # Store results
        self.dos_evolution_metrics = metrics
        
        self.profiler.stop()
        return metrics
    
    def analyze_equilibration_process(self, energy_time_series, time_points=None):
        """
        Analyze equilibration process from energy time series
        
        Parameters:
            energy_time_series: Energy over time [times]
            time_points: Time points corresponding to energy values
            
        Returns:
            equilibration_time: Estimated equilibration time
            equilibration_params: Fitted parameters
        """
        self.profiler.start("analyze_equilibration_process")
        
        # Create time array if not provided
        if time_points is None:
            time_points = np.arange(len(energy_time_series)) * self.timestep
        
        # Define exponential decay function for fitting
        def exp_decay(t, A, tau, E_inf):
            return A * np.exp(-t / tau) + E_inf
        
        # Fit exponential decay
        try:
            import scipy.optimize
            
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
            self.equilibration_time = equilibration_time
            self.equilibration_params = popt
            self.equilibration_r_squared = r_squared
            self.fitted_equilibration_curve = fitted_curve
            
            logger.info(f"Equilibration time: {equilibration_time:.3f} ps, R² = {r_squared:.4f}")
            
            self.profiler.stop()
            return equilibration_time, popt
            
        except Exception as e:
            logger.error(f"Equilibration curve fitting failed: {str(e)}")
            self.profiler.stop()
            return None, None
    
    def calculate_frequency_band_evolution(self, mode_energies_time_series, frequencies, 
                                        freq_bands=None, normalize=True):
        """
        Calculate evolution of energy in different frequency bands
        
        Parameters:
            mode_energies_time_series: Time series of mode energies [times, modes]
            frequencies: Mode frequencies in THz
            freq_bands: List of (min, max) frequency bands in THz
            normalize: Whether to normalize by number of modes in each band
            
        Returns:
            band_energies: Energy in each band over time [times, bands]
            band_definitions: Description of frequency bands
        """
        self.profiler.start("calculate_frequency_band_evolution")
        
        # Default frequency bands if not provided
        if freq_bands is None:
            freq_bands = [
                (0, 5),    # Low frequency (0-5 THz)
                (5, 15),   # Mid frequency (5-15 THz)
                (15, 30),  # High frequency (15-30 THz)
                (0, 30)    # All frequencies
            ]
        
        # Get dimensions
        n_times, n_modes = mode_energies_time_series.shape
        n_bands = len(freq_bands)
        
        # Create band_energies array
        band_energies = np.zeros((n_times, n_bands))
        
        # Create band_definitions list
        band_definitions = []
        band_mode_counts = np.zeros(n_bands, dtype=int)
        
        # Process each band
        for b, (freq_min, freq_max) in enumerate(freq_bands):
            # Find modes in this frequency band
            band_indices = (frequencies >= freq_min) & (frequencies <= freq_max)
            band_mode_counts[b] = np.sum(band_indices)
            
            # Skip if no modes in this band
            if band_mode_counts[b] == 0:
                logger.warning(f"No modes found in frequency band {freq_min}-{freq_max} THz")
                continue
            
            # Sum energy for modes in this band
            band_energies[:, b] = np.sum(mode_energies_time_series[:, band_indices], axis=1)
            
            # Normalize if requested
            if normalize:
                band_energies[:, b] /= band_mode_counts[b]
            
            # Create band definition
            band_definitions.append(f"{freq_min}-{freq_max} THz ({band_mode_counts[b]} modes)")
        
        # Store results
        self.band_energies = band_energies
        self.band_definitions = band_definitions
        self.band_mode_counts = band_mode_counts
        
        self.profiler.stop()
        return band_energies, band_definitions
    
    def get_temporal_analysis_summary(self):
        """
        Get a summary of temporal analysis results
        
        Returns:
            summary: Dictionary with analysis results
        """
        summary = {
            "time_points": self.time_points,
            "frequencies": self.frequencies,
            "dos_evolution": self.dos_evolution,
            "dos_evolution_metrics": self.dos_evolution_metrics if hasattr(self, 'dos_evolution_metrics') else None,
            "band_energies": self.band_energies if hasattr(self, 'band_energies') else None,
            "band_definitions": self.band_definitions if hasattr(self, 'band_definitions') else None,
            "equilibration_time": self.equilibration_time if hasattr(self, 'equilibration_time') else None,
        }
        
        return summary 