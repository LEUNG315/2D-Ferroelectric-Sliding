#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thermal conductivity analysis module
"""

import numpy as np
import logging
from scipy import integrate
from ..utils.timing import Timer, TimeProfiler
from ..utils.constants import KB
from ..utils.helpers import smooth_data

# Setup logging
logger = logging.getLogger(__name__)

class ThermalAnalyzer:
    """Thermal conductivity analysis for materials"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the thermal analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("ThermalAnalyzer")
        
        # Default parameters
        self.temperature = 300.0  # K
        self.volume = None  # Å³
        self.correlation_time = 5.0  # ps
        self.timestep = 0.001  # ps
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.temperature = coordinator.get_config('temperature', self.temperature)
            self.timestep = coordinator.get_config('timestep', self.timestep)
        
        # Analysis results
        self.heatflux_acf = None
        self.thermal_conductivity = None
        self.interlayer_conductance = None
        
        logger.debug("ThermalAnalyzer initialized")
    
    def calculate_heatflux_autocorrelation(self, heatflux, max_correlation_time=None):
        """
        Calculate heat flux autocorrelation function
        
        Parameters:
            heatflux: Heat flux data [frames, 3]
            max_correlation_time: Maximum correlation time in ps
            
        Returns:
            times: Time array
            acf: Autocorrelation function
        """
        self.profiler.start("calculate_heatflux_acf")
        
        # Check input dimensions
        if len(heatflux.shape) != 2 or heatflux.shape[1] != 3:
            logger.error(f"Heat flux data should have shape [frames, 3], got {heatflux.shape}")
            self.profiler.stop()
            return None, None
        
        # Determine number of frames for correlation
        n_frames = len(heatflux)
        
        if max_correlation_time is None:
            max_correlation_time = self.correlation_time
        
        max_lag = min(n_frames // 2, int(max_correlation_time / self.timestep))
        
        # Calculate autocorrelation for each dimension
        acf = np.zeros(max_lag)
        
        # Compute autocorrelation using numpy's correlate function
        for dim in range(3):
            # Normalize by subtracting mean
            flux_norm = heatflux[:, dim] - np.mean(heatflux[:, dim])
            
            # Autocorrelation
            dim_acf = np.correlate(flux_norm, flux_norm, mode='full')
            
            # Extract positive lag times and normalize
            dim_acf = dim_acf[len(dim_acf)//2:len(dim_acf)//2 + max_lag]
            
            # Normalize by zero lag
            if dim_acf[0] > 0:
                dim_acf /= dim_acf[0]
            
            # Add to total ACF
            acf += dim_acf
        
        # Average over directions
        acf /= 3.0
        
        # Create time array
        times = np.arange(max_lag) * self.timestep
        
        # Store results
        self.heatflux_acf = acf
        self.heatflux_acf_times = times
        
        self.profiler.stop()
        return times, acf
    
    def calculate_thermal_conductivity(self, heatflux=None, temperature=None, volume=None, 
                                    max_correlation_time=None, smoothing=True):
        """
        Calculate thermal conductivity using Green-Kubo method
        
        Parameters:
            heatflux: Heat flux data [frames, 3]
            temperature: System temperature in K
            volume: System volume in Å³
            max_correlation_time: Maximum correlation time in ps
            smoothing: Whether to apply smoothing to the ACF
            
        Returns:
            kappa: Thermal conductivity in W/mK
            kappa_components: Directional components of thermal conductivity
        """
        self.profiler.start("calculate_thermal_conductivity")
        
        # Use provided parameters or defaults
        if temperature is None:
            temperature = self.temperature
        
        if volume is None:
            if self.volume is None:
                # Try to get volume from coordinator
                if self.coordinator and hasattr(self.coordinator, 'get_system_volume'):
                    volume = self.coordinator.get_system_volume()
                    if volume is None:
                        logger.error("System volume not available. Please provide volume parameter.")
                        self.profiler.stop()
                        return None, None
                else:
                    logger.error("System volume not available. Please provide volume parameter.")
                    self.profiler.stop()
                    return None, None
            else:
                volume = self.volume
        
        # Store volume for future use
        self.volume = volume
        
        # Use provided heatflux data or get from coordinator
        if heatflux is None:
            if self.coordinator and hasattr(self.coordinator, 'heatflux_data'):
                heatflux = self.coordinator.heatflux_data.get('data')
                if heatflux is None:
                    logger.error("Heat flux data not available. Please provide heatflux parameter.")
                    self.profiler.stop()
                    return None, None
            else:
                logger.error("Heat flux data not available. Please provide heatflux parameter.")
                self.profiler.stop()
                return None, None
        
        # Calculate heatflux autocorrelation if needed
        if self.heatflux_acf is None or max_correlation_time is not None:
            times, acf = self.calculate_heatflux_autocorrelation(
                heatflux, max_correlation_time)
        else:
            times, acf = self.heatflux_acf_times, self.heatflux_acf
        
        # Apply smoothing if requested
        if smoothing:
            acf = smooth_data(acf, window_size=5)
        
        # Green-Kubo formula: kappa = V/(3kT) * ∫ <J(0)·J(t)> dt
        # Convert units to SI:
        # - volume from Å³ to m³
        # - time from ps to s
        # - temperature in K
        
        # Volume in m³
        volume_si = volume * 1e-30  # Å³ to m³
        
        # Calculate integral of ACF
        integral = integrate.cumulative_trapezoid(acf, times, initial=0)
        
        # Convert to thermal conductivity
        # Factor = V/(3kT), where k is Boltzmann constant
        # KB = 1.380649e-23  # J/K
        boltzmann = KB  # J/K
        
        # Scale factor for Green-Kubo formula
        scale_factor = volume_si / (3 * boltzmann * temperature)
        
        # Account for time unit conversion (ps to s)
        scale_factor *= 1e-12  # s/ps
        
        # Calculate thermal conductivity in W/(m·K)
        kappa = scale_factor * integral
        
        # Calculate directional components
        kappa_components = np.zeros((len(times), 3))
        
        for dim in range(3):
            flux_norm = heatflux[:, dim] - np.mean(heatflux[:, dim])
            dim_acf = np.correlate(flux_norm, flux_norm, mode='full')
            dim_acf = dim_acf[len(dim_acf)//2:len(dim_acf)//2 + len(times)]
            
            if dim_acf[0] > 0:
                dim_acf /= dim_acf[0]
            
            if smoothing:
                dim_acf = smooth_data(dim_acf, window_size=5)
            
            dim_integral = integrate.cumulative_trapezoid(dim_acf, times, initial=0)
            kappa_components[:, dim] = scale_factor * dim_integral
        
        # Store results
        self.thermal_conductivity = kappa
        self.thermal_conductivity_times = times
        self.kappa_components = kappa_components
        
        # Return final value of thermal conductivity
        final_kappa = kappa[-1]
        final_kappa_components = kappa_components[-1, :]
        
        self.profiler.stop()
        return final_kappa, final_kappa_components
    
    def calculate_interlayer_conductance(self, energy_transfer, temperature_difference, 
                                      contact_area, time_window=None):
        """
        Calculate thermal boundary conductance between layers
        
        Parameters:
            energy_transfer: Energy transfer rate between layers in eV/ps
            temperature_difference: Temperature difference between layers in K
            contact_area: Contact area in Å²
            time_window: Time window for averaging in ps
            
        Returns:
            g: Thermal boundary conductance in W/(m²K)
        """
        self.profiler.start("calculate_interlayer_conductance")
        
        # Convert energy transfer from eV/ps to W
        # 1 eV/ps = 1.602e-19 J / 1e-12 s = 1.602e-7 W
        energy_transfer_w = energy_transfer * 1.602e-7
        
        # Convert contact area from Å² to m²
        contact_area_m2 = contact_area * 1e-20
        
        # Calculate conductance (W/(m²K))
        g = energy_transfer_w / (temperature_difference * contact_area_m2)
        
        # Store results
        self.interlayer_conductance = g
        
        self.profiler.stop()
        return g
    
    def analyze_polarization_heatflux_correlation(self, polarization, heatflux, max_lag=None):
        """
        Analyze correlation between polarization and heat flux
        
        Parameters:
            polarization: Polarization data [frames, 3]
            heatflux: Heat flux data [frames, 3]
            max_lag: Maximum lag time in frames
            
        Returns:
            correlation_coeffs: Correlation coefficients for each direction
            cross_correlation: Cross-correlation function
        """
        self.profiler.start("analyze_polarization_heatflux")
        
        # Check input dimensions
        if len(polarization.shape) != 2 or polarization.shape[1] != 3:
            logger.error(f"Polarization data should have shape [frames, 3], got {polarization.shape}")
            self.profiler.stop()
            return None, None
        
        if len(heatflux.shape) != 2 or heatflux.shape[1] != 3:
            logger.error(f"Heat flux data should have shape [frames, 3], got {heatflux.shape}")
            self.profiler.stop()
            return None, None
        
        # Ensure same number of frames
        n_frames = min(len(polarization), len(heatflux))
        polarization = polarization[:n_frames]
        heatflux = heatflux[:n_frames]
        
        # Determine maximum lag time
        if max_lag is None:
            max_lag = n_frames // 4
        
        # Calculate correlation coefficient for each direction
        correlation_coeffs = np.zeros(3)
        for dim in range(3):
            correlation_coeffs[dim] = np.corrcoef(polarization[:, dim], heatflux[:, dim])[0, 1]
        
        # Calculate cross-correlation function
        cross_correlation = np.zeros((2*max_lag+1, 3))
        lags = np.arange(-max_lag, max_lag+1)
        
        for dim in range(3):
            # Normalize data
            p_norm = (polarization[:, dim] - np.mean(polarization[:, dim])) / np.std(polarization[:, dim])
            h_norm = (heatflux[:, dim] - np.mean(heatflux[:, dim])) / np.std(heatflux[:, dim])
            
            # Calculate cross-correlation
            cross_corr = np.correlate(p_norm, h_norm, mode='full')
            
            # Extract relevant part and normalize
            mid = len(cross_corr) // 2
            cross_correlation[:, dim] = cross_corr[mid-max_lag:mid+max_lag+1] / n_frames
        
        # Store results
        self.polarization_heatflux_correlation = correlation_coeffs
        self.polarization_heatflux_cross_correlation = cross_correlation
        self.correlation_lags = lags
        
        self.profiler.stop()
        return correlation_coeffs, cross_correlation
    
    def get_thermal_analysis_summary(self):
        """
        Get a summary of thermal analysis results
        
        Returns:
            summary: Dictionary with analysis results
        """
        summary = {
            "thermal_conductivity": self.thermal_conductivity[-1] if self.thermal_conductivity is not None else None,
            "kappa_components": self.kappa_components[-1] if self.kappa_components is not None else None,
            "interlayer_conductance": self.interlayer_conductance,
            "polarization_heatflux_correlation": self.polarization_heatflux_correlation if hasattr(self, 'polarization_heatflux_correlation') else None,
            "temperature": self.temperature,
            "volume": self.volume
        }
        
        return summary 