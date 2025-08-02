#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core phonon analysis module
"""

import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.optimize
import logging
from ..utils.timing import Timer, TimeProfiler
from ..utils.constants import KB, HBAR, temp_to_energy, energy_to_temp, freq_to_energy, energy_to_freq

# Setup logging
logger = logging.getLogger(__name__)

class PhononAnalyzer:
    """Core phonon analysis functionality"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the phonon analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("PhononAnalyzer")
        
        # Default parameters
        self.temperature = 300.0
        self.freq_max = 30.0
        self.freq_points = 1000
        self.sigma = 0.1
        self.timestep = 0.001  # ps
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.temperature = coordinator.get_config('temperature', self.temperature)
            self.freq_max = coordinator.get_config('freq_max', self.freq_max)
            self.freq_points = coordinator.get_config('freq_points', self.freq_points)
            self.sigma = coordinator.get_config('sigma', self.sigma)
            self.timestep = coordinator.get_config('timestep', self.timestep)
        
        # Analysis results
        self.frequencies = None
        self.dos = None
        self.vacf = None
        self.mode_energies = None
        self.mode_occupations = None
        
        logger.debug("PhononAnalyzer initialized")
    
    def calculate_velocity_autocorrelation(self, velocities, max_lag=None, normalize=True):
        """
        Calculate velocity autocorrelation function
        
        Parameters:
            velocities: Array of velocity data [frames, atoms, 3]
            max_lag: Maximum lag time (frames)
            normalize: Whether to normalize the VACF
            
        Returns:
            vacf: Velocity autocorrelation function
        """
        self.profiler.start("calculate_vacf")
        
        # Reshape velocities if needed
        if len(velocities.shape) == 3:
            n_frames, n_atoms, n_dims = velocities.shape
        else:
            # Assume [frames, atoms*3]
            n_frames = velocities.shape[0]
            n_atoms = velocities.shape[1] // 3
            n_dims = 3
            velocities = velocities.reshape(n_frames, n_atoms, n_dims)
        
        # Set default max_lag if not provided
        if max_lag is None:
            max_lag = n_frames // 2
        
        # Calculate VACF
        vacf = np.zeros(max_lag)
        
        # Compute autocorrelation
        for i in range(max_lag):
            # Sum over all atoms and dimensions
            vacf[i] = np.sum(velocities[:n_frames-i] * velocities[i:], axis=(0, 1, 2))
            
            # Normalize by number of frames
            vacf[i] /= (n_frames - i)
        
        # Normalize by VACF[0]
        if normalize and vacf[0] > 0:
            vacf /= vacf[0]
        
        # Store and return
        self.vacf = vacf
        self.profiler.stop()
        
        return vacf
    
    def calculate_dos(self, vacf=None, timestep=None, freq_max=None, freq_points=None, sigma=None):
        """
        Calculate phonon density of states from VACF
        
        Parameters:
            vacf: Velocity autocorrelation function (if None, use stored value)
            timestep: Time step in picoseconds (if None, use stored value)
            freq_max: Maximum frequency in THz (if None, use stored value)
            freq_points: Number of frequency points (if None, use stored value)
            sigma: Gaussian smoothing width in THz (if None, use stored value)
            
        Returns:
            freqs: Frequency array in THz
            dos: Density of states
        """
        self.profiler.start("calculate_dos")
        
        # Use stored values if not provided
        if vacf is None:
            vacf = self.vacf
            if vacf is None:
                logger.error("No VACF data available")
                self.profiler.stop()
                return None, None
        
        if timestep is None:
            timestep = self.timestep
        
        if freq_max is None:
            freq_max = self.freq_max
        
        if freq_points is None:
            freq_points = self.freq_points
        
        if sigma is None:
            sigma = self.sigma
        
        # Calculate FFT of the VACF
        n_frames = len(vacf)
        fft_result = np.fft.rfft(vacf)
        fft_freq = np.fft.rfftfreq(n_frames, d=timestep)
        
        # Convert frequency to THz
        fft_freq_thz = fft_freq * 1000  # ps^-1 to THz
        
        # Create frequency grid for output
        freqs = np.linspace(0, freq_max, freq_points)
        
        # Interpolate FFT result to our frequency grid
        # Use absolute value for amplitude
        import scipy.interpolate
        dos = np.zeros_like(freqs)
        valid_indices = fft_freq_thz <= freq_max
        
        if np.sum(valid_indices) > 1:
            # Use cubic spline interpolation if we have enough points
            try:
                interp = scipy.interpolate.CubicSpline(
                    fft_freq_thz[valid_indices], 
                    np.abs(fft_result[valid_indices])
                )
                dos = interp(freqs)
            except:
                # Fall back to linear interpolation
                interp = scipy.interpolate.interp1d(
                    fft_freq_thz[valid_indices], 
                    np.abs(fft_result[valid_indices]),
                    bounds_error=False, 
                    fill_value=0.0
                )
                dos = interp(freqs)
        
        # Apply Gaussian smoothing if sigma > 0
        if sigma > 0:
            # Convert sigma to number of points
            sigma_points = sigma / (freq_max / freq_points)
            
            # Ensure odd window size for gaussian filter
            window_size = int(6 * sigma_points)
            if window_size % 2 == 0:
                window_size += 1
            
            if window_size > 3:
                dos = scipy.ndimage.gaussian_filter1d(dos, sigma=sigma_points)
        
        # Normalize DOS
        if np.max(dos) > 0:
            dos /= np.max(dos)
        
        # Store results
        self.frequencies = freqs
        self.dos = dos
        
        self.profiler.stop()
        return freqs, dos
    
    def calculate_projected_dos(self, velocities, atom_indices, max_lag=None, timestep=None, 
                               freq_max=None, freq_points=None, sigma=None):
        """
        Calculate projected DOS for a subset of atoms
        
        Parameters:
            velocities: Array of velocity data [frames, atoms, 3]
            atom_indices: Indices of atoms to include
            max_lag: Maximum lag time (frames)
            timestep: Time step in picoseconds
            freq_max: Maximum frequency in THz
            freq_points: Number of frequency points
            sigma: Gaussian smoothing width in THz
            
        Returns:
            freqs: Frequency array in THz
            projected_dos: Density of states for selected atoms
        """
        self.profiler.start("calculate_projected_dos")
        
        # Extract velocities for selected atoms
        if len(velocities.shape) == 3:
            # [frames, atoms, 3]
            selected_velocities = velocities[:, atom_indices, :]
        else:
            # Assume [frames, atoms*3]
            n_atoms = velocities.shape[1] // 3
            v_reshape = velocities.reshape(velocities.shape[0], n_atoms, 3)
            selected_velocities = v_reshape[:, atom_indices, :]
        
        # Calculate VACF for selected atoms
        vacf = self.calculate_velocity_autocorrelation(
            selected_velocities, max_lag=max_lag, normalize=True)
        
        # Calculate DOS
        freqs, dos = self.calculate_dos(
            vacf, timestep=timestep, freq_max=freq_max, 
            freq_points=freq_points, sigma=sigma)
        
        self.profiler.stop()
        return freqs, dos
    
    def calculate_directional_dos(self, velocities, directions=None, max_lag=None, timestep=None,
                               freq_max=None, freq_points=None, sigma=None):
        """
        Calculate directional DOS
        
        Parameters:
            velocities: Array of velocity data [frames, atoms, 3]
            directions: List of directions ('x', 'y', 'z', or vectors)
            max_lag: Maximum lag time (frames)
            timestep: Time step in picoseconds
            freq_max: Maximum frequency in THz
            freq_points: Number of frequency points
            sigma: Gaussian smoothing width in THz
            
        Returns:
            freqs: Frequency array in THz
            directional_dos: Dictionary of directional DOS
        """
        self.profiler.start("calculate_directional_dos")
        
        # Use default directions if not provided
        if directions is None:
            directions = ['x', 'y', 'z']
        
        # Convert direction strings to vectors
        direction_vectors = []
        for d in directions:
            if isinstance(d, str):
                if d.lower() == 'x':
                    direction_vectors.append(np.array([1, 0, 0]))
                elif d.lower() == 'y':
                    direction_vectors.append(np.array([0, 1, 0]))
                elif d.lower() == 'z':
                    direction_vectors.append(np.array([0, 0, 1]))
                else:
                    logger.warning(f"Unknown direction '{d}', ignoring")
            else:
                # Assume it's a vector, normalize it
                vec = np.array(d, dtype=float)
                vec /= np.linalg.norm(vec)
                direction_vectors.append(vec)
        
        # Reshape velocities if needed
        if len(velocities.shape) == 3:
            n_frames, n_atoms, n_dims = velocities.shape
        else:
            # Assume [frames, atoms*3]
            n_frames = velocities.shape[0]
            n_atoms = velocities.shape[1] // 3
            n_dims = 3
            velocities = velocities.reshape(n_frames, n_atoms, n_dims)
        
        # Results dictionary
        directional_dos = {}
        
        # Calculate DOS for each direction
        for i, vec in enumerate(direction_vectors):
            # Project velocities onto this direction
            projected_velocities = np.zeros((n_frames, n_atoms, 1))
            for j in range(n_atoms):
                # Dot product with direction vector
                projected_velocities[:, j, 0] = np.sum(velocities[:, j, :] * vec, axis=1)
            
            # Calculate VACF for projected velocities
            vacf = self.calculate_velocity_autocorrelation(
                projected_velocities, max_lag=max_lag, normalize=True)
            
            # Calculate DOS
            freqs, dos = self.calculate_dos(
                vacf, timestep=timestep, freq_max=freq_max, 
                freq_points=freq_points, sigma=sigma)
            
            # Store results
            if i < len(directions):
                # Use original direction name as key
                dir_name = directions[i]
            else:
                # Use index as key
                dir_name = f"dir_{i}"
                
            directional_dos[dir_name] = dos
        
        self.profiler.stop()
        return freqs, directional_dos
    
    def calculate_mass_weighted_frequencies(self, freqs=None, dos=None, mass_weights=None):
        """
        Calculate mass-weighted frequency distribution
        
        Parameters:
            freqs: Frequency array in THz
            dos: Density of states
            mass_weights: Array of atomic masses or weight factor
            
        Returns:
            mass_freqs: Mass-weighted frequency array in THz
            mass_dos: Mass-weighted density of states
        """
        self.profiler.start("calculate_mass_weighted_frequencies")
        
        # Use stored values if not provided
        if freqs is None:
            freqs = self.frequencies
            if freqs is None:
                logger.error("No frequency data available")
                self.profiler.stop()
                return None, None
        
        if dos is None:
            dos = self.dos
            if dos is None:
                logger.error("No DOS data available")
                self.profiler.stop()
                return None, None
        
        # If mass_weights is None, use uniform weights
        if mass_weights is None:
            mass_weights = np.ones_like(freqs)
        
        # Mass-weight the frequencies
        mass_dos = dos * mass_weights
        
        # Normalize
        if np.max(mass_dos) > 0:
            mass_dos /= np.max(mass_dos)
        
        self.profiler.stop()
        return freqs, mass_dos
    
    def calculate_normal_modes(self, positions, velocities, masses=None):
        """
        Calculate normal modes from MD trajectory
        
        Parameters:
            positions: Array of atomic positions [frames, atoms, 3]
            velocities: Array of velocities [frames, atoms, 3]
            masses: Array of atomic masses [atoms]
            
        Returns:
            frequencies: Normal mode frequencies in THz
            modes: Normal mode eigenvectors
        """
        self.profiler.start("calculate_normal_modes")
        
        # Check inputs
        if len(positions.shape) != 3 or len(velocities.shape) != 3:
            logger.error("Positions and velocities must be 3D arrays [frames, atoms, 3]")
            self.profiler.stop()
            return None, None
        
        n_frames, n_atoms, n_dims = positions.shape
        
        # Set default masses if not provided
        if masses is None:
            masses = np.ones(n_atoms)
        
        # Ensure masses is 1D array of length n_atoms
        masses = np.asarray(masses).reshape(-1)
        if len(masses) != n_atoms:
            logger.error(f"Masses array length ({len(masses)}) doesn't match n_atoms ({n_atoms})")
            self.profiler.stop()
            return None, None
        
        # Calculate covariance matrix of atomic displacements
        # First, calculate average positions
        avg_positions = np.mean(positions, axis=0)
        
        # Mass-weighted displacements
        displacements = positions - avg_positions
        
        # Reshape to [frames, atoms*3]
        displacements = displacements.reshape(n_frames, n_atoms * n_dims)
        
        # Create mass weight vector [atoms*3]
        mass_weights = np.repeat(np.sqrt(masses), 3)
        
        # Mass-weight displacements
        weighted_displacements = displacements * mass_weights
        
        # Calculate covariance matrix
        covariance = np.dot(weighted_displacements.T, weighted_displacements) / n_frames
        
        # Diagonalize to get normal modes
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate frequencies from eigenvalues
        # omega^2 = eigenvalue * kB * T / mass
        kbt = KB * self.temperature
        frequencies = np.sqrt(np.maximum(0, eigenvalues) * kbt) / (2 * np.pi)
        
        # Convert to THz
        frequencies *= 1e-12
        
        # Remove zero or negative frequencies
        valid_idx = frequencies > 0
        frequencies = frequencies[valid_idx]
        eigenvectors = eigenvectors[:, valid_idx]
        
        self.profiler.stop()
        return frequencies, eigenvectors
    
    def project_to_normal_modes(self, velocities, eigenvectors, masses=None):
        """
        Project atomic velocities onto normal modes
        
        Parameters:
            velocities: Array of velocities [frames, atoms, 3]
            eigenvectors: Normal mode eigenvectors
            masses: Array of atomic masses [atoms]
            
        Returns:
            mode_velocities: Velocity projection on each normal mode
        """
        self.profiler.start("project_to_normal_modes")
        
        # Check inputs
        if len(velocities.shape) != 3:
            logger.error("Velocities must be a 3D array [frames, atoms, 3]")
            self.profiler.stop()
            return None
        
        n_frames, n_atoms, n_dims = velocities.shape
        
        # Reshape velocities to [frames, atoms*3]
        v_reshape = velocities.reshape(n_frames, n_atoms * n_dims)
        
        # Set default masses if not provided
        if masses is None:
            masses = np.ones(n_atoms)
        
        # Ensure masses is 1D array of length n_atoms
        masses = np.asarray(masses).reshape(-1)
        if len(masses) != n_atoms:
            logger.error(f"Masses array length ({len(masses)}) doesn't match n_atoms ({n_atoms})")
            self.profiler.stop()
            return None
        
        # Create mass weight vector [atoms*3]
        mass_weights = np.repeat(np.sqrt(masses), 3)
        
        # Mass-weight velocities
        weighted_velocities = v_reshape * mass_weights
        
        # Project onto normal modes
        mode_velocities = np.dot(weighted_velocities, eigenvectors)
        
        self.profiler.stop()
        return mode_velocities
    
    def calculate_mode_energies(self, mode_velocities, frequencies, temperature=None):
        """
        Calculate normal mode energies
        
        Parameters:
            mode_velocities: Velocity projection on each normal mode
            frequencies: Normal mode frequencies in THz
            temperature: System temperature in K
            
        Returns:
            mode_energies: Energy in each normal mode
        """
        self.profiler.start("calculate_mode_energies")
        
        # Check inputs
        if mode_velocities is None or frequencies is None:
            logger.error("Mode velocities and frequencies must be provided")
            self.profiler.stop()
            return None
        
        # Use provided temperature or default
        if temperature is None:
            temperature = self.temperature
        
        # Convert frequencies to angular frequency (rad/s)
        angular_freq = 2 * np.pi * frequencies * 1e12
        
        # Calculate mode energies using equipartition theorem
        # E = 0.5 * m * v^2 + 0.5 * m * omega^2 * x^2
        # We have mass-weighted velocities, so v_scaled = sqrt(m) * v
        # So kinetic energy = 0.5 * (v_scaled)^2
        
        # Kinetic energy per mode
        n_frames = mode_velocities.shape[0]
        n_modes = mode_velocities.shape[1]
        
        # Allocate array for total energies
        mode_energies = np.zeros((n_frames, n_modes))
        
        # Calculate kinetic energy for each mode
        kinetic_energy = 0.5 * mode_velocities**2
        
        # Approximate potential energy using equipartition theorem
        # At equilibrium, <KE> = <PE>, so total energy is 2 * KE
        mode_energies = 2 * kinetic_energy
        
        # Store results
        self.mode_energies = mode_energies
        
        self.profiler.stop()
        return mode_energies
    
    def calculate_mode_occupation(self, mode_energies, frequencies, temperature=None):
        """
        Calculate phonon mode occupation numbers
        
        Parameters:
            mode_energies: Energy in each normal mode
            frequencies: Normal mode frequencies in THz
            temperature: System temperature in K
            
        Returns:
            occupation: Occupation number for each mode
        """
        self.profiler.start("calculate_mode_occupation")
        
        # Check inputs
        if mode_energies is None or frequencies is None:
            logger.error("Mode energies and frequencies must be provided")
            self.profiler.stop()
            return None
        
        # Use provided temperature or default
        if temperature is None:
            temperature = self.temperature
        
        # Convert frequencies to THz if given in other units
        freq_thz = np.asarray(frequencies)
        
        # Calculate theoretical occupation using Bose-Einstein statistics
        # n = 1/(exp(ℏω/kT) - 1)
        theoretical_occupation = 1.0 / (np.exp(freq_to_energy(freq_thz) / (KB * temperature)) - 1.0)
        
        # Calculate actual occupation from mode energies
        # E = ℏω(n + 1/2)
        # So n = E/ℏω - 1/2
        
        # Convert frequency to angular frequency (rad/s)
        angular_freq = 2 * np.pi * freq_thz * 1e12
        
        # Calculate occupation
        n_frames, n_modes = mode_energies.shape
        occupation = np.zeros((n_frames, n_modes))
        
        for i in range(n_modes):
            if angular_freq[i] > 0:
                occupation[:, i] = mode_energies[:, i] / (HBAR * angular_freq[i]) - 0.5
        
        # Store results
        self.mode_occupations = occupation
        
        self.profiler.stop()
        return occupation, theoretical_occupation
    
    def calculate_mode_lifetimes(self, mode_velocities, window_size=1000):
        """
        Calculate phonon mode lifetimes from velocity autocorrelation
        
        Parameters:
            mode_velocities: Velocity projection on each normal mode
            window_size: Size of time window for autocorrelation
            
        Returns:
            mode_lifetimes: Lifetime for each mode in ps
        """
        self.profiler.start("calculate_mode_lifetimes")
        
        # Check inputs
        if mode_velocities is None:
            logger.error("Mode velocities must be provided")
            self.profiler.stop()
            return None
        
        n_frames, n_modes = mode_velocities.shape
        
        # Limit window size
        window_size = min(window_size, n_frames // 2)
        
        # Calculate autocorrelation for each mode
        mode_vacf = np.zeros((n_modes, window_size))
        
        for i in range(n_modes):
            # Extract mode velocity
            vel = mode_velocities[:, i]
            
            # Calculate autocorrelation
            for j in range(window_size):
                # Correlation at lag j
                corr = np.mean(vel[:-j] * vel[j:])
                mode_vacf[i, j] = corr
            
            # Normalize
            if mode_vacf[i, 0] > 0:
                mode_vacf[i, :] /= mode_vacf[i, 0]
        
        # Fit exponential decay to extract lifetimes
        mode_lifetimes = np.zeros(n_modes)
        
        # Time array in frames
        time_frames = np.arange(window_size)
        
        for i in range(n_modes):
            try:
                # Define exponential decay function
                def exp_decay(t, tau):
                    return np.exp(-t / tau)
                
                # Fit using non-linear least squares
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    exp_decay, time_frames, mode_vacf[i, :],
                    p0=[window_size/10],  # Initial guess
                    bounds=(0, window_size)  # Constraints
                )
                
                # Extract lifetime in frames
                lifetime_frames = popt[0]
                
                # Convert to ps
                mode_lifetimes[i] = lifetime_frames * self.timestep
                
            except:
                # If fitting fails, set lifetime to 0
                mode_lifetimes[i] = 0
        
        self.profiler.stop()
        return mode_lifetimes
    
    def calculate_thermal_conductivity(self, heatflux, temperature=None, volume=None):
        """
        Calculate thermal conductivity using Green-Kubo formula
        
        Parameters:
            heatflux: Heat flux data [frames, 3]
            temperature: System temperature in K
            volume: System volume in Å³
            
        Returns:
            kappa: Thermal conductivity in W/(m·K)
        """
        self.profiler.start("calculate_thermal_conductivity")
        
        # Check inputs
        if heatflux is None:
            logger.error("Heat flux data must be provided")
            self.profiler.stop()
            return None
        
        # Use provided temperature or default
        if temperature is None:
            temperature = self.temperature
        
        # Default volume if not provided
        if volume is None:
            logger.warning("No volume provided, using default value of 1000 Å³")
            volume = 1000.0  # Å³
        
        # Get number of frames and dimensions
        if len(heatflux.shape) == 2:
            n_frames, n_dims = heatflux.shape
        else:
            n_frames = len(heatflux)
            n_dims = 1
            heatflux = heatflux.reshape(n_frames, 1)
        
        # Limit autocorrelation to half the frames
        max_lag = n_frames // 2
        
        # Calculate heat flux autocorrelation function (HFACF)
        hfacf = np.zeros((max_lag, n_dims))
        
        for dim in range(n_dims):
            for lag in range(max_lag):
                # Correlation at lag
                hfacf[lag, dim] = np.mean(heatflux[:n_frames-lag, dim] * heatflux[lag:, dim])
        
        # Time array in ps
        time = np.arange(max_lag) * self.timestep
        
        # Integrate HFACF to get thermal conductivity
        # κ = V/(3kBT²) ∫ <J(0)·J(t)> dt
        
        # Convert constants to appropriate units
        # volume: Å³ to m³
        # temperature: K
        # timestep: ps to s
        
        # Volume in m³
        volume_m3 = volume * 1e-30
        
        # Constants for Green-Kubo formula
        const = volume_m3 / (3 * KB * temperature**2)
        
        # Integrate HFACF (trapezoidal rule)
        kappa_components = np.zeros(n_dims)
        
        for dim in range(n_dims):
            # Integrate with time in seconds
            kappa_components[dim] = const * np.trapz(hfacf[:, dim], x=time*1e-12)
        
        # Average over dimensions
        kappa = np.mean(kappa_components)
        
        self.profiler.stop()
        return kappa, kappa_components, hfacf
    
    def analyze_thermal_equilibration(self, mode_energies, timestep=None):
        """
        Analyze thermal equilibration process from mode energies
        
        Parameters:
            mode_energies: Energy in each normal mode
            timestep: Time step in ps
            
        Returns:
            time_constants: Equilibration time constant for each mode in ps
            equilibration_data: Dictionary of equilibration data
        """
        self.profiler.start("analyze_thermal_equilibration")
        
        # Check inputs
        if mode_energies is None:
            logger.error("Mode energies must be provided")
            self.profiler.stop()
            return None, None
        
        # Use provided timestep or default
        if timestep is None:
            timestep = self.timestep
        
        n_frames, n_modes = mode_energies.shape
        
        # Time array in ps
        time = np.arange(n_frames) * timestep
        
        # Results dictionary
        equilibration_data = {
            'time': time,
            'mode_energies': mode_energies,
            'fits': [],
            'final_energy': np.zeros(n_modes),
            'initial_energy': np.zeros(n_modes),
            'amplitude': np.zeros(n_modes),
            'time_constants': np.zeros(n_modes)
        }
        
        # Fit exponential decay for each mode
        for i in range(n_modes):
            # Extract mode energy
            energy = mode_energies[:, i]
            
            try:
                # Define exponential decay function
                def exp_decay(t, A, tau, E_inf):
                    return A * np.exp(-t / tau) + E_inf
                
                # Initial guess for fitting
                E0 = energy[0]
                E_inf = np.mean(energy[-int(n_frames/10):])  # Average of last 10%
                A0 = E0 - E_inf
                tau0 = n_frames * timestep / 5  # Initial guess: 1/5 of total time
                
                # Fit using non-linear least squares
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    exp_decay, time, energy,
                    p0=[A0, tau0, E_inf],  # Initial guess
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf])  # Constraints
                )
                
                # Extract parameters
                A, tau, E_inf = popt
                
                # Store fit results
                equilibration_data['fits'].append(popt)
                equilibration_data['final_energy'][i] = E_inf
                equilibration_data['initial_energy'][i] = E_inf + A
                equilibration_data['amplitude'][i] = A
                equilibration_data['time_constants'][i] = tau
                
            except:
                # If fitting fails, set values to NaN
                equilibration_data['fits'].append([np.nan, np.nan, np.nan])
                equilibration_data['final_energy'][i] = np.mean(energy[-int(n_frames/10):])
                equilibration_data['initial_energy'][i] = energy[0]
                equilibration_data['amplitude'][i] = np.nan
                equilibration_data['time_constants'][i] = np.nan
        
        self.profiler.stop()
        return equilibration_data['time_constants'], equilibration_data
    
    def analyze_energy_transfer(self, mode_energies, freq_bins=10, timestep=None):
        """
        Analyze energy transfer between frequency bins
        
        Parameters:
            mode_energies: Energy in each normal mode
            freq_bins: Number of frequency bins
            timestep: Time step in ps
            
        Returns:
            energy_transfer: Energy transfer correlation data
        """
        self.profiler.start("analyze_energy_transfer")
        
        # Check inputs
        if mode_energies is None:
            logger.error("Mode energies must be provided")
            self.profiler.stop()
            return None
        
        # Use provided timestep or default
        if timestep is None:
            timestep = self.timestep
        
        n_frames, n_modes = mode_energies.shape
        
        # Create frequency bins
        freq_bin_edges = np.linspace(0, 1, freq_bins + 1)
        
        # Assign modes to frequency bins
        mode_bins = np.zeros(n_modes, dtype=int)
        for i in range(n_modes):
            # Normalize mode index to [0, 1]
            normalized_index = i / (n_modes - 1)
            # Find bin
            bin_idx = np.searchsorted(freq_bin_edges, normalized_index) - 1
            # Ensure valid bin index
            bin_idx = max(0, min(bin_idx, freq_bins - 1))
            mode_bins[i] = bin_idx
        
        # Calculate energy in each frequency bin
        bin_energies = np.zeros((n_frames, freq_bins))
        for i in range(freq_bins):
            # Find modes in this bin
            bin_modes = (mode_bins == i)
            if np.sum(bin_modes) > 0:
                # Sum energies of modes in this bin
                bin_energies[:, i] = np.sum(mode_energies[:, bin_modes], axis=1)
        
        # Normalize bin energies
        bin_energies_norm = bin_energies.copy()
        for i in range(freq_bins):
            if np.max(bin_energies[:, i]) > 0:
                bin_energies_norm[:, i] /= np.max(bin_energies[:, i])
        
        # Calculate correlation between frequency bins
        correlation_matrix = np.corrcoef(bin_energies.T)
        
        # Time array in ps
        time = np.arange(n_frames) * timestep
        
        # Results dictionary
        energy_transfer = {
            'time': time,
            'bin_energies': bin_energies,
            'bin_energies_norm': bin_energies_norm,
            'correlation_matrix': correlation_matrix,
            'freq_bin_edges': freq_bin_edges,
            'mode_bins': mode_bins
        }
        
        self.profiler.stop()
        return energy_transfer
    
    def calculate_anharmonicity(self, frequencies, mode_energies, temperature_range=None):
        """
        Calculate phonon anharmonicity from mode energies
        
        Parameters:
            frequencies: Normal mode frequencies in THz
            mode_energies: Energy in each normal mode
            temperature_range: List of temperatures for comparison
            
        Returns:
            anharmonicity: Anharmonicity parameters for each mode
        """
        self.profiler.start("calculate_anharmonicity")
        
        # Check inputs
        if frequencies is None or mode_energies is None:
            logger.error("Frequencies and mode energies must be provided")
            self.profiler.stop()
            return None
        
        # Use default temperature range if not provided
        if temperature_range is None:
            temperature_range = [100, 200, 300, 400, 500]
        
        n_temps = len(temperature_range)
        n_modes = len(frequencies)
        
        # Calculate theoretical harmonic energy at each temperature
        harmonic_energy = np.zeros((n_temps, n_modes))
        
        for i, temp in enumerate(temperature_range):
            # E = ℏω/2 + ℏω/(exp(ℏω/kT) - 1)
            hw = freq_to_energy(frequencies)
            harmonic_energy[i, :] = hw * (0.5 + 1.0 / (np.exp(hw / (KB * temp)) - 1.0))
        
        # Calculate average actual energy for each mode
        actual_energy = np.mean(mode_energies, axis=0)
        
        # Calculate ratio of actual to harmonic energy at the simulation temperature
        # Find closest temperature in range
        temp_idx = np.argmin(np.abs(np.array(temperature_range) - self.temperature))
        energy_ratio = actual_energy / harmonic_energy[temp_idx, :]
        
        # Calculate frequency shift with temperature (anharmonicity)
        # Fit linear trend to harmonic energy vs temperature
        freq_shifts = np.zeros(n_modes)
        
        for i in range(n_modes):
            # Use linear regression
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(temperature_range, harmonic_energy[:, i])
            
            # Frequency shift is proportional to slope of energy vs temperature
            if harmonic_energy[temp_idx, i] > 0:
                freq_shifts[i] = slope / harmonic_energy[temp_idx, i]
            else:
                freq_shifts[i] = 0
        
        # Results dictionary
        anharmonicity = {
            'frequencies': frequencies,
            'temperature_range': temperature_range,
            'harmonic_energy': harmonic_energy,
            'actual_energy': actual_energy,
            'energy_ratio': energy_ratio,
            'freq_shifts': freq_shifts
        }
        
        self.profiler.stop()
        return anharmonicity 