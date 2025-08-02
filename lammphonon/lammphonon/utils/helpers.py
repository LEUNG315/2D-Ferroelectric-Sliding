#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions for phonon analysis
"""

import os
import numpy as np
import logging
import datetime
import re
import glob

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def log_message(message, logfile=None, print_console=True):
    """Log a message to file and/or console"""
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp_str}] {message}"
    
    if print_console:
        print(log_msg)
    
    if logfile:
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

def create_time_windows(n_frames, n_windows=10, overlap=0.0):
    """
    Create time windows for analysis
    
    Parameters:
        n_frames: Total number of frames
        n_windows: Number of windows to create
        overlap: Overlap ratio between windows (0-1)
        
    Returns:
        windows: List of (start, end) tuples
    """
    if n_windows <= 0:
        return [(0, n_frames)]
    
    if n_frames <= 1:
        return [(0, 1)]
    
    # Calculate window size
    if overlap < 0 or overlap >= 1:
        overlap = 0
        
    effective_windows = n_windows / (1 - overlap/2)
    window_size = int(n_frames / effective_windows)
    
    if window_size < 1:
        window_size = 1
    
    # Calculate step size
    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1
    
    # Create windows
    windows = []
    start = 0
    
    while start < n_frames:
        end = start + window_size
        if end > n_frames:
            end = n_frames
            
        # Skip empty windows
        if end > start:
            windows.append((start, end))
            
        # Move to next window
        start += step_size
        
        # Break if we can't create more windows
        if start >= n_frames - 1 or len(windows) >= n_windows:
            break
    
    return windows

def normalize_data(data, method='minmax'):
    """
    Normalize data using various methods
    
    Parameters:
        data: Data array to normalize
        method: Normalization method ('minmax', 'standard', 'l1', 'l2', 'max')
        
    Returns:
        normalized_data: Normalized data array
    """
    data = np.asarray(data)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return np.zeros_like(data)
    
    elif method == 'standard':
        # Standardization (z-score normalization)
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std > 0:
            return (data - data_mean) / data_std
        else:
            return np.zeros_like(data)
    
    elif method == 'l1':
        # L1 normalization (sum of absolute values = 1)
        data_sum = np.sum(np.abs(data))
        if data_sum > 0:
            return data / data_sum
        else:
            return np.zeros_like(data)
    
    elif method == 'l2':
        # L2 normalization (Euclidean norm = 1)
        data_norm = np.sqrt(np.sum(data**2))
        if data_norm > 0:
            return data / data_norm
        else:
            return np.zeros_like(data)
    
    elif method == 'max':
        # Max normalization (max absolute value = 1)
        data_max = np.max(np.abs(data))
        if data_max > 0:
            return data / data_max
        else:
            return np.zeros_like(data)
    
    else:
        # No normalization
        return data

def smooth_data(data, window_size=5, method='moving_avg'):
    """
    Smooth data using various methods
    
    Parameters:
        data: Data array to smooth
        window_size: Window size for smoothing
        method: Smoothing method ('moving_avg', 'gaussian', 'savgol')
        
    Returns:
        smoothed_data: Smoothed data array
    """
    data = np.asarray(data)
    
    # Check input
    if window_size < 1:
        return data
    
    if window_size >= len(data):
        window_size = len(data) - 1
    
    # Ensure window size is odd for some methods
    if method in ['savgol', 'gaussian'] and window_size % 2 == 0:
        window_size += 1
    
    if method == 'moving_avg':
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    elif method == 'gaussian':
        # Gaussian smoothing
        import scipy.ndimage
        sigma = window_size / 6.0  # Heuristic: 6 sigma spans the window
        return scipy.ndimage.gaussian_filter1d(data, sigma)
    
    elif method == 'savgol':
        # Savitzky-Golay filter
        try:
            import scipy.signal
            # Order of polynomial (must be less than window_size)
            polyorder = min(3, window_size - 1)
            return scipy.signal.savgol_filter(data, window_size, polyorder)
        except ImportError:
            # Fall back to moving average if scipy is not available
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='same')
    
    else:
        # No smoothing
        return data

def find_files_by_pattern(directory, pattern="*.*"):
    """Find files matching a pattern in a directory"""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    return glob.glob(os.path.join(directory, pattern))

def scan_data_directory(directory, file_patterns=None):
    """
    Scan directory for data files and categorize them
    
    Parameters:
        directory: Directory to scan
        file_patterns: Dictionary of patterns for each file type
        
    Returns:
        categorized_files: Dictionary of file lists by category
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    # Default patterns if not provided
    if file_patterns is None:
        file_patterns = {
            'dump': ['*.dump', '*.lammpstrj'],
            'energy': ['*energy*.dat', '*energy*.txt'],
            'velocity': ['*vel*.dat', '*velocity*.txt'],
            'force': ['*force*.dat', '*force*.txt'],
            'heatflux': ['*heat*.dat', '*flux*.dat', '*J*.dat'],
            'polarization': ['*polar*.dat', '*dipole*.dat']
        }
        
    # Scan for files
    categorized_files = {category: [] for category in file_patterns}
    
    for category, patterns in file_patterns.items():
        for pattern in patterns:
            matches = glob.glob(os.path.join(directory, pattern))
            categorized_files[category].extend(matches)
    
    # Remove duplicates and sort
    for category in categorized_files:
        categorized_files[category] = sorted(list(set(categorized_files[category])))
    
    return categorized_files

def detect_layers(positions, axis=2, method='kmeans', n_layers=2):
    """
    Detect atomic layers in a structure
    
    Parameters:
        positions: Atomic positions array
        axis: Axis for layer detection (0=x, 1=y, 2=z)
        method: Detection method ('kmeans', 'histogram', 'z-coordinate')
        n_layers: Number of layers to detect
        
    Returns:
        layer_indices: Dictionary mapping layer index to atom indices
    """
    positions = np.asarray(positions)
    
    if method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            # Use only the specified axis for clustering
            coords = positions[:, axis].reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_layers, random_state=0).fit(coords)
            labels = kmeans.labels_
            
            # Create dictionary of layer indices
            layer_indices = {}
            for i in range(n_layers):
                layer_indices[i] = np.where(labels == i)[0]
                
            return layer_indices
        except ImportError:
            # Fall back to z-coordinate method if sklearn is not available
            method = 'z-coordinate'
    
    if method == 'z-coordinate':
        # Sort atoms by z-coordinate and divide into n_layers groups
        sorted_indices = np.argsort(positions[:, axis])
        atoms_per_layer = len(positions) // n_layers
        
        layer_indices = {}
        for i in range(n_layers):
            start = i * atoms_per_layer
            end = (i + 1) * atoms_per_layer if i < n_layers - 1 else len(positions)
            layer_indices[i] = sorted_indices[start:end]
            
        return layer_indices
    
    if method == 'histogram':
        # Use histogram to find density peaks
        hist, bin_edges = np.histogram(positions[:, axis], bins=min(50, len(positions)//10))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks as local maxima
        from scipy.signal import find_peaks
        peak_indices, _ = find_peaks(hist)
        
        if len(peak_indices) < n_layers:
            # Not enough peaks found, fall back to z-coordinate method
            return detect_layers(positions, axis, 'z-coordinate', n_layers)
        
        # If too many peaks, keep the n_layers highest
        if len(peak_indices) > n_layers:
            peak_heights = hist[peak_indices]
            top_indices = np.argsort(peak_heights)[-n_layers:]
            peak_indices = peak_indices[top_indices]
        
        # Get peak positions
        peak_positions = bin_centers[peak_indices]
        
        # Assign atoms to nearest peak
        labels = np.zeros(len(positions), dtype=int)
        for i, pos in enumerate(positions[:, axis]):
            distances = np.abs(pos - peak_positions)
            labels[i] = np.argmin(distances)
        
        # Create dictionary of layer indices
        layer_indices = {}
        for i in range(n_layers):
            layer_indices[i] = np.where(labels == i)[0]
            
        return layer_indices 