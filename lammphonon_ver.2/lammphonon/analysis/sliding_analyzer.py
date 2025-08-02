#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sliding dynamics analysis module for layered materials
"""

import numpy as np
import logging
from scipy import signal
from ..utils.timing import Timer, TimeProfiler
from ..utils.helpers import detect_layers, smooth_data

# Setup logging
logger = logging.getLogger(__name__)

class SlidingAnalyzer:
    """Sliding dynamics analysis for layered materials"""
    
    def __init__(self, coordinator=None):
        """
        Initialize the sliding analyzer
        
        Parameters:
            coordinator: PhononCoordinator instance
        """
        self.coordinator = coordinator
        self.profiler = TimeProfiler("SlidingAnalyzer")
        
        # Default parameters
        self.n_layers = 2
        self.layer_direction = 'z'
        self.sliding_direction = 'x'
        self.detection_method = 'kmeans'
        self.smoothing_window = 10
        
        # Update parameters from coordinator if provided
        if coordinator:
            self.n_layers = coordinator.get_config('n_layers', self.n_layers)
            self.detection_method = coordinator.get_config('layer_detection_method', self.detection_method)
            self.smoothing_window = coordinator.get_config('smoothing_window', self.smoothing_window)
        
        # Analysis results
        self.layer_indices = None
        self.sliding_distance = None
        self.friction_force = None
        self.interlayer_distance = None
        
        logger.debug("SlidingAnalyzer initialized")
    
    def detect_material_layers(self, positions, method=None, n_layers=None, layer_direction=None):
        """
        Detect layers in a layered material
        
        Parameters:
            positions: Atomic positions array [atoms, 3]
            method: Detection method (kmeans, z-coordinate, histogram)
            n_layers: Number of layers to detect
            layer_direction: Direction perpendicular to layers (x, y, z)
            
        Returns:
            layer_indices: Dictionary mapping layer index to atom indices
        """
        self.profiler.start("detect_layers")
        
        # Use provided parameters or default values
        if method is None:
            method = self.detection_method
        
        if n_layers is None:
            n_layers = self.n_layers
        
        if layer_direction is None:
            layer_direction = self.layer_direction
        
        # Convert direction string to axis index
        if isinstance(layer_direction, str):
            if layer_direction.lower() == 'x':
                axis = 0
            elif layer_direction.lower() == 'y':
                axis = 1
            elif layer_direction.lower() == 'z':
                axis = 2
            else:
                logger.warning(f"Unknown direction '{layer_direction}', using z-axis")
                axis = 2
        else:
            # Assume it's already an axis index
            axis = layer_direction
        
        # Detect layers using helper function
        layer_indices = detect_layers(
            positions, axis=axis, method=method, n_layers=n_layers)
        
        # Store results
        self.layer_indices = layer_indices
        
        self.profiler.stop()
        return layer_indices
    
    def calculate_center_of_mass(self, positions, atom_indices=None, masses=None):
        """
        Calculate center of mass for a group of atoms
        
        Parameters:
            positions: Atomic positions array [atoms, 3]
            atom_indices: Indices of atoms to include (None for all)
            masses: Atomic masses (None for uniform mass)
            
        Returns:
            com: Center of mass coordinates [3]
        """
        # Use all atoms if indices not provided
        if atom_indices is None:
            atom_indices = np.arange(len(positions))
        
        # Extract positions for selected atoms
        selected_positions = positions[atom_indices]
        
        # Use uniform masses if not provided
        if masses is None:
            com = np.mean(selected_positions, axis=0)
        else:
            # Extract masses for selected atoms
            if len(masses) == len(positions):
                selected_masses = masses[atom_indices]
            else:
                logger.warning(f"Masses array length ({len(masses)}) doesn't match positions ({len(positions)})")
                selected_masses = np.ones(len(atom_indices))
            
            # Calculate mass-weighted center of mass
            total_mass = np.sum(selected_masses)
            if total_mass > 0:
                com = np.sum(selected_positions * selected_masses[:, np.newaxis], axis=0) / total_mass
            else:
                com = np.mean(selected_positions, axis=0)
        
        return com
    
    def calculate_sliding_distance(self, positions_trajectory, sliding_direction=None):
        """
        Calculate sliding distance between layers over time
        
        Parameters:
            positions_trajectory: Atomic positions for each frame [frames, atoms, 3]
            sliding_direction: Direction of sliding (x, y, or vector)
            
        Returns:
            sliding_distance: Sliding distance over time [frames]
        """
        self.profiler.start("calculate_sliding_distance")
        
        # Check if we have detected layers
        if self.layer_indices is None:
            # Detect layers using first frame
            self.detect_material_layers(positions_trajectory[0])
        
        # Convert direction string to axis index or vector
        if sliding_direction is None:
            sliding_direction = self.sliding_direction
        
        if isinstance(sliding_direction, str):
            if sliding_direction.lower() == 'x':
                direction_vec = np.array([1, 0, 0])
                axis = 0
            elif sliding_direction.lower() == 'y':
                direction_vec = np.array([0, 1, 0])
                axis = 1
            elif sliding_direction.lower() == 'z':
                direction_vec = np.array([0, 0, 1])
                axis = 2
            else:
                logger.warning(f"Unknown direction '{sliding_direction}', using x-axis")
                direction_vec = np.array([1, 0, 0])
                axis = 0
        elif isinstance(sliding_direction, (list, tuple, np.ndarray)):
            # Use provided vector
            direction_vec = np.array(sliding_direction, dtype=float)
            direction_vec /= np.linalg.norm(direction_vec)
            axis = None  # We'll use projection instead of component
        else:
            # Default to x-axis
            logger.warning(f"Unknown sliding_direction type, using x-axis")
            direction_vec = np.array([1, 0, 0])
            axis = 0
        
        # Calculate center of mass for each layer in each frame
        n_frames = len(positions_trajectory)
        n_layers = len(self.layer_indices)
        
        # Store CoM for each layer and frame
        layer_coms = np.zeros((n_frames, n_layers, 3))
        
        for f in range(n_frames):
            for l, indices in self.layer_indices.items():
                layer_coms[f, l] = self.calculate_center_of_mass(
                    positions_trajectory[f], indices)
        
        # Calculate relative displacement between layers
        sliding_distance = np.zeros(n_frames)
        
        if n_layers >= 2:
            # Calculate displacement between first two layers
            if axis is not None:
                # Use component along specified axis
                displacement = layer_coms[:, 1, axis] - layer_coms[:, 0, axis]
            else:
                # Project displacement onto direction vector
                displacement = np.sum(
                    (layer_coms[:, 1] - layer_coms[:, 0]) * direction_vec, 
                    axis=1
                )
            
            # First frame is reference (zero displacement)
            sliding_distance = displacement - displacement[0]
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            sliding_distance = smooth_data(
                sliding_distance, 
                window_size=self.smoothing_window, 
                method='moving_avg'
            )
        
        # Store results
        self.sliding_distance = sliding_distance
        
        self.profiler.stop()
        return sliding_distance
    
    def calculate_interlayer_distance(self, positions_trajectory, layer_direction=None):
        """
        Calculate interlayer distance over time
        
        Parameters:
            positions_trajectory: Atomic positions for each frame [frames, atoms, 3]
            layer_direction: Direction perpendicular to layers (x, y, z)
            
        Returns:
            interlayer_distance: Distance between layers over time [frames, n_layers-1]
        """
        self.profiler.start("calculate_interlayer_distance")
        
        # Check if we have detected layers
        if self.layer_indices is None:
            # Detect layers using first frame
            self.detect_material_layers(positions_trajectory[0])
        
        # Convert direction string to axis index or vector
        if layer_direction is None:
            layer_direction = self.layer_direction
        
        if isinstance(layer_direction, str):
            if layer_direction.lower() == 'x':
                direction_vec = np.array([1, 0, 0])
                axis = 0
            elif layer_direction.lower() == 'y':
                direction_vec = np.array([0, 1, 0])
                axis = 1
            elif layer_direction.lower() == 'z':
                direction_vec = np.array([0, 0, 1])
                axis = 2
            else:
                logger.warning(f"Unknown direction '{layer_direction}', using z-axis")
                direction_vec = np.array([0, 0, 1])
                axis = 2
        elif isinstance(layer_direction, (list, tuple, np.ndarray)):
            # Use provided vector
            direction_vec = np.array(layer_direction, dtype=float)
            direction_vec /= np.linalg.norm(direction_vec)
            axis = None  # We'll use projection instead of component
        else:
            # Default to z-axis
            logger.warning(f"Unknown layer_direction type, using z-axis")
            direction_vec = np.array([0, 0, 1])
            axis = 2
        
        # Calculate center of mass for each layer in each frame
        n_frames = len(positions_trajectory)
        n_layers = len(self.layer_indices)
        
        # Store CoM for each layer and frame
        layer_coms = np.zeros((n_frames, n_layers, 3))
        
        for f in range(n_frames):
            for l, indices in self.layer_indices.items():
                layer_coms[f, l] = self.calculate_center_of_mass(
                    positions_trajectory[f], indices)
        
        # Calculate distance between adjacent layers
        interlayer_distance = np.zeros((n_frames, n_layers - 1))
        
        for l in range(n_layers - 1):
            if axis is not None:
                # Use component along specified axis
                distance = np.abs(layer_coms[:, l+1, axis] - layer_coms[:, l, axis])
            else:
                # Project displacement onto direction vector
                distance = np.abs(np.sum(
                    (layer_coms[:, l+1] - layer_coms[:, l]) * direction_vec, 
                    axis=1
                ))
            
            interlayer_distance[:, l] = distance
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            for l in range(n_layers - 1):
                interlayer_distance[:, l] = smooth_data(
                    interlayer_distance[:, l], 
                    window_size=self.smoothing_window, 
                    method='moving_avg'
                )
        
        # Store results
        self.interlayer_distance = interlayer_distance
        
        self.profiler.stop()
        return interlayer_distance
    
    def calculate_friction_force(self, forces_trajectory, sliding_direction=None):
        """
        Calculate friction force between layers over time
        
        Parameters:
            forces_trajectory: Atomic forces for each frame [frames, atoms, 3]
            sliding_direction: Direction of sliding (x, y, or vector)
            
        Returns:
            friction_force: Friction force over time [frames]
        """
        self.profiler.start("calculate_friction_force")
        
        # Check if we have detected layers
        if self.layer_indices is None:
            logger.error("Layer indices not detected. Call detect_material_layers first.")
            self.profiler.stop()
            return None
        
        # Convert direction string to axis index or vector
        if sliding_direction is None:
            sliding_direction = self.sliding_direction
        
        if isinstance(sliding_direction, str):
            if sliding_direction.lower() == 'x':
                direction_vec = np.array([1, 0, 0])
                axis = 0
            elif sliding_direction.lower() == 'y':
                direction_vec = np.array([0, 1, 0])
                axis = 1
            elif sliding_direction.lower() == 'z':
                direction_vec = np.array([0, 0, 1])
                axis = 2
            else:
                logger.warning(f"Unknown direction '{sliding_direction}', using x-axis")
                direction_vec = np.array([1, 0, 0])
                axis = 0
        elif isinstance(sliding_direction, (list, tuple, np.ndarray)):
            # Use provided vector
            direction_vec = np.array(sliding_direction, dtype=float)
            direction_vec /= np.linalg.norm(direction_vec)
            axis = None  # We'll use projection instead of component
        else:
            # Default to x-axis
            logger.warning(f"Unknown sliding_direction type, using x-axis")
            direction_vec = np.array([1, 0, 0])
            axis = 0
        
        # Calculate total force on each layer in each frame
        n_frames = len(forces_trajectory)
        n_layers = len(self.layer_indices)
        
        # Store total force for each layer and frame
        layer_forces = np.zeros((n_frames, n_layers, 3))
        
        for f in range(n_frames):
            for l, indices in self.layer_indices.items():
                # Sum forces on all atoms in the layer
                layer_forces[f, l] = np.sum(forces_trajectory[f][indices], axis=0)
        
        # Calculate friction force (equal and opposite forces between layers)
        friction_force = np.zeros(n_frames)
        
        if n_layers >= 2:
            # Use force on first layer along sliding direction
            if axis is not None:
                # Use component along specified axis
                friction_force = layer_forces[:, 0, axis]
            else:
                # Project force onto direction vector
                friction_force = np.sum(layer_forces[:, 0] * direction_vec, axis=1)
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            friction_force = smooth_data(
                friction_force, 
                window_size=self.smoothing_window, 
                method='moving_avg'
            )
        
        # Store results
        self.friction_force = friction_force
        
        self.profiler.stop()
        return friction_force
    
    def calculate_friction_coefficient(self, friction_force=None, normal_force=None):
        """
        Calculate friction coefficient from friction force and normal force
        
        Parameters:
            friction_force: Friction force over time [frames]
            normal_force: Normal force between layers [frames or scalar]
            
        Returns:
            friction_coefficient: Friction coefficient over time [frames]
        """
        self.profiler.start("calculate_friction_coefficient")
        
        # Use stored friction force if not provided
        if friction_force is None:
            friction_force = self.friction_force
            if friction_force is None:
                logger.error("No friction force data available")
                self.profiler.stop()
                return None
        
        # Check normal force
        if normal_force is None:
            logger.warning("No normal force provided, using unit normal force")
            normal_force = 1.0
        
        # Calculate friction coefficient (μ = F_friction / F_normal)
        if np.isscalar(normal_force):
            # Constant normal force
            friction_coefficient = friction_force / normal_force
        else:
            # Time-dependent normal force
            if len(normal_force) != len(friction_force):
                logger.warning(f"Length mismatch: friction_force ({len(friction_force)}) vs normal_force ({len(normal_force)})")
                # Resize normal_force to match friction_force
                if len(normal_force) > len(friction_force):
                    normal_force = normal_force[:len(friction_force)]
                else:
                    # Pad normal_force with last value
                    padding = np.full(len(friction_force) - len(normal_force), normal_force[-1])
                    normal_force = np.concatenate([normal_force, padding])
            
            # Avoid division by zero
            normal_force_safe = np.where(normal_force != 0, normal_force, 1.0)
            friction_coefficient = friction_force / normal_force_safe
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            friction_coefficient = smooth_data(
                friction_coefficient, 
                window_size=self.smoothing_window, 
                method='moving_avg'
            )
        
        self.profiler.stop()
        return friction_coefficient
    
    def analyze_stick_slip(self, sliding_distance=None, friction_force=None, min_peak_height=None):
        """
        Analyze stick-slip behavior in sliding friction
        
        Parameters:
            sliding_distance: Sliding distance over time [frames]
            friction_force: Friction force over time [frames]
            min_peak_height: Minimum peak height for detection
            
        Returns:
            stick_slip_data: Dictionary with stick-slip analysis results
        """
        self.profiler.start("analyze_stick_slip")
        
        # Use stored data if not provided
        if sliding_distance is None:
            sliding_distance = self.sliding_distance
            if sliding_distance is None:
                logger.error("No sliding distance data available")
                self.profiler.stop()
                return None
        
        if friction_force is None:
            friction_force = self.friction_force
            if friction_force is None:
                logger.error("No friction force data available")
                self.profiler.stop()
                return None
        
        # Default min_peak_height to 10% of max force
        if min_peak_height is None:
            min_peak_height = 0.1 * np.max(np.abs(friction_force))
        
        # Find force peaks (stick-slip events)
        peaks, peak_properties = signal.find_peaks(
            np.abs(friction_force), 
            height=min_peak_height, 
            distance=self.smoothing_window
        )
        
        # Results dictionary
        stick_slip_data = {
            'peak_indices': peaks,
            'peak_forces': friction_force[peaks],
            'peak_distances': sliding_distance[peaks],
            'n_peaks': len(peaks)
        }
        
        # Calculate slip distances and periods
        if len(peaks) > 1:
            slip_distances = np.diff(sliding_distance[peaks])
            slip_periods = np.diff(peaks)
            
            stick_slip_data['slip_distances'] = slip_distances
            stick_slip_data['slip_periods'] = slip_periods
            stick_slip_data['avg_slip_distance'] = np.mean(slip_distances)
            stick_slip_data['avg_slip_period'] = np.mean(slip_periods)
        
        # Calculate average peak force
        if len(peaks) > 0:
            stick_slip_data['avg_peak_force'] = np.mean(np.abs(friction_force[peaks]))
        
        self.profiler.stop()
        return stick_slip_data
    
    def calculate_energy_dissipation(self, friction_force=None, sliding_velocity=None, timestep=None):
        """
        Calculate energy dissipation rate from friction force and sliding velocity
        
        Parameters:
            friction_force: Friction force over time [frames]
            sliding_velocity: Sliding velocity over time [frames]
            timestep: Time step in ps
            
        Returns:
            dissipation_rate: Energy dissipation rate over time [frames]
        """
        self.profiler.start("calculate_energy_dissipation")
        
        # Use stored friction force if not provided
        if friction_force is None:
            friction_force = self.friction_force
            if friction_force is None:
                logger.error("No friction force data available")
                self.profiler.stop()
                return None
        
        # Calculate sliding velocity if not provided
        if sliding_velocity is None:
            if self.sliding_distance is None:
                logger.error("No sliding distance data available")
                self.profiler.stop()
                return None
            
            # Get timestep from coordinator or use default
            if timestep is None:
                if self.coordinator:
                    timestep = self.coordinator.get_config('timestep', 0.001)  # ps
                else:
                    timestep = 0.001  # ps
            
            # Calculate velocity using central difference
            sliding_velocity = np.zeros_like(self.sliding_distance)
            sliding_velocity[1:-1] = (self.sliding_distance[2:] - self.sliding_distance[:-2]) / (2 * timestep)
            
            # Forward/backward difference at endpoints
            sliding_velocity[0] = (self.sliding_distance[1] - self.sliding_distance[0]) / timestep
            sliding_velocity[-1] = (self.sliding_distance[-1] - self.sliding_distance[-2]) / timestep
        
        # Calculate dissipation rate (P = F * v)
        dissipation_rate = friction_force * sliding_velocity
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            dissipation_rate = smooth_data(
                dissipation_rate, 
                window_size=self.smoothing_window, 
                method='moving_avg'
            )
        
        self.profiler.stop()
        return dissipation_rate
    
    def calculate_work_energy(self, friction_force=None, sliding_distance=None):
        """
        Calculate cumulative work done against friction
        
        Parameters:
            friction_force: Friction force over time [frames]
            sliding_distance: Sliding distance over time [frames]
            
        Returns:
            work_energy: Cumulative work energy over time [frames]
        """
        self.profiler.start("calculate_work_energy")
        
        # Use stored data if not provided
        if friction_force is None:
            friction_force = self.friction_force
            if friction_force is None:
                logger.error("No friction force data available")
                self.profiler.stop()
                return None
        
        if sliding_distance is None:
            sliding_distance = self.sliding_distance
            if sliding_distance is None:
                logger.error("No sliding distance data available")
                self.profiler.stop()
                return None
        
        # Calculate displacement between frames
        displacement_increments = np.zeros_like(sliding_distance)
        displacement_increments[1:] = sliding_distance[1:] - sliding_distance[:-1]
        
        # Calculate work done in each increment (W = F * dx)
        work_increments = friction_force * displacement_increments
        
        # Cumulative work (energy)
        work_energy = np.cumsum(work_increments)
        
        self.profiler.stop()
        return work_energy
    
    def calculate_stacking_order(self, positions_trajectory, layer_indices=None, reference_frame=0):
        """
        Calculate stacking order parameter for layered materials
        
        Parameters:
            positions_trajectory: Atomic positions for each frame [frames, atoms, 3]
            layer_indices: Dictionary mapping layer index to atom indices
            reference_frame: Reference frame for relative displacement
            
        Returns:
            stacking_parameter: Stacking order parameter over time [frames]
        """
        self.profiler.start("calculate_stacking_order")
        
        # Use stored layer indices if not provided
        if layer_indices is None:
            layer_indices = self.layer_indices
            if layer_indices is None:
                # Detect layers using first frame
                self.detect_material_layers(positions_trajectory[0])
                layer_indices = self.layer_indices
        
        # Check if we have at least 2 layers
        if len(layer_indices) < 2:
            logger.error("Need at least 2 layers to calculate stacking order")
            self.profiler.stop()
            return None
        
        # Get atomic positions in each layer
        n_frames = len(positions_trajectory)
        
        # Calculate in-plane displacement between layers (relative to reference frame)
        displacements = np.zeros((n_frames, 2))  # x, y displacement
        
        # Get reference positions (first two layers)
        ref_layer0_pos = positions_trajectory[reference_frame][layer_indices[0]]
        ref_layer1_pos = positions_trajectory[reference_frame][layer_indices[1]]
        
        # Calculate reference centers of mass
        ref_com0 = np.mean(ref_layer0_pos, axis=0)
        ref_com1 = np.mean(ref_layer1_pos, axis=0)
        
        # Reference in-plane displacement
        ref_displacement = ref_com1[:2] - ref_com0[:2]
        
        for f in range(n_frames):
            # Get layer positions in this frame
            layer0_pos = positions_trajectory[f][layer_indices[0]]
            layer1_pos = positions_trajectory[f][layer_indices[1]]
            
            # Calculate centers of mass
            com0 = np.mean(layer0_pos, axis=0)
            com1 = np.mean(layer1_pos, axis=0)
            
            # In-plane displacement (x, y)
            current_displacement = com1[:2] - com0[:2]
            
            # Displacement relative to reference frame
            displacements[f] = current_displacement - ref_displacement
        
        # Calculate stacking order parameter
        # For graphene/graphite, we can use a periodic function of the displacement
        # where AA stacking is at (0,0) and AB stacking is at (a/sqrt(3), 0)
        # where a is the lattice constant (~2.46 Å for graphene)
        
        # Estimate lattice constant from typical C-C bond length
        bond_length = 1.42  # Å
        lattice_constant = bond_length * 2.46 / 1.42  # ~2.46 Å
        
        # Calculate distance from high-symmetry points (AA, AB, BA)
        aa_distance = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
        
        ab_point = np.array([lattice_constant / np.sqrt(3), 0])
        ab_distance = np.sqrt((displacements[:, 0] - ab_point[0])**2 + 
                            (displacements[:, 1] - ab_point[1])**2)
        
        ba_point = np.array([lattice_constant / (2 * np.sqrt(3)), 
                            lattice_constant / 2])
        ba_distance = np.sqrt((displacements[:, 0] - ba_point[0])**2 + 
                            (displacements[:, 1] - ba_point[1])**2)
        
        # Normalize distances by lattice constant
        aa_distance /= lattice_constant
        ab_distance /= lattice_constant
        ba_distance /= lattice_constant
        
        # Create stacking order parameter:
        # 1.0 = AA stacking, 0.0 = AB/BA stacking, intermediate values for other stackings
        stacking_parameter = 1.0 - np.minimum(
            np.minimum(aa_distance, ab_distance), ba_distance) * 2
        
        # Clip to [0, 1] range
        stacking_parameter = np.clip(stacking_parameter, 0, 1)
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            stacking_parameter = smooth_data(
                stacking_parameter, 
                window_size=self.smoothing_window, 
                method='moving_avg'
            )
        
        self.profiler.stop()
        return stacking_parameter, displacements
    
    def get_sliding_analysis_summary(self):
        """
        Get a summary of sliding analysis results
        
        Returns:
            summary: Dictionary with analysis summary
        """
        summary = {
            'n_layers': self.n_layers,
            'layer_direction': self.layer_direction,
            'sliding_direction': self.sliding_direction,
        }
        
        # Add sliding distance statistics if available
        if self.sliding_distance is not None:
            summary['max_sliding_distance'] = np.max(np.abs(self.sliding_distance))
            summary['final_sliding_distance'] = self.sliding_distance[-1]
        
        # Add friction force statistics if available
        if self.friction_force is not None:
            summary['max_friction_force'] = np.max(np.abs(self.friction_force))
            summary['avg_friction_force'] = np.mean(np.abs(self.friction_force))
            summary['std_friction_force'] = np.std(self.friction_force)
        
        # Add interlayer distance statistics if available
        if self.interlayer_distance is not None:
            summary['avg_interlayer_distance'] = np.mean(self.interlayer_distance, axis=0)
            summary['min_interlayer_distance'] = np.min(self.interlayer_distance, axis=0)
            summary['max_interlayer_distance'] = np.max(self.interlayer_distance, axis=0)
        
        return summary 