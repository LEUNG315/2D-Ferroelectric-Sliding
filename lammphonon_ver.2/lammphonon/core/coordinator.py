#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core coordinator module for phonon analysis
"""

import os
import sys
import logging
import numpy as np
import datetime
from ..utils.timing import Timer, TimeProfiler
from ..utils.helpers import ensure_dir, log_message
from ..io.readers import DumpReader, DataReader, read_energy_file, read_heatflux_file, read_polarization_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PhononCoordinator:
    """Coordinator for phonon analysis operations"""
    
    def __init__(self, config=None):
        """
        Initialize the coordinator
        
        Parameters:
            config: Dictionary of configuration parameters
        """
        # Set default configuration
        self.config = {
            'temperature': 300.0,  # K
            'freq_max': 30.0,      # THz
            'freq_points': 1000,   # Number of frequency points
            'sigma': 0.1,          # Smoothing parameter for DOS
            'timestep': 0.001,     # ps
            'dump_interval': 10,   # Dump interval in MD steps
            'max_frames': 0,       # Maximum frames to analyze (0 = all)
            'skip_frames': 0,      # Number of frames to skip
            'high_performance': False,  # Enable high-performance mode
            'use_pca': False,      # Use PCA for dimension reduction
            'n_layers': 2,         # Number of layers in the system
            'debug': False         # Enable debug logging
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Setup output directories
        self.output_dir = self.config.get('output_dir', os.path.expanduser("~/lammphonon_results"))
        self.results_dir = os.path.join(self.output_dir, "results")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.data_dir = os.path.join(self.output_dir, "data")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # Create directories if they don't exist
        for directory in [self.output_dir, self.results_dir, self.figures_dir, self.data_dir, self.logs_dir]:
            ensure_dir(directory)
        
        # Setup logging
        log_file = os.path.join(self.logs_dir, f"lammphonon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Set debug mode if requested
        if self.config['debug']:
            logger.setLevel(logging.DEBUG)
        
        # Input files
        self.input_files = {
            'trajectory': None,
            'energy': None,
            'force': None,
            'polarization': None,
            'heatflux': None,
            'masses': None,
            'layer_definition': None
        }
        
        # Data storage
        self.trajectory_data = None
        self.energy_data = None
        self.force_data = None
        self.polarization_data = None
        self.heatflux_data = None
        
        # Analysis results
        self.results = {}
        
        # Performance profiler
        self.profiler = TimeProfiler("PhononAnalysis")
        
        logger.info("PhononCoordinator initialized")
    
    def set_config(self, key, value):
        """Set a configuration parameter"""
        self.config[key] = value
        logger.debug(f"Config set: {key} = {value}")
    
    def get_config(self, key, default=None):
        """Get a configuration parameter"""
        return self.config.get(key, default)
    
    def set_config_from_dict(self, config_dict):
        """Update configuration from a dictionary"""
        self.config.update(config_dict)
        logger.debug(f"Config updated with {len(config_dict)} parameters")
    
    def set_input_file(self, file_type, file_path):
        """
        Set an input file
        
        Parameters:
            file_type: Type of file ('trajectory', 'energy', etc.)
            file_path: Path to the file
        """
        if file_type not in self.input_files:
            logger.warning(f"Unknown file type: {file_type}")
            return False
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        self.input_files[file_type] = file_path
        logger.info(f"Set {file_type} file: {file_path}")
        return True
    
    def read_trajectory(self, filename=None, max_frames=None, skip_frames=None):
        """
        Read LAMMPS trajectory file
        
        Parameters:
            filename: Trajectory file path (if None, use previously set file)
            max_frames: Maximum frames to read (if None, use config value)
            skip_frames: Number of frames to skip (if None, use config value)
            
        Returns:
            success: True if successful
        """
        # Use provided file or previously set file
        if filename:
            self.set_input_file('trajectory', filename)
        
        filename = self.input_files['trajectory']
        if not filename:
            logger.error("No trajectory file specified")
            return False
        
        # Use provided parameters or config values
        if max_frames is None:
            max_frames = self.config['max_frames']
        
        if skip_frames is None:
            skip_frames = self.config['skip_frames']
        
        try:
            # Start profiling
            self.profiler.start("read_trajectory")
            
            # Create reader
            reader = DumpReader(filename)
            
            # Read metadata
            logger.info(f"Reading trajectory file: {filename}")
            logger.info(f"  File contains {reader.n_frames} frames, {reader.n_atoms} atoms")
            
            # Determine how many frames to read
            start_frame = skip_frames
            end_frame = reader.n_frames
            
            if max_frames > 0:
                end_frame = min(start_frame + max_frames, reader.n_frames)
            
            # Read frames
            timesteps = []
            box_bounds = []
            positions = []
            velocities = []
            
            logger.info(f"Reading frames {start_frame} to {end_frame-1} (total: {end_frame-start_frame})")
            
            for i in range(start_frame, end_frame):
                timestep, box, atoms = reader.read_frame(i)
                
                # Extract positions and velocities
                pos = np.zeros((reader.n_atoms, 3))
                vel = np.zeros((reader.n_atoms, 3))
                
                # Get column indices
                x_idx = reader.column_mapping.get('x')
                y_idx = reader.column_mapping.get('y')
                z_idx = reader.column_mapping.get('z')
                vx_idx = reader.column_mapping.get('vx')
                vy_idx = reader.column_mapping.get('vy')
                vz_idx = reader.column_mapping.get('vz')
                
                # Check if we have position data
                if x_idx is not None and y_idx is not None and z_idx is not None:
                    pos[:, 0] = atoms[:, x_idx].astype(float)
                    pos[:, 1] = atoms[:, y_idx].astype(float)
                    pos[:, 2] = atoms[:, z_idx].astype(float)
                
                # Check if we have velocity data
                if vx_idx is not None and vy_idx is not None and vz_idx is not None:
                    vel[:, 0] = atoms[:, vx_idx].astype(float)
                    vel[:, 1] = atoms[:, vy_idx].astype(float)
                    vel[:, 2] = atoms[:, vz_idx].astype(float)
                
                timesteps.append(timestep)
                box_bounds.append(box)
                positions.append(pos)
                velocities.append(vel)
            
            # Store data
            self.trajectory_data = {
                'timesteps': np.array(timesteps),
                'box_bounds': box_bounds,
                'positions': positions,
                'velocities': velocities,
                'n_atoms': reader.n_atoms,
                'n_frames': len(timesteps)
            }
            
            # Get atom types from first frame
            type_idx = reader.column_mapping.get('type')
            if type_idx is not None:
                _, _, atoms = reader.read_frame(start_frame)
                self.trajectory_data['atom_types'] = atoms[:, type_idx].astype(int)
            
            # Stop profiling
            self.profiler.stop()
            
            logger.info(f"Trajectory read successfully: {self.trajectory_data['n_frames']} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error reading trajectory: {str(e)}")
            self.profiler.stop()
            return False
    
    def read_energy_data(self, filename=None):
        """
        Read energy data file
        
        Parameters:
            filename: Energy file path (if None, use previously set file)
            
        Returns:
            success: True if successful
        """
        # Use provided file or previously set file
        if filename:
            self.set_input_file('energy', filename)
        
        filename = self.input_files['energy']
        if not filename:
            logger.error("No energy file specified")
            return False
        
        try:
            # Start profiling
            self.profiler.start("read_energy_data")
            
            # Read energy data
            logger.info(f"Reading energy file: {filename}")
            self.energy_data = read_energy_file(filename)
            
            # Stop profiling
            self.profiler.stop()
            
            logger.info(f"Energy data read successfully: {len(self.energy_data.get('time', []))} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error reading energy data: {str(e)}")
            self.profiler.stop()
            return False
    
    def read_heatflux_data(self, filename=None):
        """
        Read heat flux data file
        
        Parameters:
            filename: Heat flux file path (if None, use previously set file)
            
        Returns:
            success: True if successful
        """
        # Use provided file or previously set file
        if filename:
            self.set_input_file('heatflux', filename)
        
        filename = self.input_files['heatflux']
        if not filename:
            logger.error("No heat flux file specified")
            return False
        
        try:
            # Start profiling
            self.profiler.start("read_heatflux_data")
            
            # Read heat flux data
            logger.info(f"Reading heat flux file: {filename}")
            self.heatflux_data = read_heatflux_file(filename)
            
            # Stop profiling
            self.profiler.stop()
            
            logger.info(f"Heat flux data read successfully: {len(self.heatflux_data.get('time', []))} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error reading heat flux data: {str(e)}")
            self.profiler.stop()
            return False
    
    def read_polarization_data(self, filename=None):
        """
        Read polarization data file
        
        Parameters:
            filename: Polarization file path (if None, use previously set file)
            
        Returns:
            success: True if successful
        """
        # Use provided file or previously set file
        if filename:
            self.set_input_file('polarization', filename)
        
        filename = self.input_files['polarization']
        if not filename:
            logger.error("No polarization file specified")
            return False
        
        try:
            # Start profiling
            self.profiler.start("read_polarization_data")
            
            # Read polarization data
            logger.info(f"Reading polarization file: {filename}")
            self.polarization_data = read_polarization_file(filename)
            
            # Stop profiling
            self.profiler.stop()
            
            logger.info(f"Polarization data read successfully: {len(self.polarization_data.get('time', []))} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error reading polarization data: {str(e)}")
            self.profiler.stop()
            return False
    
    def get_performance_summary(self):
        """Get performance summary from profiler"""
        return self.profiler.summary()
    
    def save_results(self, result_name, data, description=None):
        """
        Save analysis results to the results dictionary
        
        Parameters:
            result_name: Name of the result
            data: Result data
            description: Optional description
        """
        self.results[result_name] = {
            'data': data,
            'description': description,
            'timestamp': datetime.datetime.now().isoformat()
        }
        logger.debug(f"Saved result: {result_name}")
    
    def get_result(self, result_name):
        """Get a result by name"""
        if result_name in self.results:
            return self.results[result_name]['data']
        else:
            logger.warning(f"Result not found: {result_name}")
            return None
    
    def generate_report(self):
        """Generate a summary report of all analyses"""
        report = {
            'title': 'Phonon Analysis Report',
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Configuration': self.config,
            'Input Files': {k: v for k, v in self.input_files.items() if v is not None},
            'Results Summary': []
        }
        
        # Add results summary
        for name, result in self.results.items():
            description = result.get('description', 'No description')
            timestamp = result.get('timestamp', 'Unknown')
            report['Results Summary'].append(f"{name}: {description} (generated: {timestamp})")
        
        # Add performance info
        report['Performance'] = self.get_performance_summary()
        
        return report 