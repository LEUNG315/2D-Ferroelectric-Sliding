#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File readers for LAMMPS output files
"""

import os
import numpy as np
import re
import glob
import logging
from ..utils.timing import Timer

# Setup logging
logger = logging.getLogger(__name__)

class DumpReader:
    """LAMMPS dump file reader"""
    
    def __init__(self, filename=None):
        """
        Initialize LAMMPS dump file reader
        
        Parameters:
            filename: Path to dump file (optional)
        """
        self.filename = filename
        self.n_atoms = 0
        self.n_frames = 0
        self.timesteps = []
        self.box_bounds = []
        self.column_names = []
        self.column_mapping = {}
        self.frame_positions = []  # File positions for each frame
        
        # Read file metadata if filename is provided
        if filename:
            self._read_metadata()
    
    def _read_metadata(self):
        """Read file metadata without loading all data"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        with open(self.filename, 'r') as f:
            # Read first frame to get column names and atom count
            line = f.readline()
            while line and not line.startswith("ITEM: TIMESTEP"):
                line = f.readline()
            
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            
            # Record position of first frame
            self.frame_positions.append(f.tell() - len(line))
            
            # Read timestep
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            self.timesteps.append(int(line.strip()))
            
            # Read number of atoms
            while line and not line.startswith("ITEM: NUMBER OF ATOMS"):
                line = f.readline()
            
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            self.n_atoms = int(line.strip())
            
            # Read box bounds
            while line and not line.startswith("ITEM: BOX BOUNDS"):
                line = f.readline()
            
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            
            box_bounds = []
            for _ in range(3):  # x, y, z
                line = f.readline()
                if not line:
                    raise ValueError(f"Invalid dump file format: {self.filename}")
                bounds = list(map(float, line.strip().split()))
                box_bounds.append(bounds[:2])  # Only use the first two values
            
            self.box_bounds.append(box_bounds)
            
            # Read column names
            while line and not line.startswith("ITEM: ATOMS"):
                line = f.readline()
            
            if not line:
                raise ValueError(f"Invalid dump file format: {self.filename}")
            
            self.column_names = line.strip().split()[2:]  # Skip "ITEM: ATOMS"
            
            # Create mapping from column name to index
            self.column_mapping = {name: i for i, name in enumerate(self.column_names)}
            
            # Check if file has required columns
            required_columns = ['id', 'type', 'x', 'y', 'z']
            missing_columns = [col for col in required_columns if col not in self.column_mapping]
            if missing_columns:
                logger.warning(f"Dump file missing columns: {', '.join(missing_columns)}")
            
            # Skip atoms in first frame
            for _ in range(self.n_atoms):
                line = f.readline()
            
            # Count remaining frames
            while True:
                line = f.readline()
                if not line:
                    break
                
                if line.startswith("ITEM: TIMESTEP"):
                    # Record position of frame
                    self.frame_positions.append(f.tell() - len(line))
                    
                    # Read timestep
                    line = f.readline()
                    if not line:
                        break
                    self.timesteps.append(int(line.strip()))
                    
                    # Read box bounds
                    for _ in range(9):  # Skip to BOX BOUNDS
                        line = f.readline()
                        if not line:
                            break
                    
                    if not line or not line.startswith("ITEM: BOX BOUNDS"):
                        break
                    
                    box_bounds = []
                    for _ in range(3):  # x, y, z
                        line = f.readline()
                        if not line:
                            break
                        bounds = list(map(float, line.strip().split()))
                        box_bounds.append(bounds[:2])
                    
                    if len(box_bounds) == 3:
                        self.box_bounds.append(box_bounds)
                    
                    # Skip atoms
                    for _ in range(self.n_atoms + 1):  # +1 for ITEM: ATOMS line
                        line = f.readline()
                        if not line:
                            break
            
            # Update frame count
            self.n_frames = len(self.frame_positions)
    
    def read_frame(self, frame_idx=-1):
        """
        Read a specific frame from the dump file
        
        Parameters:
            frame_idx: Frame index (-1 for last frame)
            
        Returns:
            timestep: Timestep of the frame
            box: Box bounds
            atoms: Atom data as structured array
        """
        if not self.filename:
            raise ValueError("No dump file specified")
        
        if not self.frame_positions:
            self._read_metadata()
        
        # Adjust negative indices
        if frame_idx < 0:
            frame_idx = self.n_frames + frame_idx
        
        # Check bounds
        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"Frame index {frame_idx} out of bounds (0-{self.n_frames-1})")
        
        with open(self.filename, 'r') as f:
            # Seek to frame position
            f.seek(self.frame_positions[frame_idx])
            
            # Read timestep
            line = f.readline()  # ITEM: TIMESTEP
            line = f.readline()
            timestep = int(line.strip())
            
            # Read number of atoms (just to confirm)
            line = f.readline()  # ITEM: NUMBER OF ATOMS
            line = f.readline()
            n_atoms = int(line.strip())
            
            if n_atoms != self.n_atoms:
                logger.warning(f"Frame {frame_idx} has {n_atoms} atoms, expected {self.n_atoms}")
            
            # Read box bounds
            line = f.readline()  # ITEM: BOX BOUNDS
            box_bounds = []
            for _ in range(3):  # x, y, z
                line = f.readline()
                bounds = list(map(float, line.strip().split()))
                box_bounds.append(bounds[:2])
            
            # Read column names (just to confirm)
            line = f.readline()  # ITEM: ATOMS
            columns = line.strip().split()[2:]
            
            if columns != self.column_names:
                logger.warning(f"Frame {frame_idx} has different columns than expected")
            
            # Read atom data
            atom_data = np.zeros(n_atoms, dtype=object)
            for i in range(n_atoms):
                line = f.readline()
                atom_data[i] = line.strip().split()
            
            # Convert to structured array
            atoms = np.array(atom_data.tolist())
            
            return timestep, box_bounds, atoms
    
    def read_all_frames(self):
        """
        Read all frames from the dump file
        
        Returns:
            timesteps: List of timesteps
            boxes: List of box bounds
            all_atoms: List of atom data arrays
        """
        if not self.filename:
            raise ValueError("No dump file specified")
        
        if not self.frame_positions:
            self._read_metadata()
        
        all_atoms = []
        
        for i in range(self.n_frames):
            _, _, atoms = self.read_frame(i)
            all_atoms.append(atoms)
        
        return self.timesteps, self.box_bounds, all_atoms
    
    def get_positions(self, frame_idx=-1):
        """
        Get atom positions from a specific frame
        
        Parameters:
            frame_idx: Frame index (-1 for last frame)
            
        Returns:
            positions: Atom positions as Nx3 array
        """
        _, _, atoms = self.read_frame(frame_idx)
        
        # Get column indices
        x_idx = self.column_mapping.get('x')
        y_idx = self.column_mapping.get('y')
        z_idx = self.column_mapping.get('z')
        
        if x_idx is None or y_idx is None or z_idx is None:
            raise ValueError("Dump file missing position columns")
        
        # Extract positions
        positions = np.zeros((len(atoms), 3))
        positions[:, 0] = atoms[:, x_idx].astype(float)
        positions[:, 1] = atoms[:, y_idx].astype(float)
        positions[:, 2] = atoms[:, z_idx].astype(float)
        
        return positions
    
    def get_velocities(self, frame_idx=-1):
        """
        Get atom velocities from a specific frame
        
        Parameters:
            frame_idx: Frame index (-1 for last frame)
            
        Returns:
            velocities: Atom velocities as Nx3 array
        """
        _, _, atoms = self.read_frame(frame_idx)
        
        # Get column indices
        vx_idx = self.column_mapping.get('vx')
        vy_idx = self.column_mapping.get('vy')
        vz_idx = self.column_mapping.get('vz')
        
        if vx_idx is None or vy_idx is None or vz_idx is None:
            raise ValueError("Dump file missing velocity columns")
        
        # Extract velocities
        velocities = np.zeros((len(atoms), 3))
        velocities[:, 0] = atoms[:, vx_idx].astype(float)
        velocities[:, 1] = atoms[:, vy_idx].astype(float)
        velocities[:, 2] = atoms[:, vz_idx].astype(float)
        
        return velocities
    
    def get_forces(self, frame_idx=-1):
        """
        Get atom forces from a specific frame
        
        Parameters:
            frame_idx: Frame index (-1 for last frame)
            
        Returns:
            forces: Atom forces as Nx3 array
        """
        _, _, atoms = self.read_frame(frame_idx)
        
        # Get column indices
        fx_idx = self.column_mapping.get('fx')
        fy_idx = self.column_mapping.get('fy')
        fz_idx = self.column_mapping.get('fz')
        
        if fx_idx is None or fy_idx is None or fz_idx is None:
            raise ValueError("Dump file missing force columns")
        
        # Extract forces
        forces = np.zeros((len(atoms), 3))
        forces[:, 0] = atoms[:, fx_idx].astype(float)
        forces[:, 1] = atoms[:, fy_idx].astype(float)
        forces[:, 2] = atoms[:, fz_idx].astype(float)
        
        return forces
    
    def get_atom_types(self, frame_idx=-1):
        """
        Get atom types from a specific frame
        
        Parameters:
            frame_idx: Frame index (-1 for last frame)
            
        Returns:
            types: Atom types as array
        """
        _, _, atoms = self.read_frame(frame_idx)
        
        # Get column index
        type_idx = self.column_mapping.get('type')
        
        if type_idx is None:
            raise ValueError("Dump file missing type column")
        
        # Extract types
        types = atoms[:, type_idx].astype(int)
        
        return types

class DataReader:
    """Generic data file reader for energy, heatflux, etc."""
    
    def __init__(self, filename=None):
        """
        Initialize data file reader
        
        Parameters:
            filename: Path to data file (optional)
        """
        self.filename = filename
        self.data = None
        self.header = None
        self.column_names = None
        
        # Read file if filename is provided
        if filename:
            self.read_file()
    
    def read_file(self, filename=None, skip_rows=0, delimiter=None):
        """
        Read data file
        
        Parameters:
            filename: Path to data file (override instance filename)
            skip_rows: Number of rows to skip
            delimiter: Column delimiter
            
        Returns:
            data: Data array
        """
        if filename:
            self.filename = filename
        
        if not self.filename:
            raise ValueError("No data file specified")
        
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        # Try to auto-detect header and column names
        with open(self.filename, 'r') as f:
            lines = []
            for _ in range(20):  # Read first 20 lines
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            
            # Find header lines (starting with #)
            header_lines = [line for line in lines if line.strip().startswith('#')]
            self.header = ''.join(header_lines)
            
            # Try to find column names in header
            for line in header_lines:
                # Remove # and split
                parts = line.strip('#').strip().split()
                # Check if these look like column names (no numbers)
                if parts and not any(part.replace('.', '').isdigit() for part in parts):
                    self.column_names = parts
                    break
            
            # If no column names found in header, try first non-header line
            if not self.column_names:
                non_header_lines = [line for line in lines if not line.strip().startswith('#')]
                if non_header_lines:
                    first_line = non_header_lines[0].strip()
                    # Check if first line consists of strings (not all numbers)
                    parts = first_line.split()
                    if parts and not all(part.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() for part in parts):
                        self.column_names = parts
                        skip_rows += lines.index(non_header_lines[0]) + 1
            
            # Count data rows
            f.seek(0)
            for _ in range(skip_rows):
                f.readline()
            
            data_rows = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
        
        # Read data using numpy
        try:
            self.data = np.loadtxt(self.filename, skiprows=skip_rows, delimiter=delimiter)
            return self.data
        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            raise
    
    def get_column(self, col_idx=0):
        """
        Get a specific column of data
        
        Parameters:
            col_idx: Column index or name
            
        Returns:
            column: Column data
        """
        if self.data is None:
            self.read_file()
        
        # Convert column name to index if needed
        if isinstance(col_idx, str) and self.column_names:
            try:
                col_idx = self.column_names.index(col_idx)
            except ValueError:
                raise ValueError(f"Column name '{col_idx}' not found in {self.column_names}")
        
        # Check bounds
        if col_idx < 0 or col_idx >= self.data.shape[1]:
            raise IndexError(f"Column index {col_idx} out of bounds (0-{self.data.shape[1]-1})")
        
        return self.data[:, col_idx]
    
    def get_columns(self, col_indices=None):
        """
        Get multiple columns of data
        
        Parameters:
            col_indices: List of column indices or names
            
        Returns:
            columns: Array of column data
        """
        if self.data is None:
            self.read_file()
        
        # Default to all columns
        if col_indices is None:
            return self.data
        
        # Convert column names to indices if needed
        indices = []
        for idx in col_indices:
            if isinstance(idx, str) and self.column_names:
                try:
                    indices.append(self.column_names.index(idx))
                except ValueError:
                    raise ValueError(f"Column name '{idx}' not found in {self.column_names}")
            else:
                indices.append(idx)
        
        # Check bounds
        for idx in indices:
            if idx < 0 or idx >= self.data.shape[1]:
                raise IndexError(f"Column index {idx} out of bounds (0-{self.data.shape[1]-1})")
        
        return self.data[:, indices]

def read_lammps_log(filename, data_keys=None):
    """
    Read LAMMPS log file and extract thermo data
    
    Parameters:
        filename: Path to log file
        data_keys: Optional list of thermo keywords to extract
        
    Returns:
        thermo_data: Dictionary of thermo data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Default thermo keywords
    if data_keys is None:
        data_keys = ['Step', 'Temp', 'E_pair', 'E_mol', 'TotEng', 'Press']
    
    # Initialize data dictionary
    thermo_data = {key: [] for key in data_keys}
    
    with open(filename, 'r') as f:
        in_thermo = False
        thermo_header = None
        
        for line in f:
            line = line.strip()
            
            # Look for thermo header
            if not in_thermo and any(key in line.split() for key in data_keys):
                header_parts = line.split()
                # Check if this line contains multiple thermo keywords
                if sum(1 for key in data_keys if key in header_parts) > 1:
                    thermo_header = header_parts
                    in_thermo = True
                    continue
            
            # Process thermo data
            if in_thermo:
                parts = line.split()
                
                # Check if we've reached the end of a thermo section
                if not parts or len(parts) != len(thermo_header):
                    in_thermo = False
                    continue
                
                # Try to parse as numbers
                try:
                    values = [float(part) for part in parts]
                    
                    # Add values to data dictionary
                    for key, val in zip(thermo_header, values):
                        if key in data_keys:
                            thermo_data[key].append(val)
                            
                except ValueError:
                    # Not a data line, end of thermo section
                    in_thermo = False
    
    # Convert to numpy arrays
    for key in thermo_data:
        thermo_data[key] = np.array(thermo_data[key])
    
    return thermo_data

def read_energy_file(filename):
    """
    Read energy data file
    
    Parameters:
        filename: Path to energy file
        
    Returns:
        energy_data: Dictionary of energy data
    """
    reader = DataReader(filename)
    
    # Try to identify columns
    energy_data = {}
    
    if reader.column_names:
        # Map common column names
        column_map = {
            'time': ['time', 'timestep', 'step', 'Time', 'TimeStep', 'Step'],
            'total': ['etotal', 'total', 'TotEng', 'tot_energy', 'total_energy'],
            'kinetic': ['ke', 'kinetic', 'KinEng', 'kin_energy', 'kinetic_energy'],
            'potential': ['pe', 'potential', 'PotEng', 'pot_energy', 'potential_energy']
        }
        
        # Find column indices
        for key, names in column_map.items():
            for name in names:
                if name in reader.column_names:
                    col_idx = reader.column_names.index(name)
                    energy_data[key] = reader.get_column(col_idx)
                    break
    
    # If columns not identified by name, use default positions
    if not energy_data:
        # Assume columns are: time, total, kinetic, potential
        data = reader.data
        if data.shape[1] >= 4:
            energy_data['time'] = data[:, 0]
            energy_data['total'] = data[:, 1]
            energy_data['kinetic'] = data[:, 2]
            energy_data['potential'] = data[:, 3]
        elif data.shape[1] >= 3:
            energy_data['time'] = data[:, 0]
            energy_data['total'] = data[:, 1]
            energy_data['potential'] = data[:, 2]
        elif data.shape[1] >= 2:
            energy_data['time'] = data[:, 0]
            energy_data['total'] = data[:, 1]
    
    return energy_data

def read_heatflux_file(filename):
    """
    Read heat flux data file
    
    Parameters:
        filename: Path to heat flux file
        
    Returns:
        heatflux_data: Dictionary of heat flux data
    """
    reader = DataReader(filename)
    
    # Try to identify columns
    heatflux_data = {}
    
    if reader.column_names:
        # Map common column names
        column_map = {
            'time': ['time', 'timestep', 'step', 'Time', 'TimeStep', 'Step'],
            'jx': ['jx', 'Jx', 'heatflux_x', 'flux_x'],
            'jy': ['jy', 'Jy', 'heatflux_y', 'flux_y'],
            'jz': ['jz', 'Jz', 'heatflux_z', 'flux_z']
        }
        
        # Find column indices
        for key, names in column_map.items():
            for name in names:
                if name in reader.column_names:
                    col_idx = reader.column_names.index(name)
                    heatflux_data[key] = reader.get_column(col_idx)
                    break
    
    # If columns not identified by name, use default positions
    if not heatflux_data or len(heatflux_data) < 3:
        # Assume columns are: time, jx, jy, jz
        data = reader.data
        if data.shape[1] >= 4:
            heatflux_data['time'] = data[:, 0]
            heatflux_data['jx'] = data[:, 1]
            heatflux_data['jy'] = data[:, 2]
            heatflux_data['jz'] = data[:, 3]
        elif data.shape[1] >= 3:
            heatflux_data['time'] = data[:, 0]
            heatflux_data['jx'] = data[:, 1]
            heatflux_data['jy'] = data[:, 2]
    
    return heatflux_data

def read_polarization_file(filename):
    """
    Read polarization data file
    
    Parameters:
        filename: Path to polarization file
        
    Returns:
        polarization_data: Dictionary of polarization data
    """
    reader = DataReader(filename)
    
    # Try to identify columns
    polarization_data = {}
    
    if reader.column_names:
        # Map common column names
        column_map = {
            'time': ['time', 'timestep', 'step', 'Time', 'TimeStep', 'Step'],
            'px': ['px', 'Px', 'dipole_x', 'polarization_x'],
            'py': ['py', 'Py', 'dipole_y', 'polarization_y'],
            'pz': ['pz', 'Pz', 'dipole_z', 'polarization_z']
        }
        
        # Find column indices
        for key, names in column_map.items():
            for name in names:
                if name in reader.column_names:
                    col_idx = reader.column_names.index(name)
                    polarization_data[key] = reader.get_column(col_idx)
                    break
    
    # If columns not identified by name, use default positions
    if not polarization_data or len(polarization_data) < 3:
        # Assume columns are: time, px, py, pz
        data = reader.data
        if data.shape[1] >= 4:
            polarization_data['time'] = data[:, 0]
            polarization_data['px'] = data[:, 1]
            polarization_data['py'] = data[:, 2]
            polarization_data['pz'] = data[:, 3]
        elif data.shape[1] >= 3:
            polarization_data['time'] = data[:, 0]
            polarization_data['px'] = data[:, 1]
            polarization_data['py'] = data[:, 2]
    
    return polarization_data 