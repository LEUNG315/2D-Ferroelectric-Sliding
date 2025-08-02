#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File writers for saving analysis results and data
"""

import os
import numpy as np
import json
import pickle
import logging

# Setup logging
logger = logging.getLogger(__name__)

def save_text_file(data, filename, header=None, fmt='%.8f', delimiter=' '):
    """
    Save data to text file
    
    Parameters:
        data: Data array to save
        filename: Output filename
        header: Optional header text
        fmt: Format string for numpy savetxt
        delimiter: Column delimiter
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save data
        np.savetxt(filename, data, header=header, fmt=fmt, delimiter=delimiter)
        logger.info(f"Saved text data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving text file: {e}")
        return False

def save_csv(data, filename, header=None):
    """
    Save data to CSV file
    
    Parameters:
        data: Data array to save
        filename: Output filename
        header: Optional header row (list of column names)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        # Convert header to string if provided
        header_str = None
        if header is not None:
            if isinstance(header, list):
                header_str = ','.join(header)
            else:
                header_str = str(header)
        
        # Save data
        np.savetxt(filename, data, header=header_str, fmt='%g', delimiter=',')
        logger.info(f"Saved CSV data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}")
        return False

def save_numpy(data, filename):
    """
    Save data to NumPy .npy file
    
    Parameters:
        data: Data array to save
        filename: Output filename
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .npy extension
        if not filename.lower().endswith('.npy'):
            filename += '.npy'
        
        # Save data
        np.save(filename, data)
        logger.info(f"Saved NumPy data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving NumPy file: {e}")
        return False

def save_json(data, filename, indent=2):
    """
    Save data to JSON file
    
    Parameters:
        data: Data dictionary to save
        filename: Output filename
        indent: JSON indentation level
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
        
        # Convert NumPy arrays to lists
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        # Convert data
        json_data = convert_for_json(data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=indent)
        
        logger.info(f"Saved JSON data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")
        return False

def save_pickle(data, filename):
    """
    Save data to pickle file
    
    Parameters:
        data: Python object to save
        filename: Output filename
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .pkl extension
        if not filename.lower().endswith(('.pkl', '.pickle')):
            filename += '.pkl'
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved pickle data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}")
        return False

def save_xyz(positions, filename, atom_types=None, comment=''):
    """
    Save atomic positions to XYZ file
    
    Parameters:
        positions: Atomic positions array (N, 3)
        filename: Output filename
        atom_types: Atom type strings (if not provided, defaults to 'C')
        comment: Optional comment line
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .xyz extension
        if not filename.lower().endswith('.xyz'):
            filename += '.xyz'
        
        # Default atom types if not provided
        if atom_types is None:
            atom_types = ['C'] * len(positions)
        elif isinstance(atom_types, np.ndarray) and atom_types.dtype.kind in 'iuf':
            # Convert numeric types to carbon/hydrogen
            atom_types = ['C' if t == 1 else 'H' for t in atom_types]
        
        # Check dimensions
        if len(atom_types) != len(positions):
            raise ValueError(f"Number of atom types ({len(atom_types)}) doesn't match positions ({len(positions)})")
        
        # Write XYZ file
        with open(filename, 'w') as f:
            f.write(f"{len(positions)}\n")
            f.write(f"{comment}\n")
            
            for i, pos in enumerate(positions):
                f.write(f"{atom_types[i]} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
        
        logger.info(f"Saved XYZ data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving XYZ file: {e}")
        return False

def save_report(report_data, filename):
    """
    Save analysis report to text file
    
    Parameters:
        report_data: Dictionary of report sections
        filename: Output filename
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .txt extension
        if not filename.lower().endswith('.txt'):
            filename += '.txt'
        
        # Prepare report content
        content = []
        
        # Add title
        if 'title' in report_data:
            content.append(report_data['title'])
            content.append('=' * len(report_data['title']))
            content.append('')
        
        # Add datetime
        if 'datetime' in report_data:
            content.append(f"Generated: {report_data['datetime']}")
            content.append('')
        
        # Add sections
        for section_name, section_content in report_data.items():
            if section_name in ['title', 'datetime']:
                continue
                
            # Add section header
            content.append(section_name)
            content.append('-' * len(section_name))
            
            # Add section content
            if isinstance(section_content, str):
                content.append(section_content)
            elif isinstance(section_content, list):
                content.extend(section_content)
            elif isinstance(section_content, dict):
                for key, value in section_content.items():
                    content.append(f"{key}: {value}")
            else:
                content.append(str(section_content))
            
            content.append('')
        
        # Write to file
        with open(filename, 'w') as f:
            f.write('\n'.join(content))
        
        logger.info(f"Saved report to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return False 