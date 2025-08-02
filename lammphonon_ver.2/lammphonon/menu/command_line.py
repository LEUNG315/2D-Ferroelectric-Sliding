#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for LAMMPhonon

This script provides a command-line interface for LAMMPhonon, 
allowing users to perform various phonon analyses through command line arguments.

Author: Shuming Liang (梁树铭)
Email: lsm315@mail.ustc.edu.cn
Phone: 18256949203
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

from ..core.coordinator import PhononCoordinator
from ..core.phonon_analyzer import PhononAnalyzer
from ..analysis.sliding_analyzer import SlidingAnalyzer
from ..analysis.thermal_analyzer import ThermalAnalyzer
from ..analysis.anharmonic_analyzer import AnharmonicAnalyzer
from ..analysis.equilibration_analyzer import EquilibrationAnalyzer
from ..analysis.temporal_analyzer import TemporalAnalyzer
from ..utils.timing import Timer
from ..utils.constants import atomic_mass_unit, boltzmann_constant
from ..utils.logger import setup_logger

# Create the logger
logger = logging.getLogger(__name__)

def setup_argument_parser():
    """
    Set up the argument parser for command-line options
    
    Returns:
        argparse.ArgumentParser: The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='LAMMPhonon - Phonon Analysis Toolkit for LAMMPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lammphonon_analyze -i dump.phonon -o results/ -dos
  lammphonon_analyze -i dump.phonon --energy energy.dat --heatflux heatflux.dat -thermal
  lammphonon_analyze -i dump.phonon -sliding --layer layer_def.txt
""")
    
    # Input/Output options
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input trajectory file (LAMMPS dump format)')
    parser.add_argument('-o', '--output', type=str, default='./lammphonon_results',
                       help='Output directory for results (default: ./lammphonon_results)')
    parser.add_argument('--energy', type=str,
                       help='Energy data file')
    parser.add_argument('--force', type=str,
                       help='Force data file')
    parser.add_argument('--heatflux', type=str,
                       help='Heat flux data file')
    parser.add_argument('--polarization', type=str,
                       help='Polarization data file')
    parser.add_argument('--mass', type=str,
                       help='Mass file')
    parser.add_argument('--layer', type=str,
                       help='Layer definition file')
    
    # Analysis type flags
    parser.add_argument('-dos', '--density-of-states', action='store_true',
                       help='Calculate phonon density of states')
    parser.add_argument('-vacf', '--velocity-autocorrelation', action='store_true',
                       help='Calculate velocity autocorrelation function')
    parser.add_argument('-pdos', '--projected-dos', action='store_true',
                       help='Calculate projected density of states')
    parser.add_argument('-nm', '--normal-modes', action='store_true',
                       help='Calculate normal modes')
    parser.add_argument('-po', '--phonon-occupation', action='store_true',
                       help='Calculate phonon occupation numbers')
    parser.add_argument('-tau', '--phonon-lifetime', action='store_true',
                       help='Calculate phonon lifetimes')
    parser.add_argument('-thermal', '--thermal-analysis', action='store_true',
                       help='Perform thermal analysis (requires heatflux data)')
    parser.add_argument('-sliding', '--sliding-analysis', action='store_true',
                       help='Perform sliding analysis')
    parser.add_argument('-stacking', '--stacking-analysis', action='store_true',
                       help='Perform stacking configuration analysis')
    parser.add_argument('-equil', '--equilibration-analysis', action='store_true',
                       help='Perform equilibration analysis')
    parser.add_argument('-all', '--all-analyses', action='store_true',
                       help='Perform all available analyses')
    
    # Configuration parameters
    parser.add_argument('-temp', '--temperature', type=float, default=300.0,
                       help='System temperature in Kelvin (default: 300.0)')
    parser.add_argument('--freq-max', type=float, default=50.0,
                       help='Maximum frequency in THz (default: 50.0)')
    parser.add_argument('--freq-points', type=int, default=1000,
                       help='Number of frequency points (default: 1000)')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Gaussian broadening parameter (default: 0.1)')
    parser.add_argument('--timestep', type=float, default=0.001,
                       help='MD timestep in ps (default: 0.001)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to analyze')
    parser.add_argument('--frame-stride', type=int, default=1,
                       help='Stride for frame selection (default: 1)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of material layers (default: 2)')
    
    # Debugging options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--apply-patches', action='store_true',
                       help='Apply runtime patches to fix potential issues')
    
    return parser


def check_input_files(args):
    """
    Check if required input files exist
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    # Check trajectory file
    if not os.path.exists(args.input):
        logger.error(f"Trajectory file not found: {args.input}")
        return False
    
    # Check additional input files if specified
    for file_arg in ['energy', 'force', 'heatflux', 'polarization', 'mass', 'layer']:
        filename = getattr(args, file_arg)
        if filename and not os.path.exists(filename):
            logger.error(f"{file_arg.capitalize()} file not found: {filename}")
            return False
    
    return True


def setup_output_directory(args):
    """
    Create output directory if it doesn't exist
    
    Args:
        args: Command-line arguments
        
    Returns:
        str: Path to the output directory
    """
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            return None
    
    return output_dir


def load_data(coordinator, args):
    """
    Load data into the PhononCoordinator
    
    Args:
        coordinator: PhononCoordinator instance
        args: Command-line arguments
        
    Returns:
        bool: True if data loading was successful, False otherwise
    """
    timer = Timer("Data Loading")
    timer.start()
    
    # Load trajectory data
    logger.info(f"Loading trajectory file: {args.input}")
    try:
        coordinator.read_trajectory(args.input, max_frames=args.max_frames, frame_stride=args.frame_stride)
    except Exception as e:
        logger.error(f"Failed to read trajectory file: {str(e)}")
        return False
    
    # Check if we have trajectory data
    if not coordinator.has_trajectory_data():
        logger.error("No trajectory data loaded")
        return False
    
    # Load additional data
    if args.energy:
        logger.info(f"Loading energy file: {args.energy}")
        try:
            coordinator.read_energy_data(args.energy)
        except Exception as e:
            logger.error(f"Failed to read energy file: {str(e)}")
    
    if args.force:
        logger.info(f"Loading force file: {args.force}")
        try:
            coordinator.read_force_data(args.force)
        except Exception as e:
            logger.error(f"Failed to read force file: {str(e)}")
    
    if args.heatflux:
        logger.info(f"Loading heat flux file: {args.heatflux}")
        try:
            coordinator.read_heatflux_data(args.heatflux)
        except Exception as e:
            logger.error(f"Failed to read heat flux file: {str(e)}")
    
    if args.polarization:
        logger.info(f"Loading polarization file: {args.polarization}")
        try:
            coordinator.read_polarization_data(args.polarization)
        except Exception as e:
            logger.error(f"Failed to read polarization file: {str(e)}")
    
    if args.mass:
        logger.info(f"Loading mass file: {args.mass}")
        try:
            coordinator.read_mass_data(args.mass)
        except Exception as e:
            logger.error(f"Failed to read mass file: {str(e)}")
    
    if args.layer:
        logger.info(f"Loading layer definition file: {args.layer}")
        try:
            coordinator.read_layer_data(args.layer)
        except Exception as e:
            logger.error(f"Failed to read layer file: {str(e)}")
    
    # Set system parameters
    coordinator.set_config('temperature', args.temperature)
    coordinator.set_config('freq_max', args.freq_max)
    coordinator.set_config('freq_points', args.freq_points)
    coordinator.set_config('sigma', args.sigma)
    coordinator.set_config('timestep', args.timestep)
    coordinator.set_config('n_layers', args.n_layers)
    
    timer.stop()
    logger.info(f"Data loading completed in {timer.elapsed:.3f} seconds")
    
    return True


def run_phonon_analysis(coordinator, args, output_dir):
    """
    Run phonon analysis
    
    Args:
        coordinator: PhononCoordinator instance
        args: Command-line arguments
        output_dir: Output directory path
        
    Returns:
        bool: True if at least one analysis was performed, False otherwise
    """
    # Create analyzers
    phonon_analyzer = PhononAnalyzer(coordinator)
    performed_analysis = False
    
    # Calculate VACF
    if args.velocity_autocorrelation or args.density_of_states or args.all_analyses:
        logger.info("Calculating velocity autocorrelation function (VACF)")
        timer = Timer("VACF")
        timer.start()
        
        try:
            velocities = np.array(coordinator.trajectory_data['velocities'])
            vacf = phonon_analyzer.calculate_velocity_autocorrelation(velocities)
            
            # Save VACF
            vacf_file = os.path.join(output_dir, "vacf.txt")
            np.savetxt(vacf_file, vacf, header="Time (step), VACF")
            logger.info(f"VACF saved to {vacf_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate VACF: {str(e)}")
        
        timer.stop()
        logger.info(f"VACF calculation completed in {timer.elapsed:.3f} seconds")
    
    # Calculate DOS
    if args.density_of_states or args.all_analyses:
        logger.info("Calculating phonon density of states (DOS)")
        timer = Timer("DOS")
        timer.start()
        
        try:
            if 'vacf' not in locals():
                velocities = np.array(coordinator.trajectory_data['velocities'])
                vacf = phonon_analyzer.calculate_velocity_autocorrelation(velocities)
            
            freqs, dos = phonon_analyzer.calculate_dos(vacf)
            
            # Save DOS
            dos_file = os.path.join(output_dir, "dos.txt")
            np.savetxt(dos_file, np.column_stack((freqs, dos)), 
                      header="Frequency (THz), DOS")
            logger.info(f"DOS saved to {dos_file}")
            
            # Plot DOS
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, dos)
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Density of States')
            plt.title('Phonon Density of States')
            plt.grid(True)
            dos_plot_file = os.path.join(output_dir, "dos.png")
            plt.savefig(dos_plot_file, dpi=300)
            plt.close()
            logger.info(f"DOS plot saved to {dos_plot_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate DOS: {str(e)}")
        
        timer.stop()
        logger.info(f"DOS calculation completed in {timer.elapsed:.3f} seconds")
    
    # Calculate normal modes
    if args.normal_modes or args.projected_dos or args.phonon_occupation or args.all_analyses:
        logger.info("Calculating normal modes")
        timer = Timer("NormalModes")
        timer.start()
        
        try:
            # Get positions and masses
            positions = np.array(coordinator.trajectory_data['positions'][0])
            
            # Check if we have masses
            if hasattr(coordinator, 'masses') and coordinator.masses is not None:
                masses = coordinator.masses
            else:
                # Use unit masses if not provided
                masses = np.ones(positions.shape[0])
                logger.warning("Using unit masses for normal mode calculation")
            
            # Calculate normal modes
            eigenvalues, eigenvectors = phonon_analyzer.calculate_normal_modes(positions, masses)
            
            # Save normal modes
            modes_file = os.path.join(output_dir, "normal_modes.npz")
            np.savez(modes_file, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
            logger.info(f"Normal modes saved to {modes_file}")
            
            # Save frequencies
            freqs_file = os.path.join(output_dir, "frequencies.txt")
            freqs = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
            np.savetxt(freqs_file, freqs, header="Frequency (THz)")
            logger.info(f"Frequencies saved to {freqs_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate normal modes: {str(e)}")
        
        timer.stop()
        logger.info(f"Normal modes calculation completed in {timer.elapsed:.3f} seconds")
    
    # Calculate phonon occupation
    if args.phonon_occupation or args.all_analyses:
        if not ('eigenvalues' in locals() and 'eigenvectors' in locals()):
            logger.error("Normal modes required for phonon occupation analysis")
        else:
            logger.info("Calculating phonon occupation numbers")
            timer = Timer("PhononOccupation")
            timer.start()
            
            try:
                # Get velocities
                velocities = np.array(coordinator.trajectory_data['velocities'])
                
                # Calculate occupation numbers
                occupation = phonon_analyzer.calculate_mode_occupation(
                    velocities, eigenvectors, eigenvalues, coordinator.temperature)
                
                # Save occupation numbers
                occ_file = os.path.join(output_dir, "occupation.txt")
                np.savetxt(occ_file, np.column_stack((freqs, occupation)), 
                          header="Frequency (THz), Occupation")
                logger.info(f"Phonon occupation numbers saved to {occ_file}")
                
                # Calculate theoretical occupation (Bose-Einstein)
                be_occ = phonon_analyzer.calculate_theoretical_occupation(eigenvalues, coordinator.temperature)
                
                # Save comparison
                comp_file = os.path.join(output_dir, "occupation_comparison.txt")
                np.savetxt(comp_file, np.column_stack((freqs, occupation, be_occ)), 
                          header="Frequency (THz), MD Occupation, BE Occupation")
                logger.info(f"Occupation comparison saved to {comp_file}")
                
                # Plot comparison
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.scatter(freqs, occupation, s=5, label='MD')
                plt.plot(freqs, be_occ, 'r-', label='Bose-Einstein')
                plt.xlabel('Frequency (THz)')
                plt.ylabel('Occupation Number')
                plt.title('Phonon Occupation Comparison')
                plt.legend()
                plt.grid(True)
                plt.yscale('log')
                occ_plot_file = os.path.join(output_dir, "occupation_comparison.png")
                plt.savefig(occ_plot_file, dpi=300)
                plt.close()
                logger.info(f"Occupation comparison plot saved to {occ_plot_file}")
                
                performed_analysis = True
            except Exception as e:
                logger.error(f"Failed to calculate phonon occupation: {str(e)}")
            
            timer.stop()
            logger.info(f"Phonon occupation calculation completed in {timer.elapsed:.3f} seconds")
    
    return performed_analysis


def run_thermal_analysis(coordinator, args, output_dir):
    """
    Run thermal analysis
    
    Args:
        coordinator: PhononCoordinator instance
        args: Command-line arguments
        output_dir: Output directory path
        
    Returns:
        bool: True if analysis was performed, False otherwise
    """
    if not (args.thermal_analysis or args.all_analyses):
        return False
    
    # Check if we have heat flux data
    if not hasattr(coordinator, 'heatflux_data') or coordinator.heatflux_data is None:
        logger.error("Thermal analysis requires heat flux data")
        return False
    
    logger.info("Performing thermal analysis")
    timer = Timer("ThermalAnalysis")
    timer.start()
    
    # Create thermal analyzer
    thermal_analyzer = ThermalAnalyzer(coordinator)
    
    try:
        # Calculate thermal conductivity
        kappa = thermal_analyzer.calculate_thermal_conductivity(
            coordinator.heatflux_data, 
            temperature=coordinator.temperature,
            volume=coordinator.get_volume()
        )
        
        # Save thermal conductivity
        if isinstance(kappa, tuple):
            # If kappa is a tuple of (times, kappa_values)
            times, kappa_values = kappa
            
            kappa_file = os.path.join(output_dir, "thermal_conductivity.txt")
            np.savetxt(kappa_file, np.column_stack((times, kappa_values)), 
                      header="Time (ps), Thermal Conductivity (W/mK)")
            logger.info(f"Thermal conductivity saved to {kappa_file}")
            
            # Plot thermal conductivity
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(times, kappa_values)
            plt.xlabel('Time (ps)')
            plt.ylabel('Thermal Conductivity (W/mK)')
            plt.title('Green-Kubo Thermal Conductivity')
            plt.grid(True)
            kappa_plot_file = os.path.join(output_dir, "thermal_conductivity.png")
            plt.savefig(kappa_plot_file, dpi=300)
            plt.close()
            logger.info(f"Thermal conductivity plot saved to {kappa_plot_file}")
            
            # Get final value
            final_kappa = kappa_values[-1]
        else:
            # If kappa is a single value
            final_kappa = kappa
            
            kappa_file = os.path.join(output_dir, "thermal_conductivity.txt")
            with open(kappa_file, 'w') as f:
                f.write(f"Thermal Conductivity (W/mK): {final_kappa:.6f}\n")
            logger.info(f"Thermal conductivity saved to {kappa_file}")
        
        logger.info(f"Thermal conductivity: {final_kappa:.6f} W/mK")
        
        timer.stop()
        logger.info(f"Thermal analysis completed in {timer.elapsed:.3f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to perform thermal analysis: {str(e)}")
        timer.stop()
        return False


def run_sliding_analysis(coordinator, args, output_dir):
    """
    Run sliding analysis
    
    Args:
        coordinator: PhononCoordinator instance
        args: Command-line arguments
        output_dir: Output directory path
        
    Returns:
        bool: True if analysis was performed, False otherwise
    """
    if not (args.sliding_analysis or args.stacking_analysis or args.all_analyses):
        return False
    
    logger.info("Performing sliding analysis")
    timer = Timer("SlidingAnalysis")
    timer.start()
    
    # Create sliding analyzer
    sliding_analyzer = SlidingAnalyzer(coordinator)
    
    # Detect layers if necessary
    if not hasattr(coordinator, 'layer_indices') or coordinator.layer_indices is None:
        logger.info("Detecting material layers")
        try:
            sliding_analyzer.detect_layers()
        except Exception as e:
            logger.error(f"Failed to detect layers: {str(e)}")
            timer.stop()
            return False
    
    performed_analysis = False
    
    # Calculate sliding distance
    if args.sliding_analysis or args.all_analyses:
        logger.info("Calculating sliding distance")
        try:
            distances = sliding_analyzer.calculate_sliding_distance()
            
            # Save sliding distances
            dist_file = os.path.join(output_dir, "sliding_distance.txt")
            n_frames = len(distances)
            timestep = coordinator.timestep
            times = np.arange(n_frames) * timestep
            
            np.savetxt(dist_file, np.column_stack((times, distances)), 
                      header="Time (ps), Sliding Distance (Å)")
            logger.info(f"Sliding distance saved to {dist_file}")
            
            # Plot sliding distance
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(times, distances)
            plt.xlabel('Time (ps)')
            plt.ylabel('Sliding Distance (Å)')
            plt.title('Layer Sliding Distance vs. Time')
            plt.grid(True)
            dist_plot_file = os.path.join(output_dir, "sliding_distance.png")
            plt.savefig(dist_plot_file, dpi=300)
            plt.close()
            logger.info(f"Sliding distance plot saved to {dist_plot_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate sliding distance: {str(e)}")
    
    # Calculate friction force
    if args.sliding_analysis or args.all_analyses:
        logger.info("Calculating friction force")
        try:
            forces = sliding_analyzer.calculate_friction_force()
            
            # Save friction forces
            force_file = os.path.join(output_dir, "friction_force.txt")
            n_frames = len(forces)
            timestep = coordinator.timestep
            times = np.arange(n_frames) * timestep
            
            np.savetxt(force_file, np.column_stack((times, forces)), 
                      header="Time (ps), Friction Force (eV/Å)")
            logger.info(f"Friction force saved to {force_file}")
            
            # Plot friction force
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(times, forces)
            plt.xlabel('Time (ps)')
            plt.ylabel('Friction Force (eV/Å)')
            plt.title('Friction Force vs. Time')
            plt.grid(True)
            force_plot_file = os.path.join(output_dir, "friction_force.png")
            plt.savefig(force_plot_file, dpi=300)
            plt.close()
            logger.info(f"Friction force plot saved to {force_plot_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate friction force: {str(e)}")
    
    # Calculate stacking parameters
    if args.stacking_analysis or args.all_analyses:
        logger.info("Calculating stacking parameters")
        try:
            stacking_params = sliding_analyzer.calculate_stacking_parameters()
            
            # Save stacking parameters
            param_file = os.path.join(output_dir, "stacking_parameters.txt")
            n_frames = stacking_params.shape[0]
            timestep = coordinator.timestep
            times = np.arange(n_frames) * timestep
            
            np.savetxt(param_file, np.column_stack((times, stacking_params)), 
                      header="Time (ps), registry_x, registry_y, twist_angle")
            logger.info(f"Stacking parameters saved to {param_file}")
            
            # Plot registry evolution
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(times, stacking_params[:, 0], label='x registry')
            plt.plot(times, stacking_params[:, 1], label='y registry')
            plt.xlabel('Time (ps)')
            plt.ylabel('Registry Parameter')
            plt.title('Stacking Registry Evolution')
            plt.legend()
            plt.grid(True)
            registry_plot_file = os.path.join(output_dir, "registry_evolution.png")
            plt.savefig(registry_plot_file, dpi=300)
            plt.close()
            logger.info(f"Registry evolution plot saved to {registry_plot_file}")
            
            performed_analysis = True
        except Exception as e:
            logger.error(f"Failed to calculate stacking parameters: {str(e)}")
    
    timer.stop()
    logger.info(f"Sliding analysis completed in {timer.elapsed:.3f} seconds")
    return performed_analysis


def run_equilibration_analysis(coordinator, args, output_dir):
    """
    Run equilibration analysis
    
    Args:
        coordinator: PhononCoordinator instance
        args: Command-line arguments
        output_dir: Output directory path
        
    Returns:
        bool: True if analysis was performed, False otherwise
    """
    if not (args.equilibration_analysis or args.all_analyses):
        return False
    
    # Check if we have energy data
    if not hasattr(coordinator, 'energy_data') or coordinator.energy_data is None:
        logger.error("Equilibration analysis requires energy data")
        return False
    
    logger.info("Performing equilibration analysis")
    timer = Timer("EquilibrationAnalysis")
    timer.start()
    
    # Create equilibration analyzer
    equilibration_analyzer = EquilibrationAnalyzer(coordinator)
    
    try:
        # Get energy data
        if isinstance(coordinator.energy_data, dict):
            times = coordinator.energy_data.get('time', np.arange(len(coordinator.energy_data.get('total', []))))
            energies = coordinator.energy_data.get('total', [])
        else:
            times = np.arange(len(coordinator.energy_data)) * coordinator.timestep
            energies = coordinator.energy_data
        
        # Calculate equilibration time
        equilibration_time, params = equilibration_analyzer.calculate_system_equilibration(energies, times)
        
        if equilibration_time is not None:
            # Save equilibration results
            equil_file = os.path.join(output_dir, "equilibration.txt")
            with open(equil_file, 'w') as f:
                f.write(f"Equilibration Time (ps): {equilibration_time:.6f}\n")
                if params is not None:
                    f.write(f"Fit Parameters: {params}\n")
            logger.info(f"Equilibration results saved to {equil_file}")
            
            # Plot equilibration
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(times, energies, 'o', markersize=3, label='Energy Data')
            
            if params is not None:
                # Define exponential decay function
                def exp_decay(t, A, tau, E_inf):
                    return A * np.exp(-t / tau) + E_inf
                
                # Calculate fit curve
                fit_times = np.linspace(min(times), max(times), 1000)
                fit_curve = exp_decay(fit_times, *params)
                plt.plot(fit_times, fit_curve, 'r-', label='Fitted Curve')
            
            plt.axvline(x=equilibration_time, color='g', linestyle='--',
                      label=f'Equilibration Time: {equilibration_time:.2f} ps')
            
            plt.xlabel('Time (ps)')
            plt.ylabel('Total Energy (eV)')
            plt.title('System Equilibration Analysis')
            plt.legend()
            plt.grid(True)
            equil_plot_file = os.path.join(output_dir, "equilibration.png")
            plt.savefig(equil_plot_file, dpi=300)
            plt.close()
            logger.info(f"Equilibration plot saved to {equil_plot_file}")
            
            logger.info(f"Equilibration time: {equilibration_time:.6f} ps")
        else:
            logger.warning("Failed to determine equilibration time")
        
        timer.stop()
        logger.info(f"Equilibration analysis completed in {timer.elapsed:.3f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to perform equilibration analysis: {str(e)}")
        timer.stop()
        return False


def main():
    """Main entry point for the command-line interface"""
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    setup_logger(log_level)
    
    logger.info("LAMMPhonon Analysis Tool - Command Line Interface")
    logger.info(f"Python version: {sys.version}")
    
    # Check input files
    if not check_input_files(args):
        return 1
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    if output_dir is None:
        return 1
    
    # Create PhononCoordinator
    coordinator = PhononCoordinator()
    coordinator.set_config('output_dir', output_dir)
    
    # Apply runtime patches if requested
    if args.apply_patches:
        logger.info("Applying runtime patches")
        try:
            from lammphonon.fixer import apply_all_patches
            modules = {
                'ThermalAnalyzer': ThermalAnalyzer,
                'TemporalAnalyzer': TemporalAnalyzer,
                'EquilibrationAnalyzer': EquilibrationAnalyzer
            }
            apply_all_patches(modules)
            logger.info("Runtime patches applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply runtime patches: {str(e)}")
    
    # Load data
    if not load_data(coordinator, args):
        return 1
    
    # Run analyses
    logger.info("Starting analyses")
    timer = Timer("TotalAnalysis")
    timer.start()
    
    phonon_results = run_phonon_analysis(coordinator, args, output_dir)
    thermal_results = run_thermal_analysis(coordinator, args, output_dir)
    sliding_results = run_sliding_analysis(coordinator, args, output_dir)
    equil_results = run_equilibration_analysis(coordinator, args, output_dir)
    
    timer.stop()
    
    # Check if any analysis was performed
    if not any([phonon_results, thermal_results, sliding_results, equil_results]):
        logger.warning("No analysis was performed. Please specify at least one analysis type.")
    else:
        logger.info(f"All analyses completed in {timer.elapsed:.3f} seconds")
        logger.info(f"Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 