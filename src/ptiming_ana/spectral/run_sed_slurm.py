#!/usr/bin/env python3
"""
Date: April 2026
===============================================================================
                 SPECTRAL ANALYSIS (SED) SLURM SUBMISSION SCRIPT
===============================================================================

Description:
    This script submits spectral analysis (SED) jobs to SLURM for processing.
    It supports:
    1. Submitting a single SED analysis job for a specific peak
    2. Submitting multiple jobs for different peaks (P1, P2, P3, P1+P2, P1+P2+P3)

Virtual environment:
    pulsar-lst1 (via micromamba)

Usage:
    # Submit a single SED analysis job (uses peak from config)
    python run_sed_slurm.py --config config_phasogram_SED.yaml

    # Submit job for a specific peak
    python run_sed_slurm.py --config config_phasogram_SED.yaml --peak P1

    # Submit jobs for multiple peaks
    python run_sed_slurm.py --config config_phasogram_SED.yaml --peaks P1 P2 P3

    # Specify custom output directory
    python run_sed_slurm.py --config config_phasogram_SED.yaml --peak P1 --output_dir ./results/custom/

===============================================================================
"""

import os
import subprocess
import argparse
import yaml
import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-6s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def get_shell_script_path(script_dir):
    """Get the path to the SLURM shell script."""
    return os.path.join(script_dir, 'run_sed_slurm.sh')


def create_log_dir(config, config_file):
    """Create and return the log directory for SED analysis.
    
    Adds 'sed_script_outputs' subdirectory to the base log directory from config.
    """
    paths = config.get('paths', {})
    log_base = paths.get('log_output_dir')
    
    if log_base:
        log_base = os.path.expanduser(log_base)
        log_dir = os.path.join(log_base, 'sed_script_outputs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    else:
        # Fallback: create logs directory in slurm_outputs/sed_script_outputs
        config_dir = os.path.dirname(os.path.abspath(config_file))
        log_dir = os.path.join(config_dir, '..', '..', '..', 'slurm_outputs', 'sed_script_outputs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


def submit_slurm_job(shell_script_path, config_file, peak, output_dir, log_dir):
    """
    Submit a single SED analysis job to SLURM.
    
    Args:
        shell_script_path (str): Path to run_sed_slurm.sh
        config_file (str): Path to YAML configuration file
        peak (str): Peak region to analyze (e.g., 'P1', 'P2', 'P1+P2')
        output_dir (str or None): Output directory (None to use config default)
        log_dir (str): Directory for SLURM logs
    
    Returns:
        int: Return code (0 = success)
    """
    # Create unique log filename with peak identifier
    log_file = os.path.join(log_dir, f"slurm-sed-%j-{peak}.out")
    
    # Build sbatch command
    cmd = [
        "sbatch",
        "-o", log_file,
        shell_script_path,
        config_file,
        peak,
    ]
    
    # Add output directory if specified
    if output_dir:
        cmd.append(output_dir)
    
    logger.info(f"Submitting SED job for peak: {peak}")
    #logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info(f"SED job for {peak} submitted successfully")
    else:
        logger.error(f"Failed to submit SED job for {peak}")
    
    return result.returncode


def main():
    """Main function orchestrating the SED SLURM submission workflow."""
    
    parser = argparse.ArgumentParser(
        description="Submit SED spectral analysis jobs to SLURM"
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file (YAML format). Required.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--peak',
        type=str,
        default=None,
        help='Peak region to analyze (P1, P2, P3, P1+P2, P1+P2+P3). If omitted, uses config value.'
    )
    
    parser.add_argument(
        '--peaks',
        type=str,
        nargs='+',
        default=None,
        help='Multiple peaks to analyze. Example: --peaks P1 P2 P3'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config.'
    )
    
    # parser.add_argument(
    #     '--verbose', '-v',
    #     action='store_true',
    #     help='Enable verbose logging.'
    # )
    
    args = parser.parse_args()
    
    # # Enable verbose logging if requested
    # if args.verbose:
    #     logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.peaks and args.peak:
        logger.error("Cannot specify both --peak and --peaks")
        return 1
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Get script paths
    script_dir = get_script_dir()
    shell_script = get_shell_script_path(script_dir)
    
    if not os.path.exists(shell_script):
        logger.error(f"SLURM shell script not found: {shell_script}")
        return 1
    
    # Create log directory
    log_dir = create_log_dir(config, args.config)
    logger.info(f"Using log directory: {log_dir}")
    
    # Determine which peaks to process
    if args.peaks:
        peaks = args.peaks
        logger.info(f"Processing multiple peaks: {', '.join(peaks)}")
    elif args.peak:
        peaks = [args.peak]
        logger.info(f"Processing single peak: {args.peak}")
    else:
        # Use peak from config
        peak_from_config = config.get('reader', {}).get('selected_peak', 'P1')
        peaks = [peak_from_config]
        logger.info(f"Using peak from config: {peak_from_config}")
    
    # Process each peak
    return_codes = []
    
    for peak in peaks:
        try:
            returncode = submit_slurm_job(shell_script, args.config, peak, args.output_dir, log_dir)
            return_codes.append(returncode)
        except Exception as e:
            logger.error(f"Error processing peak {peak}: {e}")
            return_codes.append(1)
    
    # Summary
    successful = sum(1 for rc in return_codes if rc == 0)
    total = len(return_codes)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SLURM submission summary: {successful}/{total} submitted")
    logger.info(f"Check logs in: {log_dir}")
    logger.info(f"{'='*80}")
    
    # Return 0 if all succeeded
    return 0 if all(rc == 0 for rc in return_codes) else 1


if __name__ == "__main__":
    exit(main())
