#!/usr/bin/env python3
"""
Date: April 2026
===============================================================================
                    SPECTRAL ANALYSIS (SED) SCRIPT
===============================================================================

Description:
    This script performs spectral energy distribution (SED) analysis on pulsar
    data. It handles:
    - Loading DL3 FITS files
    - Creating or loading DL4 datasets
    - Fitting spectral models (PowerLaw, etc.)
    - Computing flux points
    - Generating SED plots with residuals
    - Creating contour plots for fitted parameters

Virtual environment:
    pulsar-lst1 (via micromamba)

Usage:
    # Basic usage with P1 phase region
    python run_sed.py --config_path config_phasogram_SED.yaml

    # Specify phase region (P1, P2, or P3)
    python run_sed.py --config_path config_phasogram_SED.yaml --peak P1

    # Specify output directory (override config)
    python run_sed.py --config_path config_phasogram_SED.yaml --output_dir ./results/

===============================================================================
"""

import os
import argparse
import logging
import yaml
from datetime import datetime

# Try absolute import first (when run directly), fall back to relative import
try:
    from ptiming_ana.spectral.spectra import SpectralPulsarAnalysis
    from ptiming_ana.spectral.path_spectra_utils import build_spectra_output_dir
except (ImportError, ModuleNotFoundError):
    try:
        from spectra import SpectralPulsarAnalysis
        from path_spectra_utils import build_spectra_output_dir
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Could not import spectral analysis modules. Make sure PulsarTimingAnalysis is installed.")



# --- Logging Configuration ---
# This sets up a log file to record all messages (INFO, WARNING, ERROR, etc.)
# during script execution. Useful for debugging and monitoring, especially on SLURM.
logging.basicConfig(
    level=logging.INFO,  # Minimum level of messages to log (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format: timestamp - level - message
    filename='pulsar_sed_analysis.log'  # Log file name
)

logger = logging.getLogger(__name__)  # Create a logger instance


def main(config_path, peak=None, output_dir=None):
    """
    Main function to perform spectral analysis.
    
    Parameters:
        config_path (str): Path to the configuration YAML file (required).
        peak (str): Phase region to analyze ('P1', 'P2', 'P3', etc.).
                   If None, reads from config (reader.selected_peak). Default is None.
        output_dir (str): Optional output directory for results. If not provided, 
                         paths are auto-computed from config.
    """
    
    try:
        # --- Initialization ---
        logger.info("=" * 80)
        logger.info("SPECTRAL ENERGY DISTRIBUTION (SED) ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # --- Load Configuration ---
        logger.info(f"Loading configuration from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
        # --- Display Analysis Parameters ---
        # Read peak from config if not provided via CLI
        if peak is None:
            peak = conf.get("reader", {}).get("selected_peak", "P1")
            logger.info(f"Phase region selected from config: {peak}")
        else:
            logger.info(f"Phase region overridden from CLI: {peak}")
        
        if peak not in conf.get("phase_regions", {}):
            raise ValueError(f"Phase region '{peak}' not found in configuration. Available: {list(conf.get('phase_regions', {}).keys())}")
        logger.info(f"Phase range: {conf['phase_regions'][peak]}")
        
        target_name = conf.get("target", {}).get("name", "Unknown")
        logger.info(f"Target pulsar: {target_name}")
        
        # --- Output Directory Setup ---
        if output_dir is None:
            # Auto-compute output directory from config
            logger.info("Computing output directory from configuration...")
            output_dir = build_spectra_output_dir(conf)
            logger.info(f"Auto-computed output directory: {output_dir}")
        else:
            logger.info(f"Output directory provided via CLI: {output_dir}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        else:
            logger.info(f"Output directory exists: {output_dir}")
        
        # --- Spectral Analysis Initialization ---
        logger.info("Initializing SpectralPulsarAnalysis...")
        spectra = SpectralPulsarAnalysis()
        
        # --- Configuration Setup ---
        logger.info("Initializing analysis settings from configuration...")
        spectra.init_settings(configuration_file=config_path)
        logger.info("Analysis settings initialized")
        
        # --- Run Analysis ---
        logger.info(f"Starting spectral analysis for phase region: {peak}")
        logger.info("This may take several minutes depending on data size...")
        spectra.run(peak=peak)
        logger.info(f"Spectral analysis completed for {peak}")
        
        # --- Results Extraction ---
        logger.info("Extracting analysis results...")
        
        # Get fitted model parameters
        spectral_model_info = spectra.model_best.spectral_model.parameters.to_table()
        logger.info(f"Fitted spectral model parameters:")
        for param in spectral_model_info['name']:
            idx = list(spectral_model_info['name']).index(param)
            value = spectral_model_info['value'][idx]
            logger.info(f"  {param}: {value}")
        
        # Get flux points
        flux_points_table = spectra.flux_points.to_table('e2dnde')
        n_flux_points = len(flux_points_table)
        logger.info(f"Generated {n_flux_points} flux points")
        
        # --- Save Results ---
        logger.info("Saving analysis results...")
        
        # Save spectral model parameters with peak identifier
        model_table_path = os.path.join(output_dir, f'spectral_parameters_{peak}.fits')
        spectral_model_info.write(model_table_path, format='fits', overwrite=True)
        logger.info(f"Saved spectral parameters to: {model_table_path}")
        
        # Save flux points with peak identifier
        flux_points_path = os.path.join(output_dir, f'flux_points_{peak}.fits')
        flux_points_table.write(flux_points_path, format='fits', overwrite=True)
        logger.info(f"Saved flux points to: {flux_points_path}")
        
        # --- Summary Statistics ---
        logger.info("=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Target: {target_name}")
        logger.info(f"Phase region: {peak}")
        logger.info(f"Fitted spectral index: {spectral_model_info['value'][0]:.3f}")
        logger.info(f"Number of flux points: {n_flux_points}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("Analysis completed successfully!")
        
        return 0
    
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        logger.error("Analysis failed. See traceback above for details.")
        return 1


if __name__ == "__main__":
    # --- Argument Parsing ---
    # Parse command-line arguments for flexibility (e.g., SLURM jobs).
    parser = argparse.ArgumentParser(
        description="Pulsar SED (Spectral Energy Distribution) analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sed.py --config_path config_phasogram_SED.yaml
  python run_sed.py --config_path config_phasogram_SED.yaml --peak P1
  python run_sed.py --config_path config_phasogram_SED.yaml --peak P3 --output_dir ./results/
        """
    )
    
    # ==================== REQUIRED ARGUMENTS ====================
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration YAML file (required)"
    )
    
    # ==================== OPTIONAL ARGUMENTS ====================
    parser.add_argument(
        "--peak",
        type=str,
        default=None,
        help="Phase region to analyze: P1, P2, P3, etc. If not provided, uses value from config file."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results. If not provided, auto-computed from config."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # --- Main Execution ---
    # Call the main function with parsed arguments.
    main(args.config_path, args.peak, args.output_dir)
 