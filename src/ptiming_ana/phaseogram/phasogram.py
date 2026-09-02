#!/usr/bin/env python3

import os
import argparse
import logging
import time  # (LBZ)
from ptiming_ana.phaseogram import PulsarAnalysis
from lstchain.io.io import dl2_params_lstcam_key, dl2_params_src_dep_lstcam_key
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lstchain
import gammapy
import yaml

# (LBZ) Attempt absolute import first (when run directly), fall back to relative import (when run as module)
try:
    from ptiming_ana.phaseogram.path_phasogram_utils import build_phasogram_output_dir, output_file_from_dir
except (ImportError, ModuleNotFoundError):
    try:
        from .path_phasogram_utils import build_phasogram_output_dir, output_file_from_dir
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Could not import path_phasogram_utils. Make sure PulsarTimingAnalysis is installed.") 

# --- Logging Configuration ---
# This sets up a log file to record all messages (INFO, WARNING, ERROR, etc.)
# during script execution. Useful for debugging and monitoring, especially on SLURM.
logging.basicConfig(
    level=logging.INFO,  # Minimum level of messages to log (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s [%(name)s] %(message)s',  # Format: timestamp - level - message
    filename='pulsar_analysis.log'  # Log file name
)

logger = logging.getLogger(__name__)  # Create a logger instance


class Timer:  # (LBZ) Simple timing utility
    """Track execution time of different pipeline stages."""
    def __init__(self):
        self.stages = {}
        self.start_time = time.time()
        self.prev_time = 0
    
    def mark(self, stage_name):
        """Record the current time for a stage."""
        elapsed = time.time() - self.start_time
        stage_duration = elapsed - self.prev_time  # Time for this stage only
        self.stages[stage_name] = elapsed
        logger.info(f"[TIMING] {stage_name}: {stage_duration:.2f}s (cumulative: {elapsed:.2f}s)")
        self.prev_time = elapsed
    
    def report(self):
        """Print a summary of all stages."""
        total = time.time() - self.start_time
        logger.info("=" * 70)
        logger.info("EXECUTION TIME BREAKDOWN:")
        logger.info("=" * 70)
        prev_time = 0
        for stage, elapsed in self.stages.items():
            stage_duration = elapsed - prev_time
            logger.info(f"  {stage:30s}: {stage_duration:8.2f}s (cumulative: {elapsed:.2f}s)")
            prev_time = elapsed
        logger.info(f"  {'TOTAL':30s}: {total:8.2f}s")
        logger.info("=" * 70)


def main(config_path, output_dir=None):
    """
    Main function to generate phasograms.
    Parameters:
        config_path (str): Path to the configuration file.
        output_dir (str): Optional output directory for results. If None, plots are not saved.
    """
    timer = Timer()  # (LBZ) Initialize timing tracker
    
    # --- Pulsar Analysis ---
    # Initialize the PulsarAnalysis object, set the config, and run the analysis.
    h = PulsarAnalysis()
    h.set_config(config_path)
    timer.mark("01_config_setup")  # (LBZ)
    
    # Get built_paths from the PulsarAnalysis object (already computed in set_config)
    built_paths = h.built_paths  # (LBZ)
    
    # If output_dir was provided via CLI, use it as the base; otherwise use the auto-computed one
    if output_dir is not None:
        # CLI-provided output_dir takes priority for the directory structure
        output_file = os.path.join(output_dir, os.path.basename(built_paths["output_file"]))
        logger.info(f"Using CLI output_dir with auto-generated filename: {output_file}")
    else:
        # Auto-computed path (from config)
        output_dir = built_paths["output_dir"]
        output_file = built_paths["output_file"]
        logger.info(f"Auto-computed output directory from config: {output_dir}")
        logger.info(f"Auto-computed output file with postcuts: {output_file}")

    # Setup output file for PulsarAnalysis object (output_dir is guaranteed non-None at this point)
    h.output_file = output_file
    h.output_dir = os.path.dirname(output_file)
    h.get_results = True

    if not os.path.exists(h.output_dir):
        logger.info(f"Creating output directory: {h.output_dir}")
        os.makedirs(h.output_dir, exist_ok=True)

    logger.info(f"Output file: {h.output_file}")
    timer.mark("02_path_setup")  # (LBZ)

    # (LBZ) ============= CACHE MANAGEMENT =============
    cache_settings = dict(yaml.safe_load(open(config_path, 'rb')).get('cache', {}))
    
    # Determine cache file path (in phasograms folder, takes into account all cuts)
    if cache_settings.get('cache_file') is None:
        phasograms_dir = os.path.dirname(os.path.dirname(h.output_dir))  # (LBZ)
        cache_settings['cache_file'] = os.path.join(phasograms_dir, 'pulsar_data_cache.pkl')  # (LBZ)
    
    # Log cache configuration
    logger.info("=" * 70)
    logger.info("CACHE CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"use_cache: {cache_settings.get('use_cache', False)}")
    logger.info(f"save_cache: {cache_settings.get('save_cache', True)}")
    logger.info(f"cache_file: {cache_settings['cache_file']}")
    cache_exists = os.path.exists(cache_settings['cache_file'])
    logger.info(f"cache_file_exists: {cache_exists}")
    logger.info("=" * 70)
    
    # Check if we should load from cache
    use_cache = cache_settings.get('use_cache', False)
    save_cache = cache_settings.get('save_cache', True)
    
    if use_cache and cache_exists:
        # Case 1: use_cache=True AND cache file exists → Load from cache
        logger.info("=" * 70)
        logger.info("LOADING DATA FROM CACHE (skipping read/filter)")
        logger.info("=" * 70)
        # Load config setup first
        if h.load_cache(cache_settings['cache_file']):
            logger.info("Cache loaded successfully")
            # Setup regions and binning with cached data
            h.setup_regions_and_binning()
            # Continue with stats and fitting using cached data
            h.execute_stats(h.tobs)
            
            # Energy analysis if enabled
            try:
                logger.info("Performing energy-dependent analysis...")
                h.EnergyAna.run(h)
            except AttributeError:
                logger.warning(
                    "No Energy Analysis was performed. Check that you set the right energy params (test:phasogram.py)"
                )
            
            if h.get_results:
                h.save_results()
                
            logger.info("Global analysis completed (from cache).")
            timer.mark("03_cache_load_analysis")
        else:
            logger.warning("Cache load failed. Proceeding with normal read/filter...")
            h.run()
            timer.mark("03_core_analysis")
    elif use_cache and not cache_exists:
        # Case 2: use_cache=True BUT cache file does NOT exist → Normal analysis
        logger.info("=" * 70)
        logger.info("LOADING DATA FROM CACHE BUT CACHE FILE DOES NOT EXIST")
        logger.info("=" * 70)
        logger.warning(f"Cache file not found at: {cache_settings['cache_file']}")
        logger.info("Proceeding with NORMAL ANALYSIS (reading and filtering data)")
        logger.info("=" * 70)
        h.run()
        timer.mark("03_core_analysis")
        
        # Save cache if enabled
        if save_cache:
            logger.info("=" * 70)
            logger.info("SAVING DATA TO CACHE FOR NEXT RUNS")
            logger.info("=" * 70)
            h.save_cache(cache_settings['cache_file'])
            logger.info(f"Cache saved to: {cache_settings['cache_file']}")
            logger.info("For next runs, keep cache.use_cache=True to load from cache")
            timer.mark("03_cache_save")
        else:
            logger.info("CASE 2 CONTINUED: NOT SAVING CACHE")
            logger.info("o save cache for future runs, set save_cache=True in config")
    else:
        # Case 3: use_cache=False → Normal analysis (regardless of cache existence)
        logger.info("=" * 70)
        logger.info("NORMAL ANALYSIS")
        logger.info("=" * 70)
        logger.info("Proceeding with NORMAL ANALYSIS (reading and filtering data)")
        logger.info("=" * 70)

        h.run()
        timer.mark("03_core_analysis")
        
        # Save cache if enabled
        if save_cache:
            logger.info("=" * 70)
            logger.info("SAVING DATA TO CACHE FOR NEXT RUNS")
            logger.info("=" * 70)
            h.save_cache(cache_settings['cache_file'])
            logger.info(f"Cache saved to: {cache_settings['cache_file']}")
            logger.info("For next runs, set cache.use_cache=True to load from cache")
            timer.mark("03_cache_save")
        else:
            logger.info("NOT SAVING CACHE")
            logger.info("To save cache for future runs, set save_cache=True in config")

        # --- Phaseogram Creation ---
        # Generate and save the phaseogram plot.
        phaseogram = h.draw_phaseogram(phase_limits=[0, 2], colorhist='xkcd:baby blue', stats='long')
        #plt.savefig(f"{results_path}/phasogram.png", format="png", dpi=300, bbox_inches="tight")
        #plt.savefig(f"{results_path}/phasogram.pdf", format="pdf", dpi=300, bbox_inches="tight")
        #plt.close()  # Close the plot to free memory
        logger.info("Phasogram saved.")

        # --- Statistics ---
        # Display and log peak and periodicity results.
        results_peaks, results_periodicity = h.show_Presults()
        #logger.info(f"Nexcess (P1): {h.regions.P1.Nex:.1f}")
        #logger.info(f"Significance (P1): {h.regions.P1.sign:.1f}")
        #logger.info(f"Nexcess (P2): {h.regions.P2.Nex:.1f}")
        #logger.info(f"Significance (P2): {h.regions.P2.sign:.1f}")
        timer.mark("04_statistics")  # (LBZ)

        # --- Fit Results ---
        # Display and log fit results.
        fit_result = h.show_fit_results()
        logger.info("Fit results displayed.")
        timer.mark("05_fit_results")  # (LBZ)

        # --- Fitted Phaseogram ---
        # Generate and save the fitted phaseogram.
        phaseogram = h.draw_phaseogram(phase_limits=[0, 2], colorhist='xkcd:baby blue', fit=True)
        # plt.savefig(f"{results_path}/fitted_phasogram.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/fitted_phasogram.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Fitted phasogram saved.")

        # --- P1 Significance vs Time ---
        # Plot and save the P1 significance over time.
        time_binning = h.TimeEv.t
        P1_sign = h.TimeEv.P1sTime
        plt.plot(time_binning, P1_sign, 'o-', color='C1')
        plt.xlabel('Time of observation (s)')
        plt.ylabel('Significance (sigma)')
        plt.title('P1 Significance vs Time')
        plt.grid()
        # plt.savefig(f"{results_path}/P1_significance_vs_time.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/P1_significance_vs_time.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("P1 Significance vs Time plot saved.")

        # --- Binned Energy Integrated Phaseogram ---
        # Generate and save the binned energy integrated phaseogram.
        h.show_lcVsEnergy()
        h.show_lcVsEnergy(integral=None)
        # plt.savefig(f"{results_path}/binned_energy_integrated_phasogram.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/binned_energy_integrated_phasogram.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Binned energy integrated phaseogram saved.")
        timer.mark("06_energy_analysis")  # (LBZ)

        # --- Energy Statistics ---
        # Display and log energy bin statistics.
        energy_results = h.show_EnergyPresults()
        # energy_results is now a list of (peak_stat, p_stat) tuples for each energy bin
        if energy_results:
            peak_stats_bin1 = energy_results[0][0]
            peak_stats_bin2 = energy_results[0][1]
            logger.info(f"First energy bin peak statistics:\n{peak_stats_bin1}")
            logger.info(f"Total energy bins processed: {len(energy_results)}")
        else:
            logger.warning("No valid energy bin statistics to display")
        logger.info(f"First energy bin: {h.EnergyAna.energy_edges[0]*1000:.1f} GeV - {h.EnergyAna.energy_edges[1]*1000:.1f} GeV")

        # --- (Sigma, FWMH, P1/P2) vs Energy ---
        # Generate and save the (Sigma, FWMH, P1/P2) vs Energy plot.
        # Functions manage do_integral/do_diff internally, no need to pass parameter
        energy_plots = h.show_EnergyAna()
        # plt.savefig(f"{results_path}/sig_fwmh_p1/P2_VS_energy.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/sig_fwmh_p1/P2_VS_energy.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("(Sigma, FWMH, P1/P2) vs Energy plot saved.")

        # --- P1/P2 vs Energy ---
        # Generate and save the P1/P2 vs Energy plot.
        p1p2_plot = h.show_P1P2VsEnergy()
        # plt.savefig(f"{results_path}/p1/P2_VS_energy.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/p1/P2_VS_energy.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("P1/P2 vs Energy plot saved.")

        # --- Mean Phase vs (Phase, Energy) ---
        # Generate and save the Mean Phase vs (Phase, Energy) plot.
        mean_energy_plot = h.show_meanVsEnergy()
        # plt.savefig(f"{results_path}/meanPhase_VS_phase_energy.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{results_path}/meanPhase_VS_phase_energy.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Mean Phase vs (Phase, Energy) plot saved.")

        # --- Energy Fit Results ---
        # Display and log fit results per energy bin.
        fit_results_vs_energy = h.show_EnergyFitresults()
        logger.info("Fit results per energy bin displayed.")
        
        timer.report()  # (LBZ) Print timing summary

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Parse command-line arguments for flexibility (e.g., SLURM jobs).
    parser = argparse.ArgumentParser(description="Pulsar phasogram creation")
    #parser.add_argument("--workspace_dir", type=str, required=True, help="Path to the workspace directory.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration .yaml file.")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory for results. If not provided, plots are not saved.")
    #parser.add_argument("--results_output_name", type=str, required=True, help="Name of the results output directory.")
    #parser.add_argument("--gheff_cut", type=str, required=True, help="Gamma-ray efficiency cut (e.g., 'gheffcut_0.9').")
    parser.add_argument('-r', '--run_number', dest='run_number', help='Create phasogram for the specified run')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode, only valid if the option -r')
    args = parser.parse_args()

    # --- Main Execution ---
    # Call the main function with parsed arguments.
    main(args.config_path, args.output_dir)

