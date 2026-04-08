#!/usr/bin/env python3

import os
import argparse
import logging
from ptiming_ana.phaseogram import PulsarAnalysis
from lstchain.io.io import dl2_params_lstcam_key, dl2_params_src_dep_lstcam_key
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lstchain
import gammapy

# --- Logging Configuration ---
# This sets up a log file to record all messages (INFO, WARNING, ERROR, etc.)
# during script execution. Useful for debugging and monitoring, especially on SLURM.
logging.basicConfig(
    level=logging.INFO,  # Minimum level of messages to log (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format: timestamp - level - message
    filename='pulsar_analysis.log'  # Log file name
)

logger = logging.getLogger(__name__)  # Create a logger instance

def main(config_path, output_dir=None):
    """
    Main function to generate phasograms.
    Parameters:
        config_path (str): Path to the configuration file.
        output_dir (str): Optional output directory for results. If None, plots are not saved.
    """
    # --- Output Paths ---
    # Construct the full path for results using the provided arguments.
    #global_results_path = f'/fefs/aswg/workspace/lina.bretonzaourat/Crab_analysis_DL3_sourceIndep/results/preliminary/phasograms/Crab/{gheff_cut}/'
    #results_path = os.path.join(global_results_path, results_output_name)
    #os.makedirs(results_path, exist_ok=True)  # Create directory if it doesn't exist
    #logger.info(f"Results directory: {results_path}")

    # --- Pulsar Analysis ---
    # Initialize the PulsarAnalysis object, set the config, and run the analysis.
    h = PulsarAnalysis()
    h.set_config(config_path)
    h.run()  # This performs the core analysis
    logger.info("Global analysis completed.")

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
    logger.info(f"Nexcess (P1): {h.regions.P1.Nex:.1f}")
    logger.info(f"Significance (P1): {h.regions.P1.sign:.1f}")
    logger.info(f"Nexcess (P2): {h.regions.P2.Nex:.1f}")
    logger.info(f"Significance (P2): {h.regions.P2.sign:.1f}")

    # --- Fit Results ---
    # Display and log fit results.
    fit_result = h.show_fit_results()
    logger.info("Fit results displayed.")

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
    h.show_lcVsEnergy(integral=True)
    # plt.savefig(f"{results_path}/binned_energy_integrated_phasogram.png", format="png", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{results_path}/binned_energy_integrated_phasogram.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Binned energy integrated phaseogram saved.")

    # --- Energy Statistics ---
    # Display and log energy bin statistics.
    energy_results = h.show_EnergyPresults(integral=True)
    peak_stats_bin1 = energy_results[0][0]
    peak_stats_bin2 = energy_results[0][1]
    logger.info(f"First energy bin: {h.EnergyAna.energy_edges[0]*1000:.1f} GeV - {h.EnergyAna.energy_edges[1]*1000:.1f} GeV")

    # --- (Sigma, FWMH, P1/P2) vs Energy ---
    # Generate and save the (Sigma, FWMH, P1/P2) vs Energy plot.
    energy_plots = h.show_EnergyAna(integral=True)
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

