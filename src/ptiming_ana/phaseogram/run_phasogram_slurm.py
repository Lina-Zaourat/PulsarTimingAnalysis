"""
Date: April 2026
===============================================================================
                    PHASOGRAM SLURM BATCH SUBMISSION SCRIPT
===============================================================================

Description:
    This script submits phasogram analysis jobs to SLURM for parallel processing.
    It can either:
    1. Process all DL3 FITS files found in the input directory (batch mode)
    2. Process a single run specified via the -r flag (single mode)
    3. Run interactively for a single run (with -i flag combined with -r)

    All paths and configuration parameters are read from a YAML configuration
    file.

Virtual environment:
    puslar-lst1 (via micromamba)

Usage:
    # Process all runs in batch mode
    python run_phasogram_slurm.py --config config_phasogram_SED.yaml

    # Process a single run via SLURM
    python run_phasogram_slurm.py --config config_phasogram_SED.yaml -r 01234

    # Process a single run interactively (no SLURM submission)
    python run_phasogram_slurm.py --config config_phasogram_SED.yaml -r 01234 -i
===============================================================================
"""

import os
import re
import subprocess
import argparse
import yaml


def load_config(config_file):
    """
    Load configuration parameters from a YAML file.
    
    This function reads the YAML configuration file and parses it into a
    Python dictionary.
    
    Args:
        config_file (str): Path to the YAML configuration file.
                          Supports both relative and absolute paths.
    
    Returns:
        dict: A dictionary containing all configuration parameters from the YAML file.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    
    Example:
        config = load_config('config_phasogram_SED.yaml')
        workspace = config['paths']['workspace_root']
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def generate_selection_suffix(config):
    """
    Generate a unique suffix for the output directory based on selection criteria.
    
    This function creates a suffix from the date range and zenith angle range
    to ensure that different phasogram selections don't overwrite each other.
    
    Args:
        config (dict): Configuration dictionary containing 'cuts' section.
    
    Returns:
        str: A formatted suffix string (e.g., "_zd60-75_date2019-01-01to2029-02-02")
             or empty string if cuts are not defined.
    
    Example:
        suffix = generate_selection_suffix(config)
        # Returns: "_zd60-75_date2019-01-01to2029-02-02"
    """
    cuts = config.get('cuts', {})
    
    if not cuts:
        return ""
    
    suffix_parts = []
    
    # Extract zenith distance range
    zd_range = cuts.get('zd_range', None)
    if zd_range and isinstance(zd_range, list) and len(zd_range) == 2:
        zd_min, zd_max = zd_range
        suffix_parts.append(f"zmin{zd_min}_zmax{zd_max}")
    
    # Extract date range
    date_range = cuts.get('date_range', None)
    if date_range and isinstance(date_range, list) and len(date_range) == 2:
        date_start = date_range[0].replace('-', '')  # Remove dashes for compactness
        date_end = date_range[1].replace('-', '')
        suffix_parts.append(f"firstdate{date_start}_lastdate{date_end}")
    
    # Return formatted suffix
    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""


def build_paths(config,config_file):
    """
    Construct all necessary directory paths from the configuration dictionary.
    
    This function extracts configuration parameters and builds the complete
    file system paths needed for:
    - Reading input DL3 FITS files
    - Writing output phasogram results
    - Storing SLURM job logs
    - Running the shell script
    
    The paths are constructed using os.path.join() for cross-platform compatibility
    (handles path separators correctly on Linux, Windows, macOS).
    
    Args:
        config (dict): Configuration dictionary loaded from the YAML file.
                      Must contain a 'paths' key with required subkeys.
    
    Returns:
        dict: A dictionary containing:
            - 'input_dir' (str): Directory containing input DL3 FITS files
            - 'output_dir' (str): Directory for phasogram output results
            - 'log_dir' (str): Directory for SLURM job output logs
            - 'shell_script' (str): Path to the phasogram_slurm.sh script
            - 'config_file' (str): Path to the configuration file (passed through)
    
    Example:
        paths = build_paths(config)
        print(f"Input: {paths['input_dir']}")
        print(f"Output: {paths['output_dir']}")
    """
    # Extract path configuration parameters
    paths = config['paths']
    workspace = paths['workspace_root']
    pulsar = paths['pulsar_name']
    gheff = paths['gheff_cut']
    runs_folder = paths['runs_folder_name']
    
    # ==================== GENERATE SELECTION SUFFIX ====================
    # Generate suffix based on date and zenith angle selections
    # This ensures each different selection creates a unique output folder
    selection_suffix = generate_selection_suffix(config)
    
    # ==================== BUILD INPUT DIRECTORY PATH ====================
    # Constructs the path to the directory containing processed DL3 files
    # with phase information already added (from add_DL3_phase.sh)
    # Expected structure:
    # {workspace}/data/processed/DL3/phased_pulsars/{pulsar}/{gheff_cut}/{runs_folder}/
    input_dir = os.path.join(
        workspace,
        'data/processed/DL3/phased_pulsars',
        pulsar,
        gheff,
        runs_folder
    )
    
    # ==================== BUILD OUTPUT DIRECTORY PATH ====================
    # Constructs the path where phasogram analysis results will be saved
    # (plots, tables, statistics, etc.)
    # Now INCLUDES the selection suffix (dates and zenith angle)
    # Expected structure:
    # {workspace}/results/preliminary/phasograms/{pulsar}/{gheff_cut}/{runs_folder}_phasograms{selection_suffix}/
    output_dir = os.path.join(
        workspace,
        'results/preliminary/phasograms',
        pulsar,
        gheff,
        f"{runs_folder}_phasograms{selection_suffix}/"
    )
    
    # ==================== GET OPTIONAL PATHS ====================
    # Retrieves optional paths from config with sensible defaults
    # If not specified in config, defaults to './out' for logs and './phasogram_slurm.sh'
    log_dir = paths.get('log_output_dir', './out')
    shell_script = paths.get('shell_script_path', './phasogram_slurm.sh')
    
    # ==================== RETURN PATHS DICTIONARY ====================
    # Returns all constructed paths in a single dictionary for easy access
    return {
        'input_dir': input_dir,           # DL3 input files location
        'output_dir': output_dir,         # Phasogram output results location
        'log_dir': log_dir,               # SLURM job logs location
        'shell_script': shell_script,     # Shell script to execute
        'config_file': config_file        # Configuration file (for passing to sbatch)
    }


def submit_slurm_job(shell_script_path, config_file, run_number, output_dir, log_dir):
    """
    Submit a single phasogram analysis job to SLURM.
    
    This function constructs and executes an sbatch command to submit a SLURM job.
    Each job processes a single pulsar run independently, enabling parallel execution
    of multiple runs simultaneously on the HPC cluster.
    
    Args:
        shell_script_path (str): Path to the phasogram_slurm.sh script file
        config_file (str): Path to the configuration YAML file
        run_number (str): The run number to process (e.g., '01234')
        output_dir (str): Directory where results will be saved
        log_dir (str): Directory where SLURM job logs will be stored
    
    Returns:
        int: The return code from subprocess.run (0 = success, non-zero = error)
    
    Note:
        The SLURM output filename uses %j which is replaced by SLURM with the
        actual job ID, creating unique log files for each job.
    """
    # ==================== CONSTRUCT LOG FILENAME ====================
    # %j is a SLURM placeholder that gets replaced with the actual job ID
    # This ensures each job gets a unique log file
    # Format: slurm-phasogram-{JOB_ID}-{RUN_NUMBER}.out
    output_file = os.path.join(
        log_dir,
        f"slurm-phasogram-%j-{run_number}.out"
    )
    
    # ==================== CONSTRUCT SBATCH COMMAND ====================
    # sbatch: Submit a job to SLURM
    # -o: Specify output file for job logs (stdout and stderr)
    # shell_script_path: The script to execute on the compute node
 

    cmd = [
        "sbatch",                    # SLURM batch submission command
        "-o", output_file,           # Output log file path
        shell_script_path,           # Shell script to execute
        config_file,                 # Pass config file as argument 1
        run_number,                  # Pass run number as argument 2
        output_dir                   # Pass output directory as argument 3
    ]
    
    # ==================== SUBMIT JOB TO SLURM ====================
    # Execute the sbatch command and capture the return code
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """
    Main function orchestrating the phasogram SLURM submission workflow.
    
    This function:
    1. Parses command-line arguments
    2. Loads configuration from YAML file
    3. Constructs all necessary directory paths
    4. Creates output directories if they don't exist
    5. Either submits batch jobs for all runs or processes a single run
    
    Two operational modes:
    - BATCH MODE (default): Iterates through all DL3 files in input directory,
      submits one SLURM job per file for parallel processing
    - SINGLE MODE (-r flag): Processes only the specified run, optionally
      in interactive mode (-i flag)
    """
    
    # ==================== COMMAND-LINE ARGUMENT PARSING ====================
    # Create an argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(
        description="Phasogram creation with SLURM parallelization"
    )
    
    # ==================== REQUIRED ARGUMENTS ====================
    # --config: Path to the YAML configuration file
    # This argument is REQUIRED for the script to run
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file (YAML format). Required argument.'
    )
    
    # ==================== OPTIONAL ARGUMENTS ====================
    # -r / --run_number: Process only a specific run number
    # If not provided, the script processes all runs in the input directory
    parser.add_argument(
        '-r', '--run_number',
        dest='run_number',
        default=None,
        help='Process only the specified run number (e.g., 01234). If omitted, processes all runs.'
    )
    
    # ==================== OPTIONAL FLAGS ====================
    # -i / --interactive: Run in interactive mode (no SLURM submission)
    # Only valid when combined with -r flag
    # Useful for testing or debugging a single run without SLURM
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (direct execution, no SLURM). Only valid with -r flag.'
    )
    
    # ==================== PARSE ARGUMENTS ====================
    # Parse all command-line arguments
    args = parser.parse_args()
    
    # ==================== LOAD CONFIGURATION ====================
    # Read the YAML configuration file
    config = load_config(args.config)
    
    # ==================== BUILD PATHS ====================
    # Construct all necessary directory paths from config
    paths = build_paths(config,args.config)
    
    # ==================== DISPLAY CONFIGURATION ====================
    # Print the configuration to the user for verification
    print("=" * 70)
    print("PHASOGRAM SLURM SUBMISSION CONFIGURATION")
    print("=" * 70)
    print(f"Input directory:   {paths['input_dir']}")
    print(f"Output directory:  {paths['output_dir']}")
    print(f"Log directory:     {paths['log_dir']}")
    print(f"Shell script:      {paths['shell_script']}")
    print(f"Config file:       {paths['config_file']}")
    print("=" * 70)
    
    # ==================== CREATE OUTPUT DIRECTORIES ====================
    # Create directories if they don't exist
    # exist_ok=True prevents errors if directory already exists
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(paths['log_dir'], exist_ok=True)
    print(f" Output directories created/verified")
    



    # ==================== BATCH MODE: PROCESS ALL RUNS ====================
    # If no specific run is provided, process all DL3 files in the input directory
    if not args.run_number:
        print("BATCH MODE: Processing all runs in input directory...")
        
        # ==================== ARCHIVE OLD LOG FILES ====================
        # Move previous job logs to a subdirectory to keep workspace clean
        previous_jobs_dir = os.path.join(paths['log_dir'], 'previous_jobs')
        if not os.path.exists(previous_jobs_dir):
            os.makedirs(previous_jobs_dir)
        
        # Move old .out files (from previous SLURM submissions) to archive
        # Suppress error messages if no old files exist (2>/dev/null)
        os.system(f"mv {paths['log_dir']}/*.out {previous_jobs_dir}/ 2>/dev/null")
        print(f"Old log files archived to {previous_jobs_dir}")
        
        # ==================== DEFINE FILENAME PATTERN ====================
        # Regular expression to match DL3 FITS filenames
        # Pattern: dl3_LST-1.Run{digits}_{optional_parameters}.fits
        # Examples that match:
        #   - dl3_LST-1.Run01234.fits
        #   - dl3_LST-1.Run01234_gh0.7_th0.2.fits
        # The pattern captures:
        #   Group 1: Run number (digits only)
        #   Group 2: Optional parameters after the run number
        pattern = re.compile(r"dl3_LST-1\.Run(\d+)(?:_(.*?))?\.fits$")
        
        # ==================== VERIFY INPUT DIRECTORY EXISTS ====================
        # Check if the input directory exists before attempting to read it
        if not os.path.exists(paths['input_dir']):
            print(f"ERROR: Input directory does not exist: {paths['input_dir']}")
            return
        
        # ==================== ITERATE THROUGH ALL FILES ====================
        # Loop through all files in the input directory
        run_count = 0
        for filename in os.listdir(paths['input_dir']):
            # Try to match the filename against the pattern
            match = pattern.match(filename)
            
            # ==================== PROCESS MATCHING FILES ====================
            # If the filename matches the DL3 pattern, extract the run number
            if match:
                run_number = match.group(1)  # Extract run number from filename
                run_count += 1
                print(f"Submitting job for run: {run_number}")
                
                # Submit a SLURM job for this run
                return_code = submit_slurm_job(
                    paths['shell_script'],
                    paths['config_file'],
                    run_number,
                    paths['output_dir'],
                    paths['log_dir']
                )
                
                # Provide feedback on job submission
                if return_code == 0:
                    print(f"Job submitted successfully")
                else:
                    print(f"Failed to submit job (return code: {return_code})")
        
        # ==================== SUMMARY ====================
        # Print summary of batch submission
        print("\n" + "=" * 70)
        print(f"BATCH SUBMISSION COMPLETE: {run_count} jobs submitted")
        print("=" * 70)
    



    # ==================== SINGLE MODE: PROCESS ONE RUN ====================
    # If a specific run is provided, process only that run
    else:
        print(f"SINGLE MODE: Processing run {args.run_number}")
        
        # ==================== NORMALIZE RUN NUMBER ====================
        # If run number is 4 digits, prepend '0' to make it 5 digits
        # This ensures consistency with the DL3 filename format
        run_number = args.run_number
        if len(run_number) == 4:
            run_number = "0" + run_number
        
        print(f"Run number (normalized): {run_number}")
        



        # ==================== INTERACTIVE MODE ====================
        # If -i flag is provided, execute the script directly without SLURM
        if args.interactive:
            print("Running in INTERACTIVE mode (no SLURM submission)")
            
            # Execute the shell script directly in this terminal
            # Arguments:
            #   1. shell_script_path: Path to phasogram_slurm.sh
            #   2. config_file: Configuration YAML file
            #   3. run_number: The run to analyze
            #   4. output_dir: Where to save results
            cmd = [
                paths['shell_script'],
                paths['config_file'],
                run_number,
                paths['output_dir']
            ]
            
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print(f"Interactive execution completed successfully")
            else:
                print(f"Interactive execution failed (return code: {result.returncode})")
        




        # ==================== SLURM SUBMISSION MODE ====================
        # Submit the job to SLURM (default behavior for single run)
        else:
            print("Submitting to SLURM")
            
            return_code = submit_slurm_job(
                paths['shell_script'],
                paths['config_file'],
                run_number,
                paths['output_dir'],
                paths['log_dir']
            )
            
            if return_code == 0:
                print(f"Job submitted successfully to SLURM")
            else:
                print(f"Failed to submit job (return code: {return_code})")


# ==================== SCRIPT ENTRY POINT ====================
# This block ensures main() is only called when the script is run directly,
# not when it's imported as a module in another script
if __name__ == "__main__":
    main()