"""
Path construction utilities for spectral analysis results.
"""

import os
import logging

LOG_FORMAT = "%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def build_spectra_output_dir(config):
    """
    Build the spectral analysis output directory path from config.
    
    Structure:
    results/preliminary/Crab/gheffcut_0.9/thetacont_0.7/{runs_folder}_postcuts_{cuts}/spectra/
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (typically from YAML config)
        
    Returns
    -------
    str
        Full path to the spectral output directory
    """
    # Extract path configuration
    paths = config.get("paths", {})
    
    # Get workspace and relative/absolute output root
    workspace = paths.get("workspace_root")
    output_rel_dir = paths.get("output_rel_dir", "results/preliminary")
    output_root_dir = paths.get("output_root_dir")
    
    if not output_root_dir:
        if not workspace:
            raise ValueError("Either workspace_root or output_root_dir must be set in config")
        output_root_dir = os.path.join(workspace, output_rel_dir)
    
    # Get parameters for path construction
    pulsar = paths.get("pulsar_name", "Crab")
    gheff = paths.get("gheff_cut", "gheffcut_0.9")
    theta_cont = paths.get("theta_cont", "thetacont_0.7")
    runs_folder = paths.get("runs_folder_name", "")
    
    # Generate selection suffix from cuts (z-range and date-range)
    selection_suffix = _generate_selection_suffix(config)
    
    # Build the full output directory path
    # Structure: results/preliminary/Crab/gheffcut_0.9/thetacont_0.7/{runs_folder}_postcuts_{cuts}/spectra/
    output_dir = os.path.join(
        output_root_dir,  # results/preliminary
        pulsar,           # Crab
        gheff,            # gheffcut_0.9
        theta_cont,       # thetacont_0.7
        f"{runs_folder}_postcuts{selection_suffix}",  # folder_postcuts{selection_suffix}
        "spectra"         # spectra subdirectory
    )
    
    logger.info(f"Spectral output directory: {output_dir}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        logger.info(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Return with trailing slash for convenience
    return output_dir if output_dir.endswith("/") else output_dir + "/"



def _generate_selection_suffix(config):
    """
    Generate a stable suffix from cuts settings (zenith/date).
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    str
        Selection suffix like "_zmin55_zmax75_firstdate20191214_lastdate20251229"
        or empty string if cuts not specified
    """
    cuts = config.get("cuts", {})
    suffix_parts = []
    
    # Add zenith angle range if present
    zd_range = cuts.get("zd_range")
    if isinstance(zd_range, (list, tuple)) and len(zd_range) == 2:
        suffix_parts.append(f"zmin{int(zd_range[0])}_zmax{int(zd_range[1])}")
    
    # Add date range if present
    date_range = cuts.get("date_range")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_start = str(date_range[0]).replace("-", "")
        date_end = str(date_range[1]).replace("-", "")
        suffix_parts.append(f"firstdate{date_start}_lastdate{date_end}")
    
    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""
