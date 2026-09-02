import os


def build_phasogram_output_dir(config, config_file=None, script_dir=None):
    """Build phasogram output directory and result file path from the config paths section."""
    paths = config.get("paths", {})
    log_base = paths.get("log_output_dir", "./out")
    log_dir = os.path.join(log_base, "phasogram_script_outputs")

    workspace = paths.get("workspace_root")
    pulsar = paths.get("pulsar_name")
    gheff = paths.get("gheff_cut")
    theta_cont = paths.get("theta_cont")
    runs_folder = paths.get("runs_folder_name")

    has_minimal_paths = all([workspace, pulsar, gheff, theta_cont, runs_folder])
    if not has_minimal_paths:
        return {
            "input_dir": config.get("pulsar_file_dir"),
            "output_dir": paths.get("output_dir") or config.get("results", {}).get("output_directory"),
            "output_file": None,
            "log_dir": log_dir,
            "shell_script": paths.get("shell_script_path") or os.path.join(script_dir or ".", "phasogram_slurm.sh"),
            "config_file": config_file,
        }

    # Extract cuts information for both directory and filename
    cuts = config.get("cuts", {})
    zd_range = cuts.get("zd_range", [])
    date_range = cuts.get("date_range", [])
    
    # Generate suffix for output directory path (date then zmin/zmax format)
    selection_suffix_parts = []
    if isinstance(date_range, list) and len(date_range) == 2:
        date_start = str(date_range[0]).replace("-", "")
        date_end = str(date_range[1]).replace("-", "")
        selection_suffix_parts.append(f"firstdate{date_start}_lastdate{date_end}")
    if isinstance(zd_range, list) and len(zd_range) == 2:
        selection_suffix_parts.append(f"zmin{zd_range[0]}_zmax{zd_range[1]}")
    
    selection_suffix = "_" + "_".join(selection_suffix_parts) if selection_suffix_parts else ""

    # Default paths if input_rel_dir & output_rel_dir are not defined in the config file 
    input_rel_dir = paths.get("input_rel_dir", "data/processed/DL3/Phased_pulsars")
    output_rel_dir = paths.get("output_rel_dir", "results/preliminary")  # Base output dir (phasograms/ is added later)

    input_root_dir = paths.get("input_root_dir")
    output_root_dir = paths.get("output_root_dir")

    if not input_root_dir:
        input_root_dir = os.path.join(workspace, input_rel_dir)

    if not output_root_dir:
        output_root_dir = os.path.join(workspace, output_rel_dir)

    input_dir = os.path.join(
        input_root_dir,
        pulsar,
        gheff,
        theta_cont,
        runs_folder,
    )

    # Get the fitting model name
    fitting_config = config.get("fitting", {})
    model_name = fitting_config.get("model", "default_model")

    # fit_mode=fitting_config.get("model", "binned")
    # if fit_mode is "True":
    #     fit_mode='binned'
    # elif fit_mode is "False":
    #     fit_mode='unbinned'
    # else: 
    #     raise ValueError(
    #                 "Invalid value for the binned config input"
    #             )
    
    output_dir = os.path.join(
        output_root_dir,
        pulsar,
        gheff,
        theta_cont,
        f"{runs_folder}_postcuts{selection_suffix}",
        "phasograms",
        "models",  # Add phasograms subdirectory
        model_name,    # Add model-specific subdirectory
        #fit_mode,
    )

    # Build output PDF filename with all postcuts info
    output_filename_parts = ["phasograms"]
    
    # Add gheff cut in the filename
    if gheff:
        gheff_value = gheff.replace("_", "")
        output_filename_parts.append(gheff_value)
    
    # Add theta_cont in the filename
    if theta_cont:
        theta_value = theta_cont.replace("_", "")
        output_filename_parts.append(theta_value)
    
    # Add date range in the filename
    if isinstance(date_range, list) and len(date_range) == 2:
        date_start = str(date_range[0]).replace("-", "")
        date_end = str(date_range[1]).replace("-", "")
        output_filename_parts.append(f"date{date_start}to{date_end}")
    
    # Add zenith (zd) range in the filename
    if isinstance(zd_range, list) and len(zd_range) == 2:
        output_filename_parts.append(f"zdmin{int(zd_range[0])}_zdmax{int(zd_range[1])}") # ATTENTION Test 
    
    output_filename = "_".join(output_filename_parts) + ".pdf"
    output_file = os.path.join(output_dir, output_filename)

    default_shell = os.path.join(script_dir or ".", "phasogram_slurm.sh")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "output_file": output_file,
        "log_dir": log_dir,
        "shell_script": paths.get("shell_script_path", default_shell),
        "config_file": config_file,
    }


def output_file_from_dir(output_dir, filename="phasogram.pdf", config=None):
    """
    Generate output file path from directory.
    
    Note: This is a legacy/fallback function. For new code, use build_phasogram_output_dir()
    which consolidates directory and filename generation.
    
    Parameters
    ----------
    output_dir : str
        Output directory path
    filename : str, optional
        Base filename (default: "phasogram.pdf")
    config : dict, optional
        Ignored (kept for backward compatibility)
        
    Returns
    -------
    str
        Full path to output file
    """
    if output_dir is None:
        return None
    
    if output_dir.endswith(".pdf"):
        return output_dir
    
    return os.path.join(output_dir, filename)
