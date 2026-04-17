import os


def generate_selection_suffix(config):
    """Generate a stable suffix from cuts settings (zenith/date)."""
    cuts = config.get("cuts", {})
    if not cuts:
        return ""

    suffix_parts = []

    zd_range = cuts.get("zd_range")
    if isinstance(zd_range, list) and len(zd_range) == 2:
        suffix_parts.append(f"zmin{zd_range[0]}_zmax{zd_range[1]}")

    date_range = cuts.get("date_range")
    if isinstance(date_range, list) and len(date_range) == 2:
        date_start = str(date_range[0]).replace("-", "")
        date_end = str(date_range[1]).replace("-", "")
        suffix_parts.append(f"firstdate{date_start}_lastdate{date_end}")

    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""


def build_phasogram_output_dir(config, config_file=None, script_dir=None):
    """Build phasogram output directory path from the config paths section."""
    paths = config.get("paths", {})

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
            "log_dir": paths.get("log_output_dir", "./out"),
            "shell_script": paths.get("shell_script_path") or os.path.join(script_dir or ".", "phasogram_slurm.sh"),
            "config_file": config_file,
        }

    selection_suffix = generate_selection_suffix(config)

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

    output_dir = os.path.join(
        output_root_dir,
        pulsar,
        gheff,
        theta_cont,
        f"{runs_folder}_postcuts_{selection_suffix}",
        "phasograms",  # Add phasograms subdirectory
    )

    default_shell = os.path.join(script_dir or ".", "phasogram_slurm.sh")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "log_dir": paths.get("log_output_dir", "./out"),
        "shell_script": paths.get("shell_script_path", default_shell),
        "config_file": config_file,
    }


def output_file_from_dir(output_dir, filename="phasogram.pdf"):
    if output_dir is None:
        return None
    if output_dir.endswith(".pdf"):
        return output_dir
    return os.path.join(output_dir, filename)
