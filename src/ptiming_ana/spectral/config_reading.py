import astropy.units as u
import os
import yaml
import logging
from gammapy.maps import MapAxis
from .path_spectra_utils import build_spectra_output_dir, _generate_selection_suffix


LOG_FORMAT = "%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True

__all__ = ["SpectralConfigSetting"]

class SpectralConfigSetting:
    def __init__(self, config):
        if config is not None:
            if ".yaml" in config:
                self.read_configuration(config)

    def read_configuration(self, configuration_file):
        with open(configuration_file, "rb") as cfile:
            self.conf = yaml.safe_load(cfile)

    def set_general_from_config(self):
        logger.info("Reading general settings from configuration file")

        configuration_dict = self.conf

        ################################################################## (LBZ)
        # Check if using new 'paths' section configuration
        if "paths" in configuration_dict and configuration_dict["paths"] is not None:
            logger.info("Using new paths-based configuration structure")
            # Use new path building logic
            self.output_dir = build_spectra_output_dir(configuration_dict)
            
            # Build input directory similarly
            paths = configuration_dict.get("paths", {})
            workspace = paths.get("workspace_root")
            input_rel_dir = paths.get("input_rel_dir")
            input_root_dir = paths.get("input_root_dir")
            output_rel_dir = paths.get("output_rel_dir", "results/preliminary")
            output_root_dir = paths.get("output_root_dir")
            
            if input_root_dir:
                input_base = input_root_dir
            elif workspace and input_rel_dir:
                input_base = os.path.join(workspace, input_rel_dir)
            else:
                input_base = configuration_dict.get("pulsar_file_dir")
            
            if not output_root_dir:
                if workspace:
                    output_root_dir = os.path.join(workspace, output_rel_dir)
            
            pulsar = paths.get("pulsar_name", "Crab")
            gheff = paths.get("gheff_cut", "gheffcut_0.9")
            theta_cont = paths.get("theta_cont", "thetacont_0.7")
            runs_folder = paths.get("runs_folder_name", "")
            
            # Generate selection suffix from cuts
            selection_suffix = _generate_selection_suffix(configuration_dict)
            
            self.directory = os.path.join(input_base, pulsar, gheff, theta_cont, runs_folder)
            logger.info(f"Using input directory: {self.directory}")
            
            # Build DL4 directory in the same location as phasograms/spectra
            # Structure: results/preliminary/Crab/{gheff}/{theta}/{runs_folder}_postcuts_{cuts}/DL4/
            # This directory will be used by execute_makers() to save DL4 output files
            dl4_base = os.path.join(
                output_root_dir,
                pulsar,
                gheff,
                theta_cont,
                f"{runs_folder}_postcuts{selection_suffix}",
                "DL4"
            )
            self.dl4_dir = dl4_base
            
            # Create DL4 directory with parametrized path structure
            if not os.path.exists(self.dl4_dir):
                logger.info(f"Creating DL4 directory: {self.dl4_dir}")
                os.makedirs(self.dl4_dir)
            
        else:
            logger.info("Using legacy configuration structure")
            # Fallback to legacy configuration
            self.directory = configuration_dict.get("pulsar_file_dir")
            self.output_dir = configuration_dict.get("results_output_directory")
            # Use .get() to safely handle None values in config
            self.dl4_dir = configuration_dict.get("DL4_directory")
            
            if self.output_dir and not os.path.exists(self.output_dir):
                logger.info("Creating directory: " + self.output_dir)
                os.makedirs(self.output_dir)
            
            # Only create DL4 directory if it's defined and not None
            if self.dl4_dir and not os.path.exists(self.dl4_dir):
                logger.info("Creating directory: " + self.dl4_dir)
                os.makedirs(self.dl4_dir)

        # Energy dependent theta
        self.reader_info = configuration_dict.get("reader", {}) # (LBZ)

        # Target info
        self.target_info = configuration_dict.get("target", {}) # (LBZ)

        # Regions
        self.phase_region_dic = configuration_dict.get("phase_regions", {}) # (LBZ)

        # Geometry
        self.energy_geometry = configuration_dict.get("energy_geometry", {}) # (LBZ)

        # Extra settings
        self.extra_settings = configuration_dict.get("analysis_extra_settings", {}) # (LBZ)
        ######################################################################### # (LBZ)

    def set_spectral_fitting_from_config(self):
        logger.info("Reading fitting parameters from configuration file")
        configuration_dict = self.conf

        spectral_fitting = configuration_dict["spectral_fitting"]
        self.e_min_fitting = spectral_fitting["emin"] * u.Unit(
            spectral_fitting["units"]
        )
        self.e_max_fitting = spectral_fitting["emax"] * u.Unit(
            spectral_fitting["units"]
        )

        self.model = spectral_fitting["model"]

    def set_spectral_points_from_config(self):
        logger.info("Reading spectral points parameters from configuration file")

        configuration_dict = self.conf

        spectral_points = configuration_dict["spectral_points"]
        self.e_min_points = spectral_points["emin"] * u.Unit(spectral_points["units"])
        self.e_max_points = spectral_points["emax"] * u.Unit(spectral_points["units"])

        self.bins_per_decade = spectral_points["bins_per_decade"]
        self.npoints = spectral_points["number_points"]
        self.min_ts = spectral_points["min_ts"]

    def set_all(self):
        self.set_general_from_config()
        self.set_spectral_fitting_from_config()
        self.set_spectral_points_from_config()

    def extract_energy_geometry(self):
        true_energies = self.energy_geometry["real"]
        reco_energies = self.energy_geometry["reco"]

        true_energy_axis = MapAxis.from_energy_bounds(
            true_energies["emin"],
            true_energies["emax"],
            true_energies["nbinning"],
            unit=true_energies["units"],
            name="energy_true",
        )
        reco_energy_axis = MapAxis.from_energy_bounds(
            reco_energies["emin"],
            reco_energies["emax"],
            reco_energies["nbinning"],
            unit=reco_energies["units"],
            name="energy",
        )

        return (true_energy_axis, reco_energy_axis)

    def extract_detailed_reading_info(self):

        ##################################################################### (LBZ)
        # Read zenith angle range and date range from 'cuts' section (shared with phasogram)
        # to ensure both analyses use the same cuts for synchronized filtering
        cuts = self.conf.get("cuts", {})
        zd_range = cuts.get("zd_range", [0, 50])  # Fallback to [0, 50] if not defined
        
        # Read date range from cuts and convert to Unix timestamps
        date_range = cuts.get("date_range", None)
        date_cuts = None
        if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            from astropy.time import Time
            # Convert ISO format dates to Unix timestamps for filtering
            try:
                date_start = Time(date_range[0], scale='utc').unix
                date_end = Time(date_range[1], scale='utc').unix
                date_cuts = [date_start, date_end]
                logger.info(f"Date range cuts applied: {date_range[0]} to {date_range[1]}")
            except Exception as e:
                logger.warning(f"Could not parse date_range: {e}. Proceeding without date cuts.")
        
        ###################################################################### (LBZ)
        edependent_theta = self.reader_info["energy_dependent_theta"]

        if not edependent_theta:
            max_rad = self.reader_info["max_rad"]
        else:
            max_rad = None

        return (edependent_theta, max_rad, zd_range, date_cuts)
