import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy import units as u
from dataclasses import dataclass
from .ptime_analysis import PulsarTimeAnalysis
from .phase_regions import PhaseRegions, PulsarPeak
from .lightcurve import Lightcurve
from .periodicity_test import PeriodicityTest
from .pfitting import PeakFitting
from .models import get_model_list
from .phasebinning import PhaseBinning
from .penergy_analysis import PEnergyAnalysis
from .filter_object import FilterPulsarAna
from .read_events import ReadDL3File, ReadFermiFile, ReadLSTFile, ReadList
from .path_phasogram_utils import build_phasogram_output_dir, output_file_from_dir #(LBZ)
import pickle
import yaml
import logging
import os
from datetime import datetime #(LBZ)

#pd.options.mode.chained_assignment = None

#LOG_FORMAT = "%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
#logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

# logging.basicConfig(
#     level=logging.INFO,  # Minimum level of messages to log (INFO, WARNING, ERROR, etc.)
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Format: timestamp - level - message
#     filename='pulsar_analysis.log'  # Log file name
# )

logger = logging.getLogger(__name__)
#logging.getLogger("matplotlib.font_manager").disabled = True
#logging.getLogger("gammapy").disabled = True

__all__ = ["PulsarAnalysis"]


@dataclass
class PulsarAnalysis:
    """
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.

    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    pdata : List of float
        List of phases (in case no dataframe is available).
    ptimes: List of float
        List of mjd times (in case no dataframe is available).
    tobservation: float
        Total effective time of observation
    peak_limits_1: tuple
        Edges of the first peak. Set to None if no P1 is present.
    peak_limits_2: tuple
        Edges of the second peak. Set to None if no P2 is present.
    off_limits: tuple
        Edges of the OFF region
    binned: boolean
        True for a binned fitting, False for an unbinned fitting.


    Attributes
    ----------
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    tobs : float
        Effective time of observation in hours
    regions: PhaseRegions object
        Information of the OFF/signal regions
    histogram: Lightcurve object
        Information of the phaseogram
    stats: PeriodicityTest object
        Information of Statistical Tests for searching Periodicity
    fitting: PeakFitting object
        Information abot the fitting used for the peaks
    """

    def __init__(
        self, filename=None, nbins=50, tint=3600, binned=True, model="dgaussian"
    ):
        if filename is not None:
            if "fits" in filename:
                logger.info("Assuming Fermi-LAT data as an input")
                self.setFermiInputFile(filename)
                self.setTimeInterval(tint=3600 * 24 * 10)  # Every 10 days

            elif "h5" in filename:
                logger.info("Assuming LST1 data as an input")
                self.setLSTInputFile(filename)
                self.setTimeInterval(tint=3600)  # Every hour
            else:
                ValueError("FIle has no valid format")
        else:
            self.setTimeInterval(tint=tint)

        # Define default parameters for the binning
        self.setBinning(nbins=nbins)

        # Define default parameters for the fitting
        self.setFittingParams(model, binned, do_fit=False)

        # Define default parameters for the cuts
        self.setParamCuts()

    ##############################################
    # SETTINGS
    #############################################

    #def setFermiInputFile(self, filename):
    def setFermiInputFile(self, filename, date_cuts=None): #(LBZ)
        if "fits" in filename:
            #self.r = ReadFermiFile(filename)
            self.r = ReadFermiFile(filename, date_cuts=date_cuts) #(LBZ)
            self.telescope = "fermi"
            self.energy_units = "GeV"

        else:
            raise ValueError("No FITS file given for Fermi-LAT data")

    def setListsInput(
        self, plist, tlist=None, elist=None, tel="MAGIC", energy_units="GeV"
    ):
        self.r = ReadList(plist, tlist, elist, tel)
        self.telescope = tel
        self.energy_units = energy_units

    def setDL3InputFile(
        self,
        dirname=None,
        target_radec=None,
        max_rad=0.2,
        zd_cuts=[0, 60],
        date_cuts=None, #(LBZ)
        energy_dependent_theta=True,
    ):
        self.r = ReadDL3File(
            directory=dirname,
            target_radec=target_radec,
            max_rad=max_rad,
            zd_cuts=zd_cuts,
            date_cuts=date_cuts, #(LBZ)
            energy_dependent_theta=energy_dependent_theta,
        )
        self.telescope = "lst"
        self.energy_units = "TeV"
    # def setLSTInputFile(self, filename=None, dirname=None, src_dep=False):
    #     self.r = ReadLSTFile(file=filename, directory=dirname, src_dependent=src_dep)
    def setLSTInputFile(self, filename=None, dirname=None, src_dep=False, date_cuts=None): #(LBZ)
        self.r = ReadLSTFile(file=filename, directory=dirname, src_dependent=src_dep, date_cuts=date_cuts) #(LBZ)
        self.telescope = "lst"
        self.energy_units = "TeV"

    def setBinning(self, nbins, xmin=None, xmax=None):
        self.nbins = nbins
        self.binning = PhaseBinning(nbins, xmin, xmax)

    def setParamCuts(
        self,
        gammaness_cut=None,
        alpha_cut=None,
        theta2_cut=None,
        zd_cut=None,
        int_cut=None,
        energy_cut=None,
        energy_binning_cut=None,
        date_cut=None, #(LBZ)
    ):
        self.cuts = FilterPulsarAna(
            gammaness_cut,
            alpha_cut,
            theta2_cut,
            zd_cut,
            int_cut,
            energy_cut,
            energy_binning_cut,
            date_cut, #(LBZ)
        )

    def setEnergybinning(self, energy_edges, do_diff, do_integral):
        self.energy_edges = energy_edges
        self.EnergyAna = PEnergyAnalysis(self.energy_edges, do_diff, do_integral)

    def setFittingParams(self, model, binned=False, peak="both", do_fit=True):
        model_list = get_model_list()

        if model in model_list:
            self.fit_model = model
            self.peak = peak
            self.binned = binned
            self.do_fit = do_fit
        else:
            raise ValueError("Model given is not defined")

        self.fitting = PeakFitting(self.binned, self.fit_model, peak=self.peak)
        self.fitting.check_model()

    def setTimeInterval(self, tint):
        self.tint = tint
        self.TimeEv = PulsarTimeAnalysis(tint=self.tint)

    def setBackgroundLimits(self, OFF_limits):
        self.OFF_limits = OFF_limits

    def setPeaklimits(self, P1_limits=None, P2_limits=None, P3_limits=None):
        if P1_limits == "None":
            P1_limits = None
        if P2_limits == "None":
            P2_limits = None
        if P3_limits == "None":
            P3_limits = None

        P1P2_limits = []

        if P1_limits is not None:
            P1P2_limits += P1_limits

        if P2_limits is not None:
            P1P2_limits += P2_limits

        self.P1_limits = P1_limits
        self.P2_limits = P2_limits
        self.P1P2_limits = P1P2_limits
        self.P3_limits = P3_limits

    def set_config(self, configuration_file):
        # Read the configuration file
        with open(configuration_file, "rb") as cfile:
            conf = yaml.safe_load(cfile)

########################################################################### (LBZ)
        built_paths = build_phasogram_output_dir(
            conf,
            config_file=configuration_file,
            script_dir=os.path.dirname(os.path.abspath(__file__)),
        )
        
        # Save built_paths as instance attribute for later use
        self.built_paths = built_paths  # (LBZ)

        # Prefer canonical path-building from the paths section when available.
        # This removes hardcoded per-user absolute paths from the runtime flow.
        if built_paths.get("input_dir"):
            conf["pulsar_file_dir"] = built_paths["input_dir"]

########################################################################### (LBZ)

############################################################################## (LBZ)
        # Convert date_range from ISO strings to Unix timestamps
        date_cut = None
        if "date_range" in conf["cuts"] and conf["cuts"]["date_range"] is not None:
            date_range = conf["cuts"]["date_range"]
            if isinstance(date_range, list) and len(date_range) == 2:
                try:
                    t_min = datetime.fromisoformat(date_range[0]).timestamp()
                    t_max = datetime.fromisoformat(date_range[1]).timestamp()
                    date_cut = [t_min, t_max]
                    logger.info(f"Date range cut applied: {date_range[0]} to {date_range[1]}")
                    #print(f"Date range cut applied: {date_range[0]} to {date_range[1]}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse date_range: {e}")
                    #print(f"Could not parse date_range: {e}")
############################################################################## (LBZ)


        # Read files
        self.filter_data = conf["cuts"]["filter_data"]
        if conf["flags"]["DL2_format"]:
            if os.path.isdir(conf["pulsar_file_dir"]):
                self.setLSTInputFile(
                    dirname=conf["pulsar_file_dir"],
                    src_dep=conf["flags"]["src_dependent"],
                    date_cuts=date_cut, # (LBZ)
                )
            else:
                self.setLSTInputFile(
                    filename=conf["pulsar_file_dir"],
                    src_dep=conf["flags"]["src_dependent"],
                    date_cuts=date_cut,  #(LBZ)
                )

            if conf["cuts"]["include_DL2_extra_cuts"]:
                if conf["cuts"]["extra_cuts"]["energy_dependent"]:
                    if conf["flags"]["src_dependent"]:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            alpha_cut=conf["cuts"]["extra_cuts"]["alpha"],
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_binning_cut=conf["cuts"]["extra_cuts"][
                                "energy_binning"
                            ],
                            date_cut=date_cut, #(LBZ)
                        )
                    else:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            theta_cut=np.power(conf["cuts"]["extra_cuts"]["theta"], 2),
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_binning_cut=conf["cuts"]["extra_cuts"][
                                "energy_binning"
                            ],
                            date_cut=date_cut, #(LBZ)
                        )
                else:
                    if conf["flags"]["src_dependent"]:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            alpha_cut=conf["cuts"]["extra_cuts"]["alpha"],
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_cut=conf["cuts"]["extra_cuts"]["energy"],
                            date_cut=date_cut,  # (LBZ)
                        )
                    else:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            theta_cut=np.power(conf["cuts"]["extra_cuts"]["theta"], 2),
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_cut=conf["cuts"]["extra_cuts"]["energy"],
                            date_cut=date_cut, #(LBZ)
                        )

            else:
                self.setParamCuts(zd_cut=conf["cuts"]["zd_range"], date_cut=date_cut) # (LBZ)

        elif conf["flags"]["fits_format"]:
            #self.setFermiInputFile(filename=conf["pulsar_file_dir"])
            self.setFermiInputFile(filename=conf["pulsar_file_dir"], date_cuts=date_cut) #(LBZ)

        else:
            self.is_DL3_input = True
            self.setDL3InputFile(
                dirname=conf["pulsar_file_dir"],
                target_radec=[conf["target"]["ra"], conf["target"]["dec"]],
                max_rad=conf["cuts"]["max_rad"],
                zd_cuts=conf["cuts"]["zd_range"],
                date_cuts=date_cut, #(LBZ)
                energy_dependent_theta=conf["cuts"]["energy_dependent_theta"],
            )
            self.setParamCuts(energy_cut=conf["cuts"]["extra_cuts"]["energy"], date_cut=date_cut) #(LBZ)

        # Set regions
        self.setBackgroundLimits(conf["phase_regions"]["Bkg"])
        self.setPeaklimits(
            P1_limits=conf["phase_regions"]["P1"],
            P2_limits=conf["phase_regions"]["P2"],
            P3_limits=conf["phase_regions"]["P3"], # why not P1+P2 here ??? 
        )

        if not conf["phase_binning"]["custom_binning"]:
            self.setBinning(
                conf["phase_binning"]["nbins"],
                xmin=conf["phase_binning"]["xmin"],
                xmax=conf["phase_binning"]["xmax"],
            )
        else:
            self.setBinning(conf["phase_binning"]["binning"])

        if conf["time_binning"]["run_time_analysis"]:
            units_time = conf["time_binning"]["units"]
            tint = conf["time_binning"]["tint"] * u.Unit(units_time)

            self.setTimeInterval(tint.to(u.s).value)
        else:
            self.setTimeInterval(3600 * 300)

        if conf["energy_binning"]["run_energy_analysis"]:
            units_energy = conf["energy_binning"]["units"]
            nbins_energy = conf["energy_binning"]["nbins"]
            emin = conf["energy_binning"]["emin"] * u.Unit(units_energy)
            emax = conf["energy_binning"]["emax"] * u.Unit(units_energy)
            do_integral = conf["energy_binning"]["do_integral"]
            do_diff = conf["energy_binning"]["do_diff"]

            self.setEnergybinning(
                np.geomspace(emin.to(u.TeV).value, emax.to(u.TeV).value, nbins_energy),
                do_diff,
                do_integral,
            )

        if conf["fitting"]["run_fitting"]:
            if self.P1_limits is None:
                self.setFittingParams(
                    model=conf["fitting"]["model"],
                    binned=conf["fitting"]["binned"],
                    peak="P2",
                )
            elif self.P2_limits is None:
                self.setFittingParams(
                    model=conf["fitting"]["model"],
                    binned=conf["fitting"]["binned"],
                    peak="P1", # why not P1+P2 ???? 
                )
            else:
                self.setFittingParams(
                    model=conf["fitting"]["model"], binned=conf["fitting"]["binned"]
                )

        # Set output file for results
        self.get_results = conf["results"]["save_results"]
        if self.get_results:
            ################################################################################## (LBZ)
            # Single source: output file path derived from paths + cuts in build_phasogram_output_dir.
            # If missing, fallback to legacy keys for backward compatibility.
            if built_paths.get("output_file"):
                self.output_file = built_paths["output_file"]
            elif "output_file" in conf["results"]:
                self.output_file = conf["results"]["output_file"]
            elif "output_directory" in conf["results"]:
                self.output_file = output_file_from_dir(conf["results"]["output_directory"])
            else:
                raise ValueError(
                    "Missing output path. Define paths.* in config or legacy results.output_file"
                )
        ################################################################################## (LBZ)

            #self.output_dir = os.path.dirname(self.output_file)
            self.output_dir = os.path.dirname(self.output_file)
            if not os.path.exists(self.output_dir):
                logger.info("Creating directory: " + self.output_dir)
                os.makedirs(self.output_dir)
        
        # Calculate phasograms directory (parent of model-specific output_dir) (LBZ) #ISSUE ? 
        self.phasograms_dir = os.path.dirname(os.path.dirname(self.output_dir)) # (LBZ)
        
        # Set raw data output options (LBZ)
        self.save_raw_data = conf["results"].get("save_raw_data", False) #(LBZ)
        self.raw_data_format = conf["results"].get("raw_data_format", "h5") #(LBZ)
        if self.save_raw_data: #(LBZ)
            # Determine raw data filename in phasograms folder (shared across models) #(LBZ)
            if self.get_results: #(LBZ)
                extension = ".h5" if self.raw_data_format == "h5" else ".csv" #(LBZ)
                basename = os.path.basename(os.path.splitext(self.output_file)[0])  # Get base name without .pdf (LBZ)
                self.raw_data_file = os.path.join(self.phasograms_dir, f"raw_data_{basename}{extension}") #(LBZ)
                logger.info(f"Raw data will be saved to: {self.raw_data_file}") #(LBZ)
        else:
            self.output_file = None

    ##############################################
    # EXECUTION
    #############################################

    def shift_phases(self, xmin):
        for i in range(0, len(self.phases)):
            if self.phases[i] < xmin:
                self.phases[i] += 1

        self.info["pulsar_phase"] = self.phases

    def init_regions(self):
        OFFob = PulsarPeak(peak_limits=self.OFF_limits, peaktype="background")

        if self.P1_limits is not None:
            p1ob = PulsarPeak(peak_limits=self.P1_limits, peaktype="signal")
        else:
            p1ob = None

        if self.P2_limits is not None:
            p2ob = PulsarPeak(peak_limits=self.P2_limits, peaktype="signal")
        else:
            p2ob = None

        if self.P3_limits is not None:
            p3ob = PulsarPeak(peak_limits=self.P3_limits, peaktype="signal")
        else:
            p3ob = None

        if len(self.P1P2_limits) > 0:
            p1p2ob = PulsarPeak(peak_limits=self.P1P2_limits, peaktype="signal")

        self.regions = PhaseRegions(
            OFF_object=OFFob,
            P1_object=p1ob,
            P2_object=p2ob,
            P1P2_object=p1p2ob,
            P3_object=p3ob,
        )

    def update_info(self):
        # Fill the background regions
        self.regions.OFF.fillPeak(self.phases)

        # Fill and calculate statistics of Peaks
        for i in self.regions.dic:
            if self.regions.dic[i] is not None:
                self.regions.dic[i].fillPeak(self.phases)
                self.regions.dic[i].make_stats(self.regions, self.tobs)

        # Create the phaseogram using the Lightcurve class
        self.histogram = Lightcurve(self, self.binning)

        # Apply Periodicity stats and store them using the PeriodicityTest Class
        self.stats = PeriodicityTest(self)

    def initialize(self):
        # Read the data and filter
        try:
            self.r.run(self)
        except TypeError:
            self.r.run()

        # Extract each attribute
        self.phases = np.array(self.r.info["pulsar_phase"].to_list())
        self.info = self.r.info

        # Shift phases if necessary
        self.shift_phases(xmin=self.binning.xmin)

        # Initialize the regions object
        self.init_regions()

    def setup_regions_and_binning(self):  # (LBZ) # ISSUE ? 
        """Setup regions and binning without reading data.
        
        Used when loading from cache to avoid re-reading/filtering data.
        Phases must already be loaded into self.phases and region limits set.
        Recreates all dependent objects like histogram and stats.
        """
        if not hasattr(self, 'phases') or len(self.phases) == 0:  # (LBZ)
            raise ValueError("Phases not loaded. Load cache first or run initialize().")  # (LBZ)
        
        # Create self.info if it doesn't exist (it should be created by load_cache) (LBZ)
        if not hasattr(self, 'info') or self.info is None:  # (LBZ)
            logger.warning("self.info not found, creating minimal DataFrame for compatibility")  # (LBZ)
            self.info = pd.DataFrame({'pulsar_phase': self.phases})  # (LBZ)
        
        # Verify all region limits are set (should be restored from cache)
        if not hasattr(self, 'OFF_limits'):
            logger.warning("OFF_limits not set, cannot initialize regions properly")
        
        # Shift phases if necessary
        self.shift_phases(xmin=self.binning.xmin)
        
        # Initialize the regions object
        self.init_regions()  # (LBZ)
        
        # CRITICAL: Fill peaks and calculate their statistics (creates 'sign' attribute) (LBZ)
        self.update_info()  # (LBZ) 20/05

    def execute_stats(self, tobs):
        # Update the information at a certain interval of time and store final values
        self.TimeEv.run(self)
        print('check_run')

        # COmpute P1/P2 ratio
        self.regions.calculate_P1P2()
        print('check_calculate_P1P2')

        # Set the final effective time of observation
        self.tobs = tobs
        print('tobs_ok')

        # Fit the histogram using PeakFitting class. If binned is False, an Unbinned Likelihood method is used for the fitting
        if self.do_fit:
            print('enter_loop')
            logger.info("Fitting the data to the given model...")
            logger.info("Fit model: " + self.fit_model)
            logger.info("Binned fitting: " + str(self.binned))
            self.fitting.run(self)

        else:
            logger.info("No fit has been done since no fit parameters has been set")

    def run(self):
        # Initializa
        logger.info("Initializing...")
        self.initialize()

        # Excute stats
        logger.info(
            "Calculating statistics every " + str(self.tint / 60) + " minutes..."
        )
        self.execute_stats(self.r.tobs)

        # Execute stats in energy bins
        print('self.EnergyAna',self.EnergyAna)
        try:
            logger.info("Performing energy-dependent analysis...")
            self.EnergyAna.run(self)
        except AttributeError:
            logger.warning(
                "No Energy Analysis was performed. Check that you set the right energy params"
            )

        logger.info("FINISHED. Producing general results...")

        if self.get_results:
            self.save_results()

    ##############################################
    # RESULTS
    #############################################

    def check_energyana(self):
        try:
            self.EnergyAna
        except AttributeError:
            logger.info("No energy-dependent analysis has been done")
            return False
        return True

    def draw_phaseogram(
        self,
        phase_limits=[0, 2],
        stats="short",
        background=True,
        signal=["P1", "P2", "P3"],
        colorhist="xkcd:sky blue",
        colorb="black",
        colorP=["orange", "green", "purple"],
        colorfit="red",
        fit=False,
        hline=True,
    ):
        # Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
        fig = plt.figure(figsize=(15, 5))
        self.histogram.show_phaseogram(
            self,
            phase_limits,
            stats,
            background,
            signal,
            colorhist,
            colorb,
            colorP,
            colorfit,
            fit,
            hline,
        )
        return fig

    def show_Presults(self):
        rpeaks = self.regions.show_peak_results()
        rstats = self.stats.show_Pstats()

        print("RESULTS FOR THE PEAK STATISTICS:" + "\n")
        print(rpeaks)
        if self.regions.P1P2_ratio is not None:  #(TEST) Simplified check since P1P2_ratio always exists now
            print(
                "\n"
                + f"P1/P2 ratio={self.regions.P1P2_ratio:.2f}"
                + f"+/-{self.regions.P1P2_ratio_error:.2f}"
                + "\n"
            )
        print("\n \n" + "RESULTS FOR THE PERIODICITY SEARCH:" + "\n")
        print(rstats)

        return rpeaks, rstats

    def show_fit_results(self):
        fresult = self.fitting.show_result()
        print(fresult)
        return fresult

    def show_timeEvolution(self):
        fig1 = self.TimeEv.show_results()
        fig2 = self.TimeEv.compare_Peaksig()
        return fig1, fig2

    def show_EnergyAna(self, integral=None):
        if self.check_energyana():
            fig = plt.figure(figsize=(20, 4))
            if self.regions.P1 is None or self.regions.P2 is None:
                nplots = 2
            else:
                nplots = 3
                plt.subplot(1, nplots, 3)
                self.EnergyAna.P1P2VsEnergy(integral)

            plt.subplot(1, nplots, 1)
            self.EnergyAna.PSigVsEnergy(integral)

            plt.subplot(1, nplots, 2)
            self.EnergyAna.FWHMVsEnergy(integral)

            plt.tight_layout()
            return fig

    def show_EnergyPresults(self, integral=None):
        if self.check_energyana():
            # show_EnergyPresults now returns a list of (peak_stat, p_stat) tuples for all bins (LBZ)
            return self.EnergyAna.show_EnergyPresults(integral)
        return []  # (LBZ) Return empty list if no energy analysis

    def show_EnergyFitresults(self, integral=None):
        if self.check_energyana():
            fit_results = self.EnergyAna.show_Energy_fitresults(integral)
            return fit_results

    def show_meanVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = self.EnergyAna.MeanVsEnergy(integral)
            return fig

    # def show_WidthVsEnergy(self, integral=None): # (LBZ)
    #     if self.check_energyana(): # (LBZ)
    #         fig = self.EnergyAna.WidthVsEnergy(integral) # (LBZ)
    #         return fig # (LBZ)

    def show_FWHMVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.FWHMVsEnergy()
            return fig

    def show_SigVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.PSigVsEnergy(integral)
            return fig

    def show_P1P2VsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.P1P2VsEnergy(integral)
            return fig

    def show_Energy_TimeResults(self, integral=None):  #(LBZ) Same pattern as show_EnergyAna()
        """Get temporal analysis plots for each energy bin (returns flat list of figures)"""  #(LBZ)
        if self.check_energyana():  #(LBZ)
            return self.EnergyAna.get_Energy_TimeResults(integral)  #(LBZ)
        return []  #(LBZ)

    def show_lcVsEnergy(self, integral=None):
        if self.check_energyana():
            fig_array = self.EnergyAna.show_Energy_lightcurve(integral)
            return fig_array

    def show_all_lc(self, ylimits=None):
        if self.check_energyana():
            fig = self.EnergyAna.show_joined_Energy_lightcurve(ylimits=ylimits)
            return fig

    def show_all_fits(self, integral=None):
        if self.check_energyana():
            fig = self.EnergyAna.show_joined_Energy_fits(integral)
            return fig

    ####################################################### (LBZ)
    def save_cache(self, cache_file):  # (LBZ)
        """Save filtered pulsar data to cache for fast re-fitting.
        
        Caches phases, times, energies, and configuration to avoid 
        re-reading/filtering data during subsequent fits.
        
        Parameters:
            cache_file (str): Path to cache file
        """
        try:
            cache_data = {
                'phases': self.phases,
                'times': self.times,
                'energies': getattr(self, 'energies', None),  # May not exist
                'tobs': self.tobs,
                'telescope': self.telescope,
                'energy_units': self.energy_units,
                'nbins': self.nbins,
                'tint': self.tint,
                # Save all configuration/limits needed for regions
                'OFF_limits': getattr(self, 'OFF_limits', None),
                'P1_limits': getattr(self, 'P1_limits', None),
                'P2_limits': getattr(self, 'P2_limits', None),
                'P3_limits': getattr(self, 'P3_limits', None),
                'P1P2_limits': getattr(self, 'P1P2_limits', None),
            }
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Pulsar data cached to: {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            return False
    
    def load_cache(self, cache_file):  # (LBZ)
        """Load filtered pulsar data from cache.
        
        Restores phases, times, energies from cache, skipping 
        expensive read/filter operations.
        
        Parameters:
            cache_file (str): Path to cache file
            
        Returns:
            bool: True if cache loaded successfully, False otherwise
        """
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore all cached data
            self.phases = cache_data['phases']
            self.times = cache_data['times']
            self.energies = cache_data['energies']
            self.tobs = cache_data['tobs']
            self.telescope = cache_data['telescope']
            self.energy_units = cache_data['energy_units']
            self.nbins = cache_data['nbins']
            self.tint = cache_data['tint']
            
            # Restore region limits
            self.OFF_limits = cache_data.get('OFF_limits', None)
            self.P1_limits = cache_data.get('P1_limits', None)
            self.P2_limits = cache_data.get('P2_limits', None)
            self.P3_limits = cache_data.get('P3_limits', None)
            self.P1P2_limits = cache_data.get('P1P2_limits', None)
            
            # Create self.info DataFrame with both phase and energy columns for compatibility
            # Energy is needed by penergy_analysis for binning (LBZ)
            self.info = pd.DataFrame({
                'pulsar_phase': self.phases,
                'energy': self.energies if self.energies is not None else np.zeros(len(self.phases))
            })
            
            logger.info(f"Pulsar data loaded from cache: {cache_file}")
            logger.info(f"  - {len(self.phases)} events")
            logger.info(f"  - {self.tobs:.2f}s observation time")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    ####################################################### (LBZ)

    def save_df(self, output_file, file_format="h5"):
        """Save the phasogram data to file.
        
        Parameters:
            output_file (str): Full path to output file
            file_format (str): "h5" for HDF5 binary or "csv" for CSV text format
        """
        if file_format == "h5":
            self.info.to_hdf(
                output_file, key="dl2/event/telescope/parameters/LST_LSTCam"
            )
            logger.info(f"Data saved to HDF5: {output_file}") #(LBZ)

        elif file_format == "csv":
            self.info.to_csv(output_file, index=False)  #(LBZ)
            logger.info(f"Data saved to CSV: {output_file}") #(LBZ)

    def save_results(self, output_file=None):
        if output_file is not None:
            self.output_file = output_file
            self.output_dir = os.path.dirname(self.output_file)

        with PdfPages(self.output_file) as pdf:
            pdf.savefig(
                self.draw_phaseogram(
                    phase_limits=[0, 2],
                    stats="long",
                    background=True,
                    signal=["P1", "P2", "P3"],
                    colorhist="blue",
                    colorb="black",
                    colorP=["orange", "green", "purple"],
                    colorfit="red",
                    fit=False,
                    hline=True,
                ),
                bbox_inches="tight",
                pad_inches=1,
            )
            pdf.savefig(self.TimeEv.PsigVsTime())
            pdf.savefig(self.TimeEv.PexVsTime())
            pdf.savefig(self.TimeEv.StatsVsTime())

            if self.do_fit:
                pdf.savefig(
                    self.draw_phaseogram(
                        phase_limits=[0, 2],
                        stats="long",
                        background=True,
                        signal=["P1", "P2", "P3"],
                        colorhist="blue",
                        colorb="black",
                        colorP=["orange", "green", "purple"],
                        colorfit="red",
                        fit=True,
                        hline=True,
                    ),
                    bbox_inches="tight",
                    pad_inches=1,
                )

                fitting = self.fitting.show_result()
                # Create directory for fitting results by energy bin (LBZ) # ISSUE ? 
                fitting_energy_dir = os.path.join(self.output_dir, "fitting_energy_bin") #(LBZ)
                if not os.path.exists(fitting_energy_dir): #(LBZ)
                    os.makedirs(fitting_energy_dir) #(LBZ)
                    logging.info(f"Created fitting_energy_bin directory: {fitting_energy_dir}") #(LBZ)
                fitting.to_hdf(os.path.join(fitting_energy_dir, "overall_fitting.h5"), key="results") #(LBZ)
                logging.info(f"Saved overall fitting results to {os.path.join(fitting_energy_dir, 'overall_fitting.h5')}") #(LBZ)
                # fig_width = self.show_WidthVsEnergy()  # (LBZ)
                # if fig_width is not None:
                #     pdf.savefig(fig_width, bbox_inches="tight", pad_inches=1)

            try:
                pdf.savefig(self.show_EnergyAna(), bbox_inches="tight", pad_inches=1)
                pdf.savefig(self.show_meanVsEnergy(), bbox_inches="tight", pad_inches=1)  #(LBZ) Add Mean vs Energy plot                
                #pdf.savefig(self.show_WidthVsEnergy(), bbox_inches="tight", pad_inches=1)  #(LBZ) Add Width vs Energy plot                
                for i in range(0, len(self.EnergyAna.show_Energy_lightcurve())):
                    pdf.savefig(
                        self.EnergyAna.show_Energy_lightcurve()[i],
                        bbox_inches="tight",
                        pad_inches=1,
                    )
                #(LBZ) Save temporal results for each energy bin to PDF (same integral setting as lightcurves)
                time_figs = self.show_Energy_TimeResults()  #(LBZ) NO integral arg - use default (self.integral)
                for i in range(0, len(time_figs)):  #(LBZ) Iterate cached results
                    pdf.savefig(time_figs[i], bbox_inches="tight", pad_inches=1)  #(LBZ)
                if self.do_fit:
                    fit_results = self.show_EnergyFitresults() #(LBZ)
                    print('fit_results',fit_results)
                    print('len_fit_results',len(fit_results))
                    # Use fitting_energy_dir created above (LBZ)
                    for idx, df in enumerate(fit_results): #(LBZ)
                    #for idx, df in range(len(fit_results) + 1):
                        if df is not None: #(LBZ)
                            original_bin_idx = self.EnergyAna.Parray[-1]._energy_bin_index #(LBZ)
                            fit_file = os.path.join(fitting_energy_dir, f"fitting_energy_bin{original_bin_idx}.h5") #(LBZ)
                            df.to_hdf(fit_file, key="results") #(LBZ)
                            logging.info(f"Saved energy bin {original_bin_idx} fitting results to {fit_file}") #(LBZ)
            except AttributeError:
                pass
        
        # Save raw data if requested (LBZ) #ISSUE ? 
        if self.save_raw_data: #(LBZ)
            try: #(LBZ)
                logger.info(f"Saving raw phasogram data to {self.raw_data_file} (format: {self.raw_data_format})") #(LBZ)
                self.save_df(self.raw_data_file, file_format=self.raw_data_format) #(LBZ)
                logger.info("Raw data saved successfully") #(LBZ)
                
                # Save data by energy bin if energy analysis was performed (LBZ)
                if self.check_energyana(): #(LBZ)
                    logger.info("Saving raw data by energy bin...") #(LBZ)
                    self.save_raw_data_by_energy_bin() #(LBZ)
                    logger.info("Raw data by energy bin saved successfully") #(LBZ)
                    
            except Exception as e: #(LBZ)
                logger.error(f"Failed to save raw data: {e}") #(LBZ)

    def save_raw_data_by_energy_bin(self): #ISSUE ? 
        """Save filtered event data for each energy bin separately. (LBZ)"""
        # Create subdirectory for energy bin data in phasograms folder (shared across models) (LBZ)
        energy_bin_dir = os.path.join(
            self.phasograms_dir, "raw_data_by_energy_bin"
        ) #(LBZ)
        if not os.path.exists(energy_bin_dir): #(LBZ)
            os.makedirs(energy_bin_dir) #(LBZ)
            logger.info(f"Created energy bin directory: {energy_bin_dir}") #(LBZ)
        
        # Get energy edges from analysis (LBZ)
        energy_edges = self.EnergyAna.energy_edges #(LBZ)
        
        # For each energy bin, save data (LBZ)
        for i in range(len(energy_edges) - 1): #(LBZ)
            emin = energy_edges[i] #(LBZ)
            emax = energy_edges[i + 1] #(LBZ)
            
            # Filter data for this energy bin (LBZ)
            energy_mask = (self.info["energy"] >= emin) & (self.info["energy"] < emax) #(LBZ)
            energy_bin_data = self.info[energy_mask] #(LBZ)
            
            if len(energy_bin_data) == 0: #(LBZ)
                logger.warning(f"No data in energy bin {i}: {emin:.2f}-{emax:.2f} TeV") #(LBZ)
                continue #(LBZ)
            
            # Build filename for this bin (LBZ)
            extension = ".h5" if self.raw_data_format == "h5" else ".csv" #(LBZ)
            bin_filename = f"energy_bin_{i:02d}_{emin:.2f}-{emax:.2f}TeV{extension}" #(LBZ)
            bin_filepath = os.path.join(energy_bin_dir, bin_filename) #(LBZ)
            
            # Save data for this bin (LBZ)
            try: #(LBZ)
                if self.raw_data_format == "h5": #(LBZ)
                    energy_bin_data.to_hdf( #(LBZ)
                        bin_filepath, key="dl2/event/telescope/parameters/LST_LSTCam" #(LBZ)
                    ) #(LBZ)
                elif self.raw_data_format == "csv": #(LBZ)
                    energy_bin_data.to_csv(bin_filepath, index=False) #(LBZ)
                
                logger.info( #(LBZ)
                    f"Saved {len(energy_bin_data):,} events for energy bin {i} " #(LBZ)
                    f"({emin:.2f}-{emax:.2f} TeV) to {bin_filename}" #(LBZ)
                ) #(LBZ)
            except Exception as e: #(LBZ)
                logger.error(f"Failed to save energy bin {i} data: {e}") #(LBZ)

    def save_object(self, output_file):
        with open(output_file, "wb") as file:
            pickle.dump(self, file)
            file.close()
