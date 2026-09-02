import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
from iminuit.cost import LeastSquares
from iminuit import Minuit
#from .ptime_analysis import PulsarTimeAnalysis  #(LBZ) Import for independent TimeEv per energy bin

__all__ = ["PEnergyAnalysis"]

logger = logging.getLogger(__name__)


class PEnergyAnalysis:
    """
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.

    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set dataframe to False if it is not available and want to set the attributes manually.
    energy_edges: List of float
        Edges of the energy binning (in TeV)
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
    energy_edges: list of float
        List of edges for the energy binning
    energy_centres: list of float
        List of center of the energy bins
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    energies; list of float
        List of energies
    tobs : float
        Effective time of observation in hours
    regions: PhaseRegions object
        Information of the OFF/signal regions
    histogram: Lightcurve object
        Information of the phaseogram
    stats: PeriodicityTest object
        Information of Statistical Tests for searching Periodicity
    fitting: PeakFitting object
        Information about the fitting used for the peaks
    """

    def __init__(self, energy_edges, do_diff=True, do_integral=False):
        self.energy_edges = np.array(energy_edges)
        self.energy_centres = (self.energy_edges[1:] + self.energy_edges[:-1]) / 2
        self.do_integral = do_integral
        self.do_diff = do_diff

        if self.do_diff:
            self.integral = False
        else:
            if self.do_integral:
                self.integral = True
            else:
                logger.info(
                    "No energy analysis will be performed. Please set do_diff or do_integral to True"
                )

    ##############################################
    # EXECUTION
    #############################################

    def run(self, pulsarana):
        self.energy_units = pulsarana.energy_units
        self.tobs = pulsarana.tobs
        
        # (LBZ) Initialize both arrays to avoid AttributeError if one mode is disabled
        #self.Parray = [] #(LBZ)
        #self.Parray_integral = [] #(LBZ)

        # (LBZ) Unified energy binning: DIFFERENTIAL and INTEGRAL in a single loop structure
        # DIFFERENTIAL: energy range [edge[i], edge[i+1]]
        # INTEGRAL: energy threshold E > edge[i]
        min_events_threshold = 1  # (LBZ) WARNING: should be moved to config file
        
        if self.do_diff:
            self.Parray=[]
            # Create array of PulsarPhases objects binning in energy (DIFFERENTIAL)
            for i in range(0, len(self.energy_edges) - 1):
                dataframe = pulsarana.info
                di = dataframe[
                    (dataframe["energy"] > self.energy_edges[i])
                    & (dataframe["energy"] < self.energy_edges[i + 1])
                ]

                # (LBZ) Check if bin has enough events
                if len(di) < min_events_threshold:
                    logger.warning(
                        f"Skipping differential energy bin {self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f} {self.energy_units}: "
                        f"only {len(di)} events (< {min_events_threshold} threshold)"
                    )
                    continue
                ################################################################################# (LBZ)

                logger.info(
                    f"Creating differential bin ({self.energy_units}):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f} "
                    f"with {len(di)} events (>= {min_events_threshold} threshold)"
                )
                self.Parray.append(copy.copy(pulsarana)) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                print('copy parray done')
                self.Parray[-1]._energy_bin_index = i # (LBZ) Track original energy bin index for visualization after skipped bins
                print('track_original_index')
                #self.Parray[i].setTimeInterval(self.Parray[i].tint) 
                self.Parray[-1].setTimeInterval(self.Parray[-1].tint) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                print('setTimeInterval done')
                self.Parray[-1].phases = np.array(di["pulsar_phase"].to_list()) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                print('phases_done')
                self.Parray[-1].info = di #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                print('info_done')
                self.Parray[-1].init_regions() #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                print('init_region_done')

                if self.Parray[-1].do_fit: #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                    print('if_do_fit_done')
                    self.Parray[-1].setFittingParams( #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        self.Parray[-1].fit_model, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        self.Parray[-1].binned, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        peak=self.Parray[-1].peak, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                    )

                # Update the information every 1 hour and store final values
                logger.info("Calculating statistics (differential mode)...")
                try:
                    self.Parray[-1].execute_stats(self.tobs) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                    print('execute stats done')
                except (ZeroDivisionError, ValueError, RuntimeError) as e:
                    logger.warning(
                        f"Differential energy bin {self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f} TeV: "
                        f"Error during statistics calculation ({type(e).__name__}). "
                        f"This bin will be kept in results but fitting was skipped."
                    )

                # (LBZ) Summary of differential binning
                total_diff_events = sum(len(obj.info) for obj in self.Parray)
                logger.info(
                    f"Energy differential binning complete: {len(self.Parray)} bins created "
                    f"(out of {len(self.energy_edges)-1} total energy bins) "
                    f"with total {total_diff_events} events"
                )

        if self.do_integral:
            # Create array of PulsarPhases objects with integral energy thresholds (INTEGRAL/CUMULATIVE)
            self.Parray_integral = []
            for i in range(0, len(self.energy_edges) - 1):
                dataframe = pulsarana.info
                di = dataframe[(dataframe["energy"] > self.energy_edges[i])]

                # (LBZ) Check if bin has enough events (same threshold applied to different data)
                if len(di) < min_events_threshold:
                    logger.warning(
                        f"Skipping integral energy bin E > {self.energy_edges[i]:.2f} {self.energy_units}: "
                        f"only {len(di)} events (< {min_events_threshold} threshold)"
                    )
                    continue
                ########################################################################################################

                logger.info(
                    f"Creating integral bin ({self.energy_units}):E > {self.energy_edges[i]:.2f} "
                    f"with {len(di)} events (>= {min_events_threshold} threshold)"
                )
                self.Parray_integral.append(copy.copy(pulsarana))
                self.Parray_integral[-1]._energy_bin_index = i  # (LBZ) Track original energy bin index for visualization after skipped bins
                #self.Parray[-1].TimeEv = PulsarTimeAnalysis(tint=self.Parray[-1].tint)
                self.Parray_integral[-1].setTimeInterval(self.Parray_integral[-1].tint) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                self.Parray_integral[-1].phases = np.array(di["pulsar_phase"].to_list()) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                self.Parray_integral[-1].info = di #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)

                self.Parray_integral[-1].init_regions() #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)

                if self.Parray_integral[-1].do_fit: #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                    self.Parray_integral[-1].setFittingParams( #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        self.Parray_integral[-1].fit_model, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        self.Parray_integral[-1].binned, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                        peak=self.Parray_integral[-1].peak, #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                    )

                logger.info("Calculating statistics (integral mode)...")
                try:
                    self.Parray_integral[-1].execute_stats(self.tobs) #(LBZ) energy bin low events modification (-1 instead of i to avoid index out of range)
                except (ZeroDivisionError, ValueError, RuntimeError) as e:
                    logger.warning(
                        f"Integral energy bin E > {self.energy_edges[i]:.2f} TeV: "
                        f"Error during statistics calculation ({type(e).__name__}). "
                        f"This bin will be kept in results but fitting was skipped."
                    )

                # (LBZ) Summary of integral binning
                total_integral_events = sum(len(obj.info) for obj in self.Parray_integral)
                logger.info(
                    f"Energy integral binning complete: {len(self.Parray_integral)} bins created "
                    f"(out of {len(self.energy_edges)-1} total energy thresholds) "
                    f"with total {total_integral_events} events"
                )

    ##############################################
    # RESULTS
    #############################################

    def show_Energy_lightcurve(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        fig_array = []
        for idx, obj in enumerate(histogram_array): #(LBZ)
            # Get original energy bin index to handle skipped bins correctly
            i = obj._energy_bin_index #(LBZ)
            # Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
            fig = plt.figure(figsize=(12, 5))
            histogram_array[idx].histogram.show_phaseogram( #(LBZ)
                histogram_array[idx], #(LBZ)
                [0, 2],
                colorhist="C" + str(idx), #(LBZ)
                fit=True,
                time_label=False,
                stats_label=False,
                add_legend=False,
            )

            if integral:
                energy_label = (
                    "ENERGY RANGE (GeV):" + f" E > {self.energy_edges[i]*1000:.0f}"
                )
            else:
                energy_label = (
                    "ENERGY RANGE (GeV):"
                    + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}"
                )

            plt.annotate(
                energy_label,
                xy=(0.1, 0.9),
                xytext=(0.31, 0.9),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="left",
                verticalalignment="top",
            )

            text_towrite = ""
            count = 0
            # Map dictionary keys to region attributes (P1+P2 → P1P2) # (TEST1)
            peak_mapping = {"P1": "P1", "P2": "P2", "P1+P2": "P1P2"} #(TEST1)
            for key, value in histogram_array[idx].regions.dic.items(): #(LBZ)
                if value is not None:
                    ################### #(TEST1)
                    # Get the actual peak object using the mapping
                    attr_name = peak_mapping.get(key) 
                    if attr_name and hasattr(histogram_array[idx].regions, attr_name):
                        peak_obj = getattr(histogram_array[idx].regions, attr_name)
                        if hasattr(peak_obj, 'sign'):
                            if count == 0:
                                text_towrite = (
                                    text_towrite
                                    + key
                                    + f": Sig(Li&Ma):{peak_obj.sign:.2f}$\sigma$"
                                )
                                count += 1
                            else:
                                text_towrite = (
                                    text_towrite
                                    + "\n"
                                    + key
                                    + f": Sig(Li&Ma):{peak_obj.sign:.2f}$\sigma$"
                                )
                    ############ #(TEST1) 
            plt.annotate(
                text_towrite + "\n" + f"Entries={len(histogram_array[idx].phases)}", # (LBZ)
                xy=(1.05, 1.0),
                xytext=(1.05, 1.0),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black"),
                horizontalalignment="left",
                verticalalignment="top",
            )

            plt.legend(loc=4, bbox_to_anchor=(1.2, 0), fontsize=15)
            fig_array.append(fig)
            plt.show()

        return fig_array

    def show_joined_Energy_fits(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        for i in range(0, len(histogram_array)):
            if histogram_array[i].fitting.check_fit_result():
                fig = plt.figure(figsize=(17, 8))
                break
            elif (
                i == len(histogram_array) - 1
                and not histogram_array[i].fitting.check_fit_result()
            ):
                print("No fit available for any energy bin")
                return
        for idx, obj in enumerate(histogram_array):
            if histogram_array[idx].fitting.check_fit_result():
                # Get original energy bin index to handle skipped bins correctly (LBZ)
                i = histogram_array[idx]._energy_bin_index
                histogram_array[idx].histogram.draw_fitting(
                    histogram_array[idx],
                    color="C" + str(idx),
                    density=True,
                    label="Energies(GeV):"
                    + f"{self.energy_edges[i]*1000:.2f}-{self.energy_edges[i+1]*1000:.2f}",
                )
        plt.xlim(
            2 * histogram_array[0].fitting.shift,
            1 + 2 * histogram_array[-1].fitting.shift, #(LBZ)
        )
        plt.legend(fontsize=20)
        return fig

    def show_joined_Energy_lightcurve(
        self,
        colorh=[
            "tab:red",
            "tab:purple",
            "tab:blue",
            "tab:brown",
            "tab:cyan",
            "tab:olive",
            "tab:pink",
        ],
        colorP=["orange", "green", "purple"],
        ylimits=None,
    ):
        fig = plt.figure(figsize=(17, 8))

        # (LBZ) Iterate using actual objects to get correct energy_bin_index for skipped bins
        for idx, obj in enumerate(self.Parray):
            i = obj._energy_bin_index  # Get original energy bin index
            obj.histogram.draw_density_hist(
                [0.7, 1.7],
                colorhist=colorh[idx],
                label="Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}",
                fill=False,
            )

        self.Parray[-1].histogram.draw_background(self.Parray[-1], "grey", hline=False) # (LBZ)

        signal = ["P1", "P2", "P1+P2"] # (LBZ) Fixed: removed non-existent P3
        for j in range(0, len(signal)):
            self.Parray[0].histogram.draw_peakregion(
                self.Parray[0], signal[j], color=colorP[j]
            )

        plt.legend(fontsize=15)

        if ylimits is not None:
            plt.ylim(ylimits)

        return fig

    def show_EnergyPresults(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0:
            logger.warning("No energy bins available for show_EnergyPresults (all bins skipped)")
            return [] #(LBZ)

        # (LBZ) Accumulate results for ALL bins instead of just returning the last one
        all_results = [] #(LBZ)

        # (LBZ) Iterate over actual histogram_array instead of energy_edges to handle skipped bins
        for idx, obj in enumerate(histogram_array):
            i = obj._energy_bin_index  # Get original energy bin index
            print(
                "Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}"
                + "\n"
            )
            peak_stat, p_stat = obj.show_Presults() #(LBZ)
            all_results.append((peak_stat, p_stat))  # (LBZ) Accumulate result
            print("\n \n")
            print("-------------------------------------------------------------------")

        return all_results  # (LBZ) Return list of ALL results 

    def show_Energy_fitresults(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Iterate over actual histogram_array instead of energy_edges to handle skipped bins
        fit_results = []  # (LBZ)

        for obj in histogram_array:
            i = obj._energy_bin_index  # Get original energy bin index
            print(
                "Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.2f}-{self.energy_edges[i+1]*1000:.2f}"
                + "\n"
            )
            if obj.fitting.check_fit_result(): # (LBZ)
                fit_results.append(obj.show_fit_results()) # (LBZ)
            else: 
                print("No fit available for this energy range")
                fit_results.append(None) # (LBZ)
            print("\n \n")
            print("-------------------------------------------------------------------")

        return fit_results

    def PSigVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0:
            logger.warning("No energy bins available for PSigVsEnergy plot (all bins skipped)")
            return

        # (LBZ) Iterate over actual histogram_array instead of energy_centres to handle skipped bins
        P1_s = []
        P2_s = []
        P1P2_s = [] # ISSUE ? why not the same color ? 
        energy_centres_actual = []

        for obj in histogram_array: # (LBZ)
            i = obj._energy_bin_index  # Get original energy bin index
            energy_centres_actual.append(self.energy_centres[i]) # (LBZ)
            if obj.regions.dic["P1"] is not None: # (LBZ)
                P1_s.append(obj.regions.P1.sign) # (LBZ)
            if obj.regions.dic["P2"] is not None: # (LBZ)
                P2_s.append(obj.regions.P2.sign) # (LBZ)
            if obj.regions.dic["P1+P2"] is not None: # (LBZ)
                P1P2_s.append(obj.regions.P1P2.sign) # (LBZ) ISSUE ? why not the same color ? 

        # Only plot if we have data to display
        #if len(P1P2_s) > 0 or len(P1_s) > 0 or len(P2_s) > 0: (ISSUE ? )
        if len(P1P2_s) > 0:
            plt.plot(energy_centres_actual, P1P2_s, "o-", color="tab:red", label="P1+P2") # (LBZ)

        if len(P1_s) > 0:
            plt.plot(energy_centres_actual, P1_s, "o-", color="tab:orange", label="P1") # (LBZ)

        if len(P2_s) > 0:
            plt.plot(energy_centres_actual, P2_s, "o-", color="tab:green", label="P2") # (LBZ)

        plt.ylabel("Significance($\sigma$)")
        plt.xticks(
            [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
            labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
        )
        plt.xlabel("E (GeV)")
        plt.legend()
        plt.grid(which="both")
        #try:
        plt.xscale("log")
        #except ValueError as e: #ISSUE ?
        #logger.warning(f"Cannot set log scale for PSigVsEnergy: {e}. Using linear scale.") #ISSUE ?
        # else: #ISSUE ?
        #     plt.annotate(
        #         "No valid data to plot (PSigVsEnergy)", #ISSUE ?
        #         xy=(0.5, 0.5), #ISSUE ?
        #         xycoords="axes fraction", #ISSUE ?
        #         ha="center", #ISSUE ?
        #         va="center", #ISSUE ?
        #         fontsize=12, #ISSUE ?
        #     )

    def P1P2_ratioVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        P1P2E = []
        P1P2E_error = []
        energy_centres_actual = [] # (LBZ)

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0:
            logger.warning("No energy bins available for P1P2_ratioVsEnergy (all bins skipped)")
            return (P1P2E, P1P2E_error, energy_centres_actual)

        # (LBZ) Iterate over actual histogram_array instead of energy_centres to handle skipped bins
        # Check if P1P2_ratio is not None before accessing it (TEST: defensive check)
        #if len(histogram_array) > 0 and histogram_array[0].regions.P1P2_ratio is not None:  #(TEST)
        if len(histogram_array) > 0:  #(TEST)
            for obj in histogram_array: #(LBZ)
                i = obj._energy_bin_index  # Get original energy bin index #(LBZ)
                energy_centres_actual.append(self.energy_centres[i]) #(LBZ)
                P1P2E.append(obj.regions.P1P2_ratio) #(LBZ)
                P1P2E_error.append(obj.regions.P1P2_ratio_error) #(LBZ)
        else:
            print("Cannot calculate P1/P2 since one of the peaks is not defined or P1P2_ratio not computed")

        return (P1P2E, P1P2E_error, energy_centres_actual) # (LBZ)

    def P1P2VsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        ratio, ratio_error, energy_centres_actual = self.P1P2_ratioVsEnergy(integral=integral) # (LBZ)
        
        # Only plot if we have data
        #if len(ratio) > 0 and len(energy_centres_actual) > 0: #ISSUE ?
            # (LBZ) Use actual energy centres computed from non-skipped bins
        plt.fill_between(
            energy_centres_actual, # (LBZ)
            np.array(ratio) + np.array(ratio_error),
            np.array(ratio) - np.array(ratio_error),
            alpha=0.3,
        )
        plt.plot(energy_centres_actual, ratio, "o-", label="LST-1") # (LBZ)

        plt.ylabel("P1/P2")
        plt.xticks(
            [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
            labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
        )
        plt.xlabel("E (GeV)")
        plt.title("P1/P2 vs Energy")
        plt.grid(which="both")
        try:
            plt.xscale("log")
        except ValueError as e:
            logger.warning(f"Cannot set log scale for P1P2VsEnergy: {e}. Using linear scale.")
    # else:
    #         plt.annotate(
    #             "No valid data to plot (P1_P2_ratioVsEnergy)",
    #             xy=(0.5, 0.5),
    #             xycoords="axes fraction",
    #             ha="center",
    #             va="center",
    #             fontsize=12,
    #         )

    def FWHMVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        FP1 = []
        FP2 = []
        FP1_err = []
        FP2_err = []
        energies_F1 = []
        energies_F2 = []

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0:
            logger.warning("No energy bins available for FWHMVsEnergy plot (all bins skipped)")
            # plt.annotate(
            #     "No valid data to plot(FWHMVsEnergy)",
            #     xy=(0.5, 0.5),
            #     xycoords="axes fraction",
            #     ha="center",
            #     va="center",
            #     fontsize=12,
            # )
            return

        # (LBZ) Build energy_centres_actual once from non-skipped bins
        energy_centres_actual = [] # (LBZ)
        for obj in histogram_array: # (LBZ)
            i = obj._energy_bin_index # (LBZ) #ISSUE ? (why not the same color _energy_bin_index attribute?
            energy_centres_actual.append(self.energy_centres[i]) # (LBZ)

        if histogram_array[0].fitting.model == "asym_dgaussian":
            prefactor = 2.35482
            # (LBZ) Iterate over actual objects to handle skipped bins
            for idx, obj in enumerate(histogram_array): # (LBZ)
                try:
                    FP1.append(
                        prefactor * obj.fitting.params[1] / 2 # (LBZ)
                        + prefactor * obj.fitting.params[2] / 2 # (LBZ)
                    )
                    energies_F1.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        FP1_err.append(
                            FP1[-1] # (LBZ)
                            * np.sqrt(
                                (
                                    obj.fitting.errors[1] # (LBZ) error before (histogram_array[i].errors.params[1])
                                    / obj.fitting.params[1] # (LBZ)
                                )
                                ** 2
                                + (
                                    obj.fitting.errors[2] # (LBZ)
                                    / obj.fitting.params[2] # (LBZ)
                                )
                                ** 2
                            )
                        )
                    except (AttributeError, TypeError):
                        FP1_err.append(0)
                except (AttributeError, TypeError):
                    pass

                try:
                    FP2.append(
                        prefactor * obj.fitting.params[4] / 2 # (LBZ)
                        + prefactor * obj.fitting.params[5] / 2 # (LBZ)
                    )
                    energies_F2.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        FP2_err.append(
                            FP2[-1] # (LBZ) 
                            * np.sqrt(
                                (
                                    obj.fitting.errors[4] # (LBZ) error before (histogram_array[i].errors.params[1])
                                    / obj.fitting.params[4] # (LBZ)
                                )
                                ** 2
                                + (
                                    obj.fitting.errors[5] # (LBZ)
                                    / obj.fitting.params[5] # (LBZ)
                                )
                                ** 2
                            )
                        )
                    except (AttributeError, TypeError):
                        FP2_err.append(0)
                except (AttributeError, TypeError):
                    pass

        else:
            if histogram_array[0].fitting.model in ("dgaussian", "dgaussian_heaviside", "dgaussian_step"): # (LBZ)
                prefactor = 2.35482

            elif histogram_array[0].fitting.model == "lorentzian":
                prefactor = 2

            else:
                prefactor = 0

            # (LBZ) Iterate over actual objects to handle skipped bins
            for idx, obj in enumerate(histogram_array): # (LBZ)
                try:
                    FP1.append(prefactor * obj.fitting.params[1]) # (LBZ)
                    energies_F1.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        FP1_err.append(prefactor * obj.fitting.errors[1]) # (LBZ)
                    except (AttributeError, TypeError):
                        FP1_err.append(0)
                except (AttributeError, TypeError):
                    pass

                try:
                    FP2.append(prefactor * obj.fitting.params[3]) # (LBZ)
                    energies_F2.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        FP2_err.append(prefactor * obj.fitting.errors[3]) # (LBZ)
                    except (AttributeError, TypeError):
                        FP2_err.append(0)
                except (AttributeError, TypeError):
                    pass

        energies_F1 = np.array(energies_F1)
        energies_F2 = np.array(energies_F2)
        FP1 = np.array(FP1)
        FP2 = np.array(FP2)
        FP1_err = np.array(FP1_err)
        FP2_err = np.array(FP2_err)

        if len(FP1) > 0:
            plt.errorbar(
                energies_F1 * 1000,
                FP1,
                yerr=FP1_err,
                fmt="o-",
                color="tab:orange",
                label="P1",
            )
        if len(FP2) > 0:
            plt.errorbar(
                energies_F2 * 1000,
                FP2,
                yerr=FP2_err,
                fmt="o-",
                color="tab:green",
                label="P2",
            )
        if len(FP1) <= 0 and len(FP2) <= 0:
            plt.annotate(
                "Plot not available",
                xy=(0.6, 0.6),
                xytext=(0.6, 0.6),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )
        else:
            plt.ylabel("FWHM")
            plt.xlabel("E (GeV)")
            plt.legend()
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

    def MeanVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0: # (LBZ)
            logger.warning("No energy bins available for MeanVsEnergy plot (all bins skipped)") # (LBZ)
            return # (LBZ)

        M1 = []
        M2 = []
        M1_err = []
        M2_err = []
        energies_M1 = []
        energies_M2 = []

        # (LBZ) Build energy_centres_actual once from non-skipped bins
        energy_centres_actual = [] # (LBZ)
        for obj in histogram_array: # (LBZ)
            i = obj._energy_bin_index # (LBZ)
            energy_centres_actual.append(self.energy_centres[i]) # (LBZ)

        if histogram_array[0].fitting.model == "asym_dgaussian":
            # (LBZ) Iterate over actual objects to handle skipped bins
            for idx, obj in enumerate(histogram_array): # (LBZ)
                try:
                    M1.append(obj.fitting.params[0]) # (LBZ)
                    energies_M1.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        M1_err.append(obj.fitting.errors[0]) # (LBZ)
                    except (AttributeError, TypeError):
                        M1_err.append(0)
                except (AttributeError, TypeError):
                    pass

                try:
                    M2.append(obj.fitting.params[3]) # (LBZ)
                    energies_M2.append(energy_centres_actual[idx]) # (LBZ)
                    try:
                        M2_err.append(obj.fitting.errors[3]) # (LBZ) before it was M1 err ???????? 
                    except (AttributeError, TypeError):
                        M2_err.append(0)  # (LBZ) before it was M1 err ????????
                except (AttributeError, TypeError): 
                    pass

        elif (
            histogram_array[0].fitting.model == "dgaussian"
            or histogram_array[0].fitting.model == "dgaussian_heaviside" # (LBZ)
            or histogram_array[0].fitting.model == "dgaussian_step" #(LBZ)
            or histogram_array[0].fitting.model == "lorentzian"
        ):
            # (LBZ) Iterate over actual objects to handle skipped bins
            for idx, obj in enumerate(histogram_array): # (LBZ) 
                try:
                    M1.append(obj.fitting.params[0]) # (LBZ) 
                    energies_M1.append(energy_centres_actual[idx]) # (LBZ) 
                    try:
                        M1_err.append(obj.fitting.errors[0]) # (LBZ) 
                    except (AttributeError, TypeError):
                        M1_err.append(0)
                except (AttributeError, TypeError):
                    pass

                try:
                    M2.append(obj.fitting.params[2])   # (LBZ)
                    energies_M2.append(energy_centres_actual[idx])  # (LBZ)
                    try:
                        M2_err.append(obj.fitting.errors[2])  # (LBZ)
                    except (AttributeError, TypeError):
                        M2_err.append(0)
                except (AttributeError, TypeError):
                    pass

        if len(M1) == 0 and len(M2) == 0:
            print("No fit available for plotting")
            return
        elif len(M1) > 0 and len(M2) > 0:
            nplots = 2
        else:
            nplots = 1

        fig = plt.figure(figsize=(10, 5))
        if len(M1) > 0:
            plt.subplot(nplots, 1, 1)
            #energies_M1 = np.array(energies_M1)  #(LBZ) #ISSUE ?
            #M1 = np.array(M1)  #(LBZ)  #ISSUE ?
            #M1_err = np.array(M1_err)  #(LBZ) #ISSUE ?
            plt.errorbar(energies_M1, M1, yerr=M1_err, fmt="o-", color="#0C8F1ECA", label="Data")
            
            ##################################################################################
            # (LBZ) Linear fit using iminuit: a*E + b
            def linear_model(E, a, b):  #(LBZ)
                return a * np.log10(E) + b  #(LBZ)
            
            chi2_M1 = LeastSquares(energies_M1, M1, M1_err, linear_model)  #(LBZ)
            minuit_M1 = Minuit(chi2_M1, a=1., b=0)  #(LBZ) ATTENTION maybe a and b not hardcoded after 
            minuit_M1.migrad()  #(LBZ)
            a_M1 = minuit_M1.values["a"]  #(LBZ)
            b_M1 = minuit_M1.values["b"]  #(LBZ)
            a_err_M1 = minuit_M1.errors["a"]  #(LBZ)
            b_err_M1 = minuit_M1.errors["b"]  #(LBZ)
            
            # (LBZ) Log detailed Minuit fit information
            logger.info(f"P1 Mean Phase Fit Results:")
            logger.info(f"  Fit converged: {minuit_M1.valid}")
            logger.info(f"  χ² value: {minuit_M1.fval:.4f}")
            logger.info(f"  Degrees of freedom: {len(energies_M1) - 2}")
            logger.info(f"  Reduced χ²: {minuit_M1.fval / (len(energies_M1) - 2):.4f}")
            logger.info(f"  Parameters: a = ({a_M1:.6f} ± {a_err_M1:.6f}), b = ({b_M1:.6f} ± {b_err_M1:.6f})")
            logger.info(f"  Linear model: y = ({a_M1:.6f} ± {a_err_M1:.6f})·E + ({b_M1:.6f} ± {b_err_M1:.6f})")
            
            fit_energies_M1 = np.linspace(min(energies_M1), max(energies_M1), 100)  #(LBZ)
            fit_values_M1 = linear_model(fit_energies_M1, a_M1, b_M1)  #(LBZ)
            plt.plot(fit_energies_M1, fit_values_M1, linestyle='-',color="#075211B8", linewidth=2, label=f"Fit: {a_M1:.4f}·E + {b_M1:.4f}")  #(LBZ)
            
            ##################################################################################
            plt.ylabel("Mean phase")
            plt.xticks(
                [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
                labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
            )
            plt.xlabel("E (GeV)")
            plt.title("P1 mean phase")
            plt.legend()  # (LBZ)
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

        if len(M2) > 0:
            plt.subplot(nplots, 1, nplots)
            # energies_M2 = np.array(energies_M2)  #(LBZ) #ISSUE ?
            # M2 = np.array(M2)  #(LBZ) #ISSUE ?
            # M2_err = np.array(M2_err)  #(LBZ)  #ISSUE ?
            plt.errorbar(energies_M2, M2, yerr=M2_err, fmt="o-", color="#F38E0AD5", label="Data")
            
            #########################################################################################
            # (LBZ) Linear fit using iminuit: a*E + b
            def linear_model(E, a, b):  #(LBZ)
                return a * np.log10(E) + b  #(LBZ)
            
            chi2_M2 = LeastSquares(energies_M2, M2, M2_err, linear_model)  #(LBZ)
            minuit_M2 = Minuit(chi2_M2, a=1., b=0)  #(LBZ)
            minuit_M2.migrad()  #(LBZ)
            a_M2 = minuit_M2.values["a"]  #(LBZ)
            b_M2 = minuit_M2.values["b"]  #(LBZ)
            a_err_M2 = minuit_M2.errors["a"]  #(LBZ)
            b_err_M2 = minuit_M2.errors["b"]  #(LBZ)
            
            # (LBZ) Log detailed Minuit fit information
            logger.info(f"P2 Mean Phase Fit Results:")
            logger.info(f"  Fit converged: {minuit_M2.valid}")
            logger.info(f"  χ² value: {minuit_M2.fval:.4f}")
            logger.info(f"  Degrees of freedom: {len(energies_M2) - 2}")
            logger.info(f"  Reduced χ²: {minuit_M2.fval / (len(energies_M2) - 2):.4f}")
            logger.info(f"  Parameters: a = ({a_M2:.6f} ± {a_err_M2:.6f}), b = ({b_M2:.6f} ± {b_err_M2:.6f})")
            logger.info(f"  Linear model: y = ({a_M2:.6f} ± {a_err_M2:.6f})·E + ({b_M2:.6f} ± {b_err_M2:.6f})")
            
            fit_energies_M2 = np.linspace(max(energies_M2), max(energies_M2), 100)  #(LBZ)
            fit_values_M2 = linear_model(fit_energies_M2, a_M2, b_M2)  #(LBZ)
            plt.plot(fit_energies_M2, fit_values_M2, color="#D37D0CB9",linestyle='-', linewidth=2, label=f"Fit: {a_M2:.4f}·E + {b_M2:.4f}")  #(LBZ)
            ################################################################################################


            plt.title("P2 mean phase")
            plt.ylabel("Mean phase")
            plt.xticks(
                [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
                labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
            )
            plt.xlabel("E (GeV)")
            plt.legend()  # (LBZ)
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

        return fig

    # def WidthVsEnergy(self, integral=None):  #(LBZ)
    #     """Plot sigma (width) vs energy for P1 and P2 peaks."""  #(LBZ)
    #     if integral is None:
    #         integral = self.integral

        # if integral:
        #     if not self.do_integral:
        #         raise ValueError(
        #             "Energy Integral results not produced. Check if do_integral parameter is set to True"
        #         )
        #     histogram_array = self.Parray_integral
        # else:
        #     if not self.do_diff:
        #         raise ValueError(
        #             "Energy Differential results not produced. Check if do_diff parameter is set to True"
        #         )
        #     histogram_array = self.Parray

        # if len(histogram_array) == 0:
        #     logger.warning("No energy histograms available for WidthVsEnergy plot")
        #     return

        # W1 = []
        # W2 = []
        # W1_err = []
        # W2_err = []
        # energies_W1 = []
        # energies_W2 = []
        
        ################################ see later to implement the asym sgaussian ########################
        # For asym_dgaussian model: params = [mean1, sigma1_left, sigma1_right, mean2, sigma2_left, sigma2_right]  #(LBZ)
        # if histogram_array[0].fitting.model == "asym_dgaussian":  #(LBZ)
        #     for i in range(0, len(self.energy_centres)):  #(LBZ)
        #         try:  #(LBZ)
        #             W1.append(histogram_array[i].fitting.params[1])  # sigma1_left  #(LBZ)
        #             energies_W1.append(self.energy_centres[i])  #(LBZ)
        #             try:  #(LBZ)
        #                 W1_err.append(histogram_array[i].fitting.errors[1])  #(LBZ)
        #             except AttributeError:  #(LBZ)
        #                 W1_err.append(0)  #(LBZ)
        #         except AttributeError:  #(LBZ)
        #             pass  #(LBZ)
        #         try:  #(LBZ)
        #             W2.append(histogram_array[i].fitting.params[4])  # sigma2_left  #(LBZ)
        #             energies_W2.append(self.energy_centres[i])  #(LBZ)
        #             try:  #(LBZ)
        #                 W2_err.append(histogram_array[i].fitting.errors[4])  #(LBZ)
        #             except AttributeError:  #(LBZ)
        #                 W2_err.append(0)  #(LBZ)
        #         except AttributeError:  #(LBZ)
        #             pass  #(LBZ)
        ########################################################################################################""

        # For dgaussian/lorentzian models: params = [mean1, sigma1, mean2, sigma2]
        # if (
        #     histogram_array[0].fitting.model == "dgaussian"
        #     or histogram_array[0].fitting.model == "lorentzian"
        # ):
        #     for i in range(0, len(self.energy_centres)):
        #         try:
        #             W1.append(histogram_array[i].fitting.params[1])  # sigma1
        #             energies_W1.append(self.energy_centres[i])
        #             try:
        #                 W1_err.append(histogram_array[i].fitting.errors[1])
        #             except AttributeError:
        #                 W1_err.append(0)
        #         except AttributeError:
        #             pass

                # try:
                #     W2.append(histogram_array[i].fitting.params[3])  # sigma2
                #     energies_W2.append(self.energy_centres[i])
                #     try:
                #         W2_err.append(histogram_array[i].fitting.errors[3])
                #     except AttributeError:
                #         W2_err.append(0)
                # except AttributeError:
                #     pass

    #         if len(W1) == 0 and len(W2) == 0:
    #             print("No fit available for plotting")
    #             return
    #         elif len(W1) > 0 and len(W2) > 0:
    #             nplots = 2
    #         else:
    #             nplots = 1

    #         fig = plt.figure(figsize=(10, 5))
    #         if len(W1) > 0:
    #             plt.subplot(nplots, 1, 1)
    #             plt.errorbar(energies_W1, W1, yerr=W1_err, fmt="o-", color="tab:orange")
    #             plt.ylabel("Mean width")
    #             plt.xticks(
    #                 [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
    #                 labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
    #             )
    #             plt.xlabel("E (GeV)")
    #             plt.title("P1 mean width")
    #             plt.tight_layout()
    #             plt.grid(which="both")
    #             plt.xscale("log")

    #         if len(W2) > 0:
    #             plt.subplot(nplots, 1, nplots)
    #             plt.errorbar(energies_W2, W2, yerr=W2_err, fmt="o-", color="tab:green")
    #             plt.title("P2 mean width")
    #             plt.ylabel("Mean width")
    #             plt.xticks(
    #                 [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
    #                 labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
    #             )
    #             plt.xlabel("E (GeV)")
    #             plt.tight_layout()
    #             plt.grid(which="both")
    #             plt.xscale("log")

        # return fig

    ########################################################################################################## (LBZ)
    

    def PeaksVsEnergy(self, integral=None):
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        self.PSigVsEnergy(integral)

        plt.subplot(1, 3, 2)
        self.P1P2VsEnergy(integral)

        plt.subplot(1, 3, 3)
        self.FWHMVsEnergy(integral)

        plt.tight_layout()
        plt.show()

    #(LBZ) Get temporal results for each energy bin
    def get_Energy_TimeResults(self, integral=None):
        """
        Return temporal analysis plots (PsigVsTime, PexVsTime, StatsVsTime) for each energy bin.
        Returns a flat list of figure objects (fig1, fig2, fig3, fig4, fig5, fig6, ...).
        Pattern: same as show_Energy_lightcurve() which also returns fig_array.
        """
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )
            histogram_array = self.Parray_integral
            analysis_type = "Integral"
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )
            histogram_array = self.Parray
            analysis_type = "Differential"

        # (LBZ) Defensive check: all bins might be skipped
        if len(histogram_array) == 0:
            logger.warning(f"No energy bins available for {analysis_type} time results")
            return []  #(LBZ) Return empty list (consistent with other show_*Energy methods)

        time_results_list = []  #(LBZ) List to store all figure objects (flat list)

        # (LBZ) Iterate over each energy bin and get temporal plots
        for idx, obj in enumerate(histogram_array):
            i = obj._energy_bin_index  # Get original energy bin index
            
            # Get energy range for this bin
            if integral:
                energy_label = f"E > {self.energy_edges[i]*1000:.0f} GeV"
            else:
                energy_label = f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f} GeV"
            
            # Only get results if TimeAnalysis object exists and has data
            if hasattr(obj, 'TimeEv') and obj.TimeEv is not None:
                try:
                    fig1, fig2, fig3 = obj.TimeEv.show_results()
                    
                    # Add energy range to figure titles
                    fig1.suptitle(f"Significance vs Time - {analysis_type} ({energy_label})", fontsize=14, y=1.00)  #(LBZ)
                    fig2.suptitle(f"Excess Events vs Time - {analysis_type} ({energy_label})", fontsize=14, y=1.00)  #(LBZ)
                    fig3.suptitle(f"Statistical Tests vs Time - {analysis_type} ({energy_label})", fontsize=14, y=1.00)  #(LBZ)
                    
                    # Append all 3 figures to the flat list  #(LBZ)
                    time_results_list.append(fig1)  #(LBZ)
                    time_results_list.append(fig2)  #(LBZ)
                    time_results_list.append(fig3)  #(LBZ)
                except Exception as e:  #(LBZ)
                    logger.warning(f"Could not get temporal results for energy bin {energy_label}: {e}")  #(LBZ)
            else:  #(LBZ)
                logger.debug(f"No temporal analysis data for energy bin {energy_label}")  #(LBZ)

        return time_results_list  #(LBZ) Return flat list of figures
