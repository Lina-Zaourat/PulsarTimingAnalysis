import numpy as np
import pandas as pd
from gammapy.stats import WStatCountsStatistic
import logging  #(LBZ)

__all__ = ["calculate_CountStats", "PhaseRegions", "PulsarPeak"]

logger = logging.getLogger(__name__) #(LBZ)


def calculate_CountStats(on_file, off_file=None, factor=None):
    if off_file is None:
        raise ValueError("No off data given for the pulsar analysis")
    elif on_file is None: #(LBZ07)
            raise ValueError("No on data given for the pulsar analysis") #(LBZ07)
    
    Non = len(on_file)
    print('Non :',Non)
    Noff = len(off_file)
    print('Noff :',Noff)
    stat = WStatCountsStatistic(n_on=Non, n_off=Noff, alpha=factor)
    print('stat',stat)
    yerr = np.sqrt(Non + ((factor**2) * Noff))

    return (stat, yerr, Noff * factor)


class PhaseRegions:
    def __init__(
        self,
        OFF_object=None,
        P1_object=None,
        P2_object=None,
        P1P2_object=None,
        P3_object=None,
        #P1P2P3_object=None, #(LBZ)
    ):
        # Define background and null regions for both peaks and
        if OFF_object is None:
            print("No peak statistics available since no background is provided")
        else:
            self.OFF = OFF_object

        # Create the peaks in the previosly defined null regions
        #self.create_dic(P1_object, P2_object, P1P2_object, P3_object,P1P2P3_object) #(LBZ)
        self.create_dic(P1_object, P2_object, P1P2_object, P3_object)

    ##############################################
    # EXECUTION
    #############################################

    def create_dic(
        # self, P1_object=None, P2_object=None, P1P2_object=None, P3_object=None,P1P2P3_object=None, #(LBZ)
        self, P1_object=None, P2_object=None, P1P2_object=None, P3_object=None,
    ):
        if P1_object is not None:
            self.P1 = P1_object
            npeaks = 1
        else:
            npeaks = 0
            self.P1 = None
            print("No P1 limits. Cant create P1 object")

        if P2_object is not None:
            self.P2 = P2_object
            npeaks = npeaks + 1
        else:
            self.P2 = None
            print("No P2 limits. Cant create P2 object")

        if P1P2_object is not None:
            self.P1P2 = P1P2_object
        else:
            self.P1P2 = None

        if P3_object is not None:
            self.P3 = P3_object
            npeaks = npeaks + 1
        else:
            self.P3 = None

        # if P1P2P3_object is not None: #(LBZ)
        #     self.P1P2P3 = P1P2P3_object #(LBZ)
        # else: #(LBZ)
        #     self.P1P2P3 = None #(LBZ)

        self.npeaks = npeaks

        self.dic = {
            "P1": P1_object,
            "P2": P2_object,
            "P1+P2": P1P2_object,
            "P3": P3_object,
            #"P1+P2+P3": P1P2P3_object, #(LBZ)
        }
        
        # Initialize ratio attributes to None to ensure they always exist (TEST: defensive initialization)
        # self.P1P2_ratio = None # TEST # ISSUE ? 
        # self.P1P2_ratio_error = None # TEST # ISSUE ? 
        # self.P1P3_ratio = None # TEST # ISSUE ? 
        # self.P1P3_ratio_error = None # TEST # ISSUE ? 
        # self.P2P3_ratio = None # TEST # ISSUE ? 
        # self.P2P3_ratio_error = None # TEST # ISSUE ? 

    def remove_peak(self, name):
        del self.dic[name]
        self.npeaks = self.npeaks - 1

    def create_peak(self, peak_object, name):
        if self.dic[name]:
            self.dic[name] = peak_object

        else:
            self.dic[name] = peak_object
            self.npeaks = self.npeaks + 1

    def calculate_P1P2(self):
        print('enter_calulate_P1P2')
        if self.dic["P1"] is not None and self.dic["P2"] is not None :
            print('P1.Nex',self.P1.Nex ) # (LBZ07)
            print('P2.Nex',self.P2.Nex )# (LBZ07)
            self.P1P2_ratio = self.P1.Nex / self.P2.Nex
            self.P1P2_ratio_error = self.P1P2_ratio * np.sqrt(
                (self.P1.yerr / self.P1.Nex) ** 2 + (self.P2.yerr / self.P2.Nex) ** 2
                )
        else:
             
             self.P1P2_ratio = None
             self.P1P2_ratio_error = None

    # # def calculate_P1P3(self): #(LBZ)
    #     if self.dic["P1"] is not None and self.dic["P3"] is not None: #(LBZ)
    #         self.P1P3_ratio = self.P1.Nex / self.P3.Nex #(LBZ)
    #         self.P1P3_ratio_error = self.P1P3_ratio * np.sqrt( #(LBZ)
    #             (self.P1.yerr / self.P1.Nex) ** 2 + (self.P3.yerr / self.P3.Nex) ** 2 #(LBZ)
    #         ) #(LBZ)
    #     else:
    #         self.P1P3_ratio = None #(LBZ)
    #         self.P1P3_ratio_error = None #(LBZ)
        
    # def calculate_P2P3(self): #(LBZ)
    #     if self.dic["P2"] is not None and self.dic["P3"] is not None: #(LBZ)
    #         self.P2P3_ratio = self.P2.Nex / self.P3.Nex #(LBZ)
    #         self.P2P3_ratio_error = self.P2P3_ratio * np.sqrt( #(LBZ)
    #             (self.P2.yerr / self.P2.Nex) ** 2 + (self.P3.yerr / self.P3.Nex) ** 2 #(LBZ)
    #         ) #(LBZ)

    #     else:
    #         self.P2P3_ratio = None #(LBZ)
    #         self.P2P3_ratio_error = None  #(LBZ)

    ##############################################
    # RESULTS
    #############################################

    def show_peak_results(self):
        peak_results = pd.DataFrame(
            data={"P1": [0, 0, 0, 0, 0, 0, 0]},
            index=[
                "Significance",
                "Nex",
                "Nex_error",
                "Number",
                "noff",
                "sign_t_ratio",
                "s/n ratio",
            ],
        )
        for key, value in self.dic.items():
            #if value is not None and hasattr(value, 'sign'):  #(TEST) Check if 'sign' attribute exists before accessing it
            if value is not None:  #(ISSUE ? )
                # print('key:',key)
                # print('value:',value)
                peak_results[key] = [

                    value.sign,
                    value.Nex,
                    value.yerr,
                    value.number,
                    value.noff,
                    value.sign_ratio,
                    value.s_n_ratio,
                ]
                print('value.sign',value.sign)
                print('value.Nex',value.Nex)

        return peak_results


class PulsarPeak:
    """
    A class to manipulate and store information about the statistics of certain regions in a phase list object. 2 types of regions:
        1. Signal region: peaks present in the lightcurve
        2. Background region: background statistics


    Parameters
    ----------
    lc_data : PulsarPhases class object
        PulsarPhases object from which we extract the pulsar phases and the OFF region statistics
    peak_limits : list
        list of phase edges used to define the region.
        To define a non continuous region (for instance, the sum of two independent peaks), a list of four edges can be provided.
    peak_type : str, optional
        'Background' or 'Signal' type to calculate especific statistics.

    Note: To obtain peak statistics an OFF region must be defined in the phase list object first.


    Attributes
    ----------
    limits: str
        region phase edges
    type: str
        the type of region defined (signal or background)
    deltaP: float
        the total phase range of the region
    phases: numpy array
        list of phases that fall into the region
    number: int
        number of events in the region
    nregions: int
        number of phase intervals that define the region


    For 'signal' type only:

    sign: float
        Li&Ma Significance of the excess of events from the peak with respect to the background
    Nex: int
        Number of excess events in the peaks
    yerr: float
        Estimated error of the number of excess events
    sign_ratio: float
        Ratio of the significance and the square root of the time of observation.
    s_n_ratio: float
        Signal to noise ratio
    """

    def __init__(self, peak_limits, peaktype="signal"):
        # Define the features of the region
        self.limits = peak_limits
        self.type = peaktype
        self.deltaP = 0.0
        # self.sign = 0.0 # (LBZ)
        # self.Nex = 0.0 # (LBZ)
        # self.yerr = 0.0 # (LBZ)
        # self.sign_ratio = 0.0 # (LBZ)
        # self.s_n_ratio = 0.0 # (LBZ)
        # self.noff = 0.0 # (LBZ)
        # self.number = 0  # Initialize number attribute to prevent AttributeError
        # self.nregions = 0  # Initialize nregions attribute to prevent AttributeError
        # self.phases = np.array([])  # Initialize phases as empty array
        if len(self.limits) % 2 == 0:
            for i in range(1, len(self.limits), 2):
                self.deltaP = self.deltaP + (self.limits[i] - self.limits[i - 1])

    ##############################################
    # EXECUTION
    #############################################

    def fillPeak(self, phases):
        self.phases = np.array([])
        if len(self.limits) % 2 == 0:
            for i in range(1, len(self.limits), 2):
                self.phases = np.concatenate(
                    [
                        self.phases,
                        phases[
                            (phases > self.limits[i - 1]) & (phases < self.limits[i])
                        ],
                    ]
                )

            self.number = len(self.phases)
            self.nregions = len(self.limits) / 2
        else:
            raise ValueError("Wrong defined regions")

    # Make statistics if the region is signal type only
    def make_stats(self, regions, tobs):
        print('type:',self.type)
        if self.type == "signal":

            # (LBZ) Enhanced version with error handling
            try:  # (LBZ)
                stats, yerror, noff = calculate_CountStats(
                    self.phases,
                    off_file=regions.OFF.phases,
                    factor=(self.deltaP) / regions.OFF.deltaP,
                )
                #print('stats : ',stats)
                #print('yerror: ', yerror)
                #print('noff:',noff)
                self.sign = stats.sqrt_ts.item()
                self.Nex = stats.n_sig
                self.yerr = yerror
                self.sign_ratio = self.sign / np.sqrt(tobs)
                self.s_n_ratio = self.Nex / np.sqrt(noff)
                self.noff = noff
            except Exception as e: # (LBZ)
                # Handle cases where stats cannot be calculated (e.g., too few events, empty regions)
                logger.warning(f"Could not calculate statistics for {getattr(self, 'name', 'unknown peak')}: {e}. Putting all the values at zero.")
                # Set default values to prevent AttributeError
                # self.sign = 0.0 # (LBZ)
                # self.Nex = 0.0 # (LBZ)
                # self.yerr = 0.0 # (LBZ)
                # self.sign_ratio = 0.0 # (LBZ)
                # self.s_n_ratio = 0.0 # (LBZ)
                # self.noff = 0.0 # (LBZ) 

        else:
            print("Cannot calculate statistics for a background region")
