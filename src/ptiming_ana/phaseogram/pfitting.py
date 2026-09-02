import numpy as np
import logging
from scipy.optimize import curve_fit
from iminuit import Minuit, cost
import pandas as pd
from .models import (
    get_model_list,
    gaussian,
    double_gaussian,
    double_gaussian_heaviside,#(LBZ)
    triple_gaussian,
    assymetric_double_gaussian,
    double_lorentz,
    lorentzian,
    step_function, #(LBZ)
    dgaussian_step, #(LBZ)
)
from more_itertools import sort_together

__all__ = ["PeakFitting"]


logger = logging.getLogger(__name__) #(LBZ)


class PeakFitting:
    def __init__(self, binned, model, peak="both"):
        # Define and check model
        self.model = model
        self.shift = 0
        self.peak_tofit = peak
        self.check_model()
        self.binned = binned

    ##############################################
    # EXECUTION
    #############################################

    def run(self, pulsar_phases):
        # Estimate initial values
        self.est_initial_values(pulsar_phases)
        # Do the fitting
        if self.binned:
            self.fit_Binned(pulsar_phases)
        else:
            self.fit_ULmodel(pulsar_phases)

    def check_model(self):
        model_list = get_model_list()
        if self.model not in model_list:
            raise ValueError("The model is not in the available model list")

        if self.peak_tofit == "both" and self.model == "gaussian":
            raise ValueError("Gaussian model can only fit one peak")

        if self.model == "dgaussian_heaviside" and self.peak_tofit != "both": #(LBZ)
            raise ValueError("Heaviside model needs all three peaks fitted together") #(LBZ)
        
        if self.model == "dgaussian_step" and self.peak_tofit != "both": #(LBZ)
            raise ValueError("Gaussian_step model needs all three peaks fitted together") #(LBZ)

        if self.peak_tofit == "P1" and self.model == "dgaussian":
            raise ValueError("Dgaussian model needs two peaks")

        if self.peak_tofit == "P2" and self.model == "dgaussian":
            raise ValueError("Gaussian model needs two peaks")

    def est_initial_values(self, pulsar_phases):
        self.check_model()
        self.init = []
        intensity = []
        height = []

        if self.model == "dgaussian_heaviside": #(LBZ)
            for name in ["P1", "P2"]: #(LBZ)
                P_info = pulsar_phases.regions.dic[name] #(LBZ)
                if P_info is None: #(LBZ)
                    raise ValueError("Heaviside model needs P1 and P2 peaks") #(LBZ)

                intensity.append(P_info.Nex / P_info.noff) #(LBZ)
                height.append(P_info.Nex) #(LBZ)
                self.shift = ( #(LBZ)
                    pulsar_phases.regions.OFF.limits[1] #(LBZ)
                    + pulsar_phases.regions.OFF.limits[0] #(LBZ)
                ) / 2 #(LBZ)

                if len(P_info.limits) > 2: #(LBZ)
                    extension = (P_info.limits[0] + 1 + P_info.limits[3]) / 2 #(LBZ)
                else: #(LBZ)
                    extension = (P_info.limits[0] + P_info.limits[1]) / 2 #(LBZ)

                if extension < self.shift: #(LBZ)
                    extension = extension + 1 #(LBZ)

                self.init.extend([extension, P_info.deltaP / 2]) #(LBZ)

            P3_info = pulsar_phases.regions.dic["P3"] #(LBZ)
            if P3_info is None:
                raise ValueError("Heaviside model needs P3 peak") #(LBZ)

            p3_threshold = (P3_info.limits[0] + P3_info.limits[1]) / 2 #(LBZ)
            if p3_threshold < self.shift: #(LBZ)
                p3_threshold = p3_threshold + 1 #(LBZ)

            self.init.append(p3_threshold) #(LBZ)
            height.append(P3_info.Nex) #(LBZ)

            bkg = np.mean( #(LBZ)
                (
                    pulsar_phases.histogram.lc[0][ #(LBZ)
                        (
                            pulsar_phases.histogram.lc[1][:-1] #(LBZ)
                            > (pulsar_phases.regions.OFF.limits[0]) #(LBZ)
                        )
                        & (
                            pulsar_phases.histogram.lc[1][1:] #(LBZ)
                            < pulsar_phases.regions.OFF.limits[1] #(LBZ)
                        )
                    ]
                )
            )

            self.init.extend(height) #(LBZ)
            self.init.append(bkg) #(LBZ)
            return #(LBZ)

        if self.model == "dgaussian_step": #(LBZ)
            # P1 and P2 as Gaussians, P3 as step rectangle

            
            heights = []  # Store heights separately
            for name in ["P1", "P2"]: #(LBZ)
                P_info = pulsar_phases.regions.dic[name] #(LBZ)
                if P_info is None: #(LBZ)
                    raise ValueError("dgaussian_step model needs P1 and P2 peaks") #(LBZ)

                self.shift = ( #(LBZ)
                    pulsar_phases.regions.OFF.limits[1] #(LBZ)
                    + pulsar_phases.regions.OFF.limits[0] #(LBZ)
                ) / 2 #(LBZ)

                if len(P_info.limits) > 2: #(LBZ)
                    extension = (P_info.limits[0] + 1 + P_info.limits[3]) / 2 #(LBZ)
                else: #(LBZ)
                    extension = (P_info.limits[0] + P_info.limits[1]) / 2 #(LBZ)

                if extension < self.shift: #(LBZ)
                    extension = extension + 1 #(LBZ)

                self.init.extend([extension, P_info.deltaP / 2]) #(LBZ) mu, sigma
                heights.append(P_info.Nex) #(LBZ) collect height separately
            
            # P3 as step rectangle
            P3_info = pulsar_phases.regions.dic["P3"] #(LBZ)
            if P3_info is None: #(LBZ)
                raise ValueError("dgaussian_step model needs P3 peak") #(LBZ)
            
            phi1 = P3_info.limits[0] #(LBZ) left edge of P3
            phi2 = P3_info.limits[1] #(LBZ) right edge of P3
            self.init.extend([phi1, phi2]) #(LBZ) append phi1, phi2 after position parameters
            self.init.extend(heights) #(LBZ) append heights: B, C
            #self.init.append(P3_info.Nex) #(LBZ) append D (P3 height)
            self.init.append(P3_info.number)
            
            bkg = np.mean( #(LBZ)
                (
                    pulsar_phases.histogram.lc[0][ #(LBZ)
                        (
                            pulsar_phases.histogram.lc[1][:-1] #(LBZ)
                            > (pulsar_phases.regions.OFF.limits[0]) #(LBZ)
                        )
                        & (
                            pulsar_phases.histogram.lc[1][1:] #(LBZ)
                            < pulsar_phases.regions.OFF.limits[1] #(LBZ)
                        )
                    ]
                )
            )
            
            self.init.append(bkg) #(LBZ) append baseline (A) at the end
            return #(LBZ)

        # Set different initial values for different models
        if self.model == "tgaussian":
            regions_names = ["P1", "P2", "P3"]
        elif self.model == "gaussian":
            regions_names = ["P2"]
        else:
            regions_names = ["P1", "P2"]

        for name in regions_names:
            P_info = pulsar_phases.regions.dic[name]
            if P_info is not None:
                if name == self.peak_tofit or self.peak_tofit == "both":
                    intensity.append(P_info.Nex / P_info.noff)
                    height.append(P_info.Nex)
                    self.shift = (
                        pulsar_phases.regions.OFF.limits[1]
                        + pulsar_phases.regions.OFF.limits[0]
                    ) / 2

                    if len(P_info.limits) > 2:
                        extension = (P_info.limits[0] + 1 + P_info.limits[3]) / 2
                    else:
                        extension = (P_info.limits[0] + P_info.limits[1]) / 2

                    if extension < self.shift:
                        extension = extension + 1

                    self.init.extend([extension, P_info.deltaP / 2])

                    if self.model == "asym_dgaussian":
                        self.init.append(P_info.deltaP / 2)
            else:
                if self.model != "gaussian" and self.model != "lorentzian":
                    raise ValueError("Double Gaussian model needs two peaks")

        bkg = np.mean(
            (
                pulsar_phases.histogram.lc[0][
                    (
                        pulsar_phases.histogram.lc[1][:-1]
                        > (pulsar_phases.regions.OFF.limits[0])
                    )
                    & (
                        pulsar_phases.histogram.lc[1][1:]
                        < pulsar_phases.regions.OFF.limits[1]
                    )
                ]
            )
        )

        self.init.extend(height)
        self.init.append(bkg)

    # Unbinned fitting
    def fit_ULmodel(self, pulsar_phases):
        self.check_model()

        # Shift the phases if one of the peak is near the interval edge
        shift_phases = pulsar_phases.phases
        if self.shift != 0:
            for i in range(0, len(shift_phases)):
                if shift_phases[i] < self.shift:
                    shift_phases[i] = shift_phases[i] + 1

        if self.model == "dgaussian":
            unbinned_likelihood = cost.UnbinnedNLL(
                np.array(shift_phases), double_gaussian
            )  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                mu=self.init[0],
                sigma=self.init[1],
                mu_2=self.init[2],
                sigma_2=self.init[3],
                A=self.init[4],
                B=self.init[5],
                C=self.init[6],
            )

            self.parnames = ["mu", "sigma", "mu_2", "sigma_2", "A", "B", "C"]
            for par in ["mu", "sigma", "mu_2", "sigma_2", "B", "C"]:
                minuit.fixed[par] = False
            minuit.fixed["A"] = True

        elif self.model == "dgaussian_heaviside": #(LBZ)
            def custom_dgaussian_heaviside( #(LBZ)
                x, mu, sigma, mu_2, sigma_2, mu_3, B, C, D #(LBZ)
            ):
                return double_gaussian_heaviside( #(LBZ)
                    x, #(LBZ)
                    mu, #(LBZ)
                    sigma, #(LBZ)
                    mu_2, #(LBZ)
                    sigma_2, #(LBZ)
                    mu_3,  #(LBZ)
                    self.init[-1], #(LBZ)
                    B, #(LBZ)
                    C, #(LBZ)
                    D, #(LBZ)
                )

            unbinned_likelihood = cost.UnbinnedNLL( #(LBZ)
                np.array(shift_phases), custom_dgaussian_heaviside #(LBZ)
            )  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit( #(LBZ)
                unbinned_likelihood, #(LBZ)
                mu=self.init[0], #(LBZ)
                sigma=self.init[1], #(LBZ)
                mu_2=self.init[2], #(LBZ)
                sigma_2=self.init[3], #(LBZ)
                mu_3=self.init[4], #(LBZ)
                B=self.init[5], #(LBZ)
                C=self.init[6], #(LBZ)
                D=self.init[7], #(LBZ)
            )

            self.parnames = [ #(LBZ)
                "mu", #(LBZ)
                "sigma", #(LBZ)
                "mu_2", #(LBZ)
                "sigma_2", #(LBZ)
                "mu_3", #(LBZ)
                "A", #(LBZ)
                "B", #(LBZ)
                "C", #(LBZ)
                "D", #(LBZ)
            ]
            for par in ["mu", "sigma", "mu_2", "sigma_2", "mu_3", "B", "C", "D"]: #(LBZ)
                minuit.fixed[par] = False
            minuit.fixed["A"] = True

        if self.model == "tgaussian":
            unbinned_likelihood = cost.UnbinnedNLL(
                np.array(shift_phases), triple_gaussian
            )  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                Bkg=self.init[-1],
                mu=self.init[0],
                sigma=self.init[1],
                mu_2=self.init[2],
                sigma_2=self.init[3],
                mu_3=self.init[4],
                sigma_3=self.init[5],
                A=self.init[6],
                B=self.init[7],
                C=self.init[8],
            )

            self.parnames = [
                "Bkg",
                "mu",
                "sigma",
                "mu_2",
                "sigma_2",
                "mu_3",
                "sigma_3",
                "A",
                "B",
                "C",
            ]
            for par in self.parnames:
                minuit.fixed[par] = False

            minuit.fixed["Bkg"] = True

        elif self.model == "asym_dgaussian":
            unbinned_likelihood = cost.UnbinnedNLL(
                np.array(shift_phases), assymetric_double_gaussian
            )  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                mu=self.init[0],
                sigma1=self.init[1],
                sigma2=self.init[2],
                mu_2=self.init[3],
                sigma1_2=self.init[4],
                sigma2_2=self.init[5],
                A=self.init[6],
                B=self.init[7],
                C=self.init[8],
            )
            self.parnames = [
                "mu",
                "sigma1",
                "sigma2",
                "mu_2",
                "sigma1_2",
                "sigma2_2",
                "A",
                "B",
                "C",
            ]

        elif self.model == "double_lorentz":
            unbinned_likelihood = cost.UnbinnedNLL(
                np.array(shift_phases), double_lorentz
            )  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                mu_1=self.init[0],
                gamma_1=self.init[1],
                mu_2=self.init[2],
                gamma_2=self.init[3],
                A=self.init[4],
                B=self.init[5],
                C=self.init[6],
            )
            self.parnames = ["mu_1", "gamma_1", "mu_2", "gamma_2", "A", "B", "C"]

        elif self.model == "lorentzian":
            unbinned_likelihood = cost.UnbinnedNLL(np.array(shift_phases), lorentzian)  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                mu_1=self.init[0],
                gamma_1=self.init[1],
                A=self.init[4],
                B=self.init[5],
            )
            self.parnames = ["mu_1", "gamma_1", "A", "B"]

        elif self.model == "gaussian":
            unbinned_likelihood = cost.UnbinnedNLL(np.array(shift_phases), gaussian)  #(LBZ) swapped argument order: (data, model)
            minuit = Minuit(
                unbinned_likelihood,
                mu=self.init[0],
                sigma=self.init[1],
                A=self.init[2],
                B=self.init[3],
            )
            self.parnames = ["mu", "sigma", "A", "B"]

        elif self.model == "dgaussian_step": #(LBZ)
            def custom_dgaussian_step(x, mu, sigma, mu_2, sigma_2, phi1, phi2, B, C, D): #(LBZ)
                return dgaussian_step(x, mu, sigma, mu_2, sigma_2, phi1, phi2, self.init[-1], B, C, D) #(LBZ)
            
            unbinned_likelihood = cost.UnbinnedNLL(np.array(shift_phases), custom_dgaussian_step) #(LBZ)
            minuit = Minuit( #(LBZ)
                unbinned_likelihood, #(LBZ)
                mu=self.init[0], sigma=self.init[1], mu_2=self.init[2], sigma_2=self.init[3], #(LBZ)
                phi1=self.init[4], phi2=self.init[5], B=self.init[6], C=self.init[7], D=self.init[8], #(LBZ)
            ) #(LBZ)
            self.parnames = ["mu", "sigma", "mu_2", "sigma_2", "phi1", "phi2", "A", "B", "C", "D"] #(LBZ)
            for par in ["mu", "sigma", "mu_2", "sigma_2", "phi1", "phi2", "B", "C", "D"]: #(LBZ)
                minuit.fixed[par] = False #(LBZ)
            minuit.fixed["A"] = True #(LBZ)

        minuit.errordef = 0.5
        minuit.migrad()

        # Store results as minuit object
        self.minuit = minuit
        self.unbinned_lk = unbinned_likelihood

        # Store the result of params and errors
        self.params = []
        self.errors = []
        for name in self.parnames:
            self.params.append(self.minuit.values[name])
            self.errors.append(self.minuit.errors[name])

        self.create_result_df()

    ####################################################### (LBZ)
    def validate_initial_parameters(self):
        """Check if initial parameters are valid for fitting.
        
        Validates ALL parameters in self.init:
        - No NaN or infinity values
        - Sigma/width parameters > 0
        - Heights/amplitudes > 0
        - Background >= 0
        - Position parameters within reasonable bounds [0, 2]
        
        Returns:
            bool: True if parameters are valid, False if fitting should be skipped
        """
        # self.init correspond to the first parameters given to the fitter 
        if not self.init or len(self.init) == 0:
            logger.warning("No initial parameters available for fitting (sparse data)")
            return False
        
        # **Step 1: Check ALL parameters for NaN/Inf (universal check)**
        for idx, param in enumerate(self.init):
            if np.isnan(param) or np.isinf(param):
                logger.warning(
                    f"Invalid initial parameter at index {idx}: {param} (NaN or Inf). "
                    f"Data too sparse or bad peak detection. Skipping fit for this bin."
                )
                return False
        
        # **Step 2: Check model-specific parameter validity**
        if self.model == "dgaussian":
            # [mu, sigma, mu_2, sigma_2, height1, height2, bkg]
            sigma_indices = [1, 3]
            height_indices = [4, 5]
            mu_indices = [0, 2]
            
        elif self.model == "asym_dgaussian":
            # [mu, sigma1, sigma2, mu_2, sigma1_2, sigma2_2, height1, height2, bkg]
            sigma_indices = [1, 2, 4, 5]
            height_indices = [6, 7]
            mu_indices = [0, 3]
            
        elif self.model == "tgaussian":
            # [mu, sigma, mu_2, sigma_2, mu_3, sigma_3, height1, height2, height3, bkg]
            sigma_indices = [1, 3, 5]
            height_indices = [6, 7, 8]
            mu_indices = [0, 2, 4]
            
        elif self.model == "double_lorentz":
            # [mu_1, gamma_1, mu_2, gamma_2, height, bkg]
            sigma_indices = [1, 3]  # gamma params for lorentzians
            height_indices = [4]
            mu_indices = [0, 2]
            
        elif self.model == "lorentzian":
            # [mu, gamma, height, bkg]
            sigma_indices = [1]
            height_indices = [2]
            mu_indices = [0]
            
        elif self.model == "gaussian":
            # [mu, sigma, height, bkg]
            sigma_indices = [1]
            height_indices = [2]
            mu_indices = [0]

        elif self.model == "dgaussian_heaviside": #(LBZ)
            # [mu, sigma, mu_2, sigma_2, mu_3, height1, height2, height3, bkg]
            sigma_indices = [1, 3] #(LBZ) 
            height_indices = [5, 6, 7] #(LBZ)
            mu_indices = [0, 2, 4] #(LBZ)
        
        elif self.model == "dgaussian_step": #(LBZ)
            # [mu, sigma, mu_2, sigma_2, phi1, phi2, B, C, D, A]
            sigma_indices = [1, 3] #(LBZ)
            height_indices = [6, 7, 8] #(LBZ) B, C, D
            mu_indices = [0, 2, 4, 5] #(LBZ) mu, mu_2, phi1, phi2
        
        else:
            return True  # Model not recognized, skip validation
        ################################ ATTENTION verify if the parameter selections are physically correct and maybe configurable in config file ? #####################
        # Check sigma/width parameters > 0
        for idx in sigma_indices:
            if idx < len(self.init):
                if self.init[idx] <= 0:
                    logger.warning(
                        f"Sigma/width parameter at index {idx}: {self.init[idx]} ≤ 0. "
                        f"Continuing with the fit using this initial guess."
                    )
                    # logger.warning(
                    #     f"Invalid sigma/width parameter at index {idx}: {self.init[idx]} ≤ 0. "
                    #     f"Skipping fit for this bin."
                    # )
                    # return False
                    continue
        
        # Check height/amplitude parameters > 0
        for idx in height_indices:
            if idx < len(self.init):
                if self.init[idx] <= 0:
                    logger.warning(
                        f"Height/amplitude parameter at index {idx}: {self.init[idx]} ≤ 0. "
                        f"Continuing with the fit using this initial guess."
                    )
                    # logger.warning(
                    #     f"Invalid height/amplitude parameter at index {idx}: {self.init[idx]} ≤ 0. "
                    #     f"Skipping fit for this bin."
                    # )
                    # return False
                    continue
        
        # Check position parameters are within phase bounds [0, 1 + shift]
        max_phase = 1 + abs(self.shift)  # Account for shifted phases
        for idx in mu_indices:
            if idx < len(self.init):
                if not (0 <= self.init[idx] <= max_phase):
                    logger.warning(
                        f"Position parameter at index {idx}: {self.init[idx]} (outside [0, {max_phase}]). "
                        f"Continuing with the fit using this initial guess."
                    )
                    # logger.warning(
                    #     f"Invalid position parameter at index {idx}: {self.init[idx]} (outside [0, {max_phase}]). "
                    #     f"Skipping fit for this bin."
                    # )
                    # return False
                    continue
        
        # Check background is non-negative (last parameter is usually bkg)
        if len(self.init) > 0:
            bkg_idx = len(self.init) - 1
            if self.init[bkg_idx] < 0:
                logger.warning(
                    f"Background parameter at index {bkg_idx}: {self.init[bkg_idx]} < 0. "
                    f"Continuing with the fit using this initial guess."
                )
                # logger.warning(
                #     f"Invalid background parameter at index {bkg_idx}: {self.init[bkg_idx]} < 0. "
                #     f"Skipping fit for this bin."
                # )
                # return False
        
        return True

    ####################################################### (LBZ)
    
    # Binned fitting
    def fit_Binned(self, pulsar_phases):
        self.check_model()
        
        # (LBZ) Validate initial parameters before attempting fit
        if not self.validate_initial_parameters():
        
            logger.info("Fit skipped for this energy bin due to invalid initial parameters")
            self.fitting_performed = False
            return  # Skip fitting but continue processing this bin
        
        self.fitting_performed = True
        ###########################################################################
        histogram = pulsar_phases.histogram

        # Shift the phases if one of the peak is near the interval edge
        shift_phases = list(histogram.lc[1][:-1])
        bin_height = list(histogram.lc[0])

        if self.shift != 0:
            for i in range(0, len(shift_phases)):
                if shift_phases[i] < self.shift:
                    shift_phases[i] = shift_phases[i] + 1

        bin_height = np.array(sort_together([shift_phases, bin_height])[1])
        shift_phases.sort()
        shift_phases.append(shift_phases[0] + 1)
        shift_phases = np.array(shift_phases)
        bin_centres = (shift_phases[1:] + shift_phases[0:-1]) / 2

        try: #(LBZ)
            if self.model == "dgaussian":

                def custom_dgaussian(x, mu, sigma, mu_2, sigma_2, B, C):
                    return double_gaussian(x, mu, sigma, mu_2, sigma_2, self.init[-1], B, C)

                params, pcov_l = curve_fit(
                    custom_dgaussian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )
                self.parnames = ["mu", "sigma", "mu_2", "sigma_2", "A", "B", "C"]

            elif self.model == "asym_dgaussian":
                assymetric_double_gaussian_vec = np.vectorize(
                    assymetric_double_gaussian, excluded=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                )

                def custom_adgaussian(
                    x, mu, sigma1, sigma2, mu_2, sigma1_2, sigma2_2, B, C
                ):
                    return assymetric_double_gaussian_vec(
                        x, mu, sigma1, sigma2, mu_2, sigma1_2, sigma2_2, self.init[-1], B, C
                    )

                params, pcov_l = curve_fit(
                    custom_adgaussian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )
                self.parnames = [
                    "mu",
                    "sigma1",
                    "sigma2",
                    "mu_2",
                    "sigma1_2",
                    "sigma2_2",
                    "A",
                    "B",
                    "C",
                ]

            elif self.model == "tgaussian":

                def custom_tgaussian(x, mu, sigma, mu_2, sigma_2, mu_3, sigma_3, B, C, D):
                    return triple_gaussian(
                        x, self.init[-1], mu, sigma, mu_2, sigma_2, mu_3, sigma_3, B, C, D
                    )

                # bounds = ([0,0,0,0,0,0.1,0,0,0],[2,2,2,2,2,2,np.inf,np.inf,np.inf])
                params, pcov_l = curve_fit(
                    custom_tgaussian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )
                self.parnames = [
                    "A",
                    "mu",
                    "sigma",
                    "mu_2",
                    "sigma_2",
                    "mu_3",
                    "sigma_3",
                    "B",
                    "C",
                    "D",
                ]

            elif self.model == "double_lorentz":

                def custom_lorentzian(x, mu_1, gamma_1, mu_2, gamma_2, B, C):
                    return double_lorentz(
                        x, mu_1, gamma_1, mu_2, gamma_2, self.init[-1], B, C
                    )

                params, pcov_l = curve_fit(
                    custom_lorentzian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )

                self.parnames = ["mu_1", "gamma_1", "mu_2", "gamma_2", "A", "B", "C"]

            elif self.model == "lorentzian":

                def custom_lorentzian(x, mu_1, gamma_1, B):
                    return lorentzian(x, mu_1, gamma_1, self.init[-1], B)

                params, pcov_l = curve_fit(
                    custom_lorentzian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )

                self.parnames = ["mu_1", "gamma_1", "A", "B"]

            elif self.model == "gaussian":

                def custom_gaussian(x, mu, sigma, B):
                    return gaussian(x, mu, sigma, self.init[-1], B)

                params, pcov_l = curve_fit(
                    custom_gaussian,
                    bin_centres,
                    bin_height,
                    sigma=np.sqrt(bin_height),
                    p0=self.init[:-1],
                )
                self.parnames = ["mu", "sigma", "A", "B"]

            elif self.model == "dgaussian_heaviside": #(LBZ)
                def custom_dgaussian_heaviside( #(LBZ)
                    x, mu, sigma, mu_2, sigma_2, mu_3, B, C, D #(LBZ)
                ): #(LBZ)
                    return double_gaussian_heaviside( #(LBZ)
                        x, mu, sigma, mu_2, sigma_2, mu_3, self.init[-1], B, C, D #(LBZ)
                    ) #(LBZ)

                params, pcov_l = curve_fit( #(LBZ)
                    custom_dgaussian_heaviside, #(LBZ)
                    bin_centres, bin_height, #(LBZ)
                    sigma=np.sqrt(bin_height), #(LBZ)
                    p0=self.init[:-1], #(LBZ)
                ) #(LBZ)
                self.parnames = ["mu", "sigma", "mu_2", "sigma_2", "mu_3", "A", "B", "C", "D"] #(LBZ)

            elif self.model == "dgaussian_step": #(LBZ)
                def custom_dgaussian_step( #(LBZ)
                    x, mu, sigma, mu_2, sigma_2, phi1, phi2, B, C, D #(LBZ)
                ): #(LBZ)
                    return dgaussian_step( #(LBZ)
                        x, mu, sigma, mu_2, sigma_2, phi1, phi2, self.init[-1], B, C, D #(LBZ)
                    ) #(LBZ)
                
                params, pcov_l = curve_fit( #(LBZ)
                    custom_dgaussian_step, #(LBZ)
                    bin_centres, bin_height, #(LBZ)
                    sigma=np.sqrt(bin_height), #(LBZ)
                    p0=self.init[:-1], #(LBZ) exclude A
                ) #(LBZ)
                self.parnames = ["mu", "sigma", "mu_2", "sigma_2", "phi1", "phi2", "A", "B", "C", "D"] #(LBZ)
        # (LBZ) skip the fit if it is not "correct" 
        except (ZeroDivisionError, ValueError, RuntimeError) as e:
            logger.warning(
                f"Fitting failed for model '{self.model}': {type(e).__name__}: {e}. "
                f"This can happen with very few events or poor initial parameters. "
                f"Skipping fit for this energy bin."
            )
            self.fitting_performed = False
            self.params = None
            return  # Skip rest of fit processing
        ######################################################################################
        
        # Store the result of params and errors
        self.params = []
        self.errors = []
        j = 0
        for i in range(0, len(self.parnames)):
            if self.parnames[i] == "A":
                self.params.append(self.init[-1])
                self.errors.append(0)
                j = 1
            else:
                self.params.append(params[i - j])
                self.errors.append(np.sqrt(pcov_l[i - j][i - j]))

        self.create_result_df()

    ##############################################
    # RESULTS
    #############################################

    def check_fit_result(self):
        try:
            self.params
        except AttributeError:
            return False
        return True

    def create_result_df(self):
        d = {"Name": self.parnames, "Value": self.params, "Error": self.errors}
        self.df_result = pd.DataFrame(data=d)

    def show_result(self):
        try:
            return self.df_result
        except AttributeError:
            print("No fit has been done so far")
