import pandas as pd
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from lstchain.reco.utils import get_effective_time, add_delta_t_key
from lstchain.io.io import (
    dl2_params_lstcam_key,
    get_srcdep_params,
)
import os
from gammapy.data import DataStore
from gammapy.utils.regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
import logging
from regions import PointSkyRegion

logger = logging.getLogger(__name__)

#logging.getLogger("gammapy").disabled = True

__all__ = [
    "compute_theta2",
    "ReadFermiFile",
    "ReadDL3File",
    "ReadLSTFile",
    "ReadtxtFile",
    "ReadList"
]


def compute_theta2(reco_src_x, reco_src_y, src_x, src_y):
    coma_correction = 1.0466
    nominal_focal_length = 28 * coma_correction
    # src_x *= coma_correction
    # src_y *= coma_correction
    theta_meters = np.sqrt(
        np.power(reco_src_x - src_x, 2) + np.power(reco_src_y - src_y, 2)
    )
    theta = np.rad2deg(np.arctan2(theta_meters, nominal_focal_length))
    return np.power(theta, 2)


class ReadFermiFile:
    def __init__(self, file, date_cuts=None): # (LBZ) pas sûre de l'utilité là (ajout de date_cuts=None)
        if "fits" not in file:
            raise ValueError("No FITS file provided for Fermi-LAT data")
        else:
            self.fname = file
        self.date_cuts = date_cuts # (LBZ) pas sûre de l'utilité là 

    def read_file(self):
        f = fits.open(self.fname)
        fits_table = f[1].data
        return fits_table

    def create_df_from_info(self, fits_table):
        time = fits_table["BARYCENTRIC_TIME"].byteswap().newbyteorder()
        phases = fits_table["PULSE_PHASE"].byteswap().newbyteorder()
        energies = fits_table["ENERGY"].byteswap().newbyteorder()
        dataframe = pd.DataFrame(
            {
                "mjd_time": time,
                "pulsar_phase": phases,
                "dragon_time": time * 3600 * 24,
                "energy": energies / 1e6,
            }
        )
        dataframe = dataframe.sort_values(by=["mjd_time"])
        
        ############################################################################### (LBZ)
        # Apply date cuts PRE-FILTER at event level (for memory efficiency)
        # NOTE: Fermi data is a single file without observation-level metadata (unlike DL3)
        # So date filtering is applied at event level in create_df_from_info
        # (consistent with LST architecture, unlike DL3 which filters at __init__)
        # date_cuts are in Unix timestamp (seconds since 1970)
        # mjd_time is in Modified Julian Date (days), needs conversion
        if self.date_cuts is not None:
            from astropy.time import Time
            # Convert mjd_time (days) to Unix timestamps for comparison
            mjd_unix = Time(dataframe["mjd_time"], format='mjd', scale='utc').unix
            len_before = len(dataframe)
            
            if isinstance(self.date_cuts[0], (float, int)):
                dataframe = dataframe[mjd_unix > self.date_cuts[0]]
            if isinstance(self.date_cuts[1], (float, int)):
                dataframe = dataframe[mjd_unix < self.date_cuts[1]]
            len_after = len(dataframe)
            logger.info(f"Fermi date pre-filter: {len_before} → {len_after} events")
        ############################################################################## (LBZ) 
        self.info = dataframe
        return self.info

    def calculate_tobs(self):
        diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
            self.info["mjd_time"].to_list()[0:-1]
        )
        diff[diff > 5 / 24] = 0
        return sum(diff) * 24

    def run(self):
        logger.info("Reading Fermi-LAT data file")
        ftable = self.read_file()
        self.create_df_from_info(ftable)
        self.tobs = self.calculate_tobs()
        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h" + "\n")


class ReadDL3File:
    def __init__(
        self,
        directory=None,
        target_radec=None,
        max_rad=0.1,
        zd_cuts=[0, 60], 
        date_cuts=None, #(LBZ)
        energy_dependent_theta=True,
    ):
        if directory is not None:
            self.direc = directory
            self.datastore = DataStore.from_dir(self.direc)

        self.target_radec = target_radec
        self.energydep_radmax = energy_dependent_theta
        self.date_cuts = date_cuts #(LBZ)
        self.info = None

        # ZENITH ANGLE cut at observation level (like zd_cuts)
        d_zen_max = [self.datastore.obs_table["ZEN_PNT"] < zd_cuts[1]]
        d_zen_min = [self.datastore.obs_table["ZEN_PNT"] > zd_cuts[0]]
        logger.info(f"Zenith angle range cut applied: [{zd_cuts[0]},{zd_cuts[1]}]") #(LBZ)
        

        ########################################################################### (LBZ)
        # DATE cut at observation level (for consistency with zd_cuts)
        obs_mask = d_zen_max[0] * d_zen_min[0] 
        logger.info(f"Zenith angle cut applied at reading files level: {obs_mask.sum()}/{len(self.datastore.obs_table)} runs remaining")

        
        if self.date_cuts is not None:
            # Convert date_cuts (Unix timestamp) to observation date range
            # DL3 obs_table typically has DATE-OBS or similar columns
            try:
                # Try to get observation date boundaries
                if "DATE-OBS" in self.datastore.obs_table.colnames:
                    from astropy.time import Time
                    obs_dates = Time(self.datastore.obs_table["DATE-OBS"], scale='utc').unix
                    d_date_min = [obs_dates > self.date_cuts[0]]
                    d_date_max = [obs_dates < self.date_cuts[1]]
                    obs_mask = obs_mask * d_date_min[0] * d_date_max[0]
                    logger.info(f"Date cut applied at reading files level: {obs_mask.sum()}/{len(self.datastore.obs_table)} runs remaining")
            except (KeyError, Exception) as e:
                logger.warning(f"Could not apply date cut at reading files level: {e}. Will apply at event level instead.")

        self.ids = self.datastore.obs_table[obs_mask]["OBS_ID"]
        self.zd_mask = obs_mask
        ################################################################################# (LBZ) 

        #self.ids = self.datastore.obs_table[d_zen_max[0] * d_zen_min[0]]["OBS_ID"]
        #self.zd_mask = d_zen_max[0] * d_zen_min[0]

        if not self.energydep_radmax:
            self.max_rad = max_rad

    def read_all_DL3file(self):
        obs = self.datastore.get_observations(self.ids, required_irf="point-like")
        # obs = self.datastore.get_observations(self.ids, required_irf=None)
        pos_target = SkyCoord(
            ra=self.target_radec[0] * u.deg,
            dec=self.target_radec[1] * u.deg,
            frame="icrs",
        )

        if not self.energydep_radmax:
            on_radius = self.max_rad * u.deg
            self.on_region = CircleSkyRegion(pos_target, on_radius)
        else:
            self.on_region = PointSkyRegion(pos_target)

        return obs

    def read_DL3file(self, obs_id):
        obs = self.datastore.get_observations([obs_id], required_irf="point-like")
        pos_target = SkyCoord(
            ra=self.target_radec[0] * u.deg,
            dec=self.target_radec[1] * u.deg,
            frame="icrs",
        )
        self.obs_object = obs

        if not self.energydep_radmax:
            on_radius = self.max_rad * u.deg
            self.on_region = CircleSkyRegion(pos_target, on_radius)
            self.events = obs[0].events.select_region(self.on_region).table
        else:
            self.on_region = PointSkyRegion(pos_target)
            self.events = (
                obs[0].events.select_rad_max(obs[0].rad_max, position=pos_target).table
            )

        info = self.create_dataframe()
        return info

    def calculate_tobs(self, mask=None):
        if mask is None:
            mask = self.zd_mask
            #mask = self.obs_mask # (LBZ)
        return self.datastore.obs_table[mask]["LIVETIME"].data.sum() / 3600

    def create_dataframe(self):
        df = self.events.to_pandas()
        df = df.sort_values("TIME")

        lst = Time("2018-10-01", scale="utc")
        time_orig = df["TIME"]

        time = time_orig + lst.to_value(format="unix")
        # timelist = list(Time(time, format="unix").to_value("mjd"))

        info = pd.DataFrame(
            {
                "gammaness": df["GAMMANESS"].to_list(),
                "mjd_time": df["BARYCENT_TIME"].to_list(),
                "orig_time": time_orig.to_list(),
                "dragon_time": list(time),
                "energy": df["ENERGY"].to_list(),
                "pulsar_phase": df["PHASE"].to_list(),
            }
        )

        info = add_delta_t_key(info)
        
        ######################################################################## (LBZ)
        # Apply date cuts POST-FILTER at event level (safety check)
        # NOTE: For DL3, date cut was ALREADY applied at observation level in __init__
        # This is a secondary verification at event level for extra safety
        # (consistency with event-level filtering in Fermi and LST)
        if self.date_cuts is not None:
            len_before = len(info)
            if isinstance(self.date_cuts[0], (float, int)):
                info = info[info["dragon_time"] > self.date_cuts[0]]
            if isinstance(self.date_cuts[1], (float, int)):
                info = info[info["dragon_time"] < self.date_cuts[1]]
            len_after = len(info)
            logger.info(f"DL3 date post-filter safety check at the events level: {len_before} → {len_after} events")
        ######################################################################### (LBZ) 

        return info

    def run(self, pulsarana):
        logger.info("Reading DL3 data files")
        info_list = []
        for obs_id in self.ids:
            logger.info("Reading run number " + str(obs_id))
            try:
                info_file = self.read_DL3file(obs_id)
                info_list.append(info_file)
            except AttributeError:
                raise ValueError("Failing when reading:" + str(obs_id))

        self.info = pd.concat(info_list)
        self.tobs = self.calculate_tobs()


class ReadLSTFile:
    #def __init__(self, file=None, directory=None, src_dependent=False):
    def __init__(self, file=None, directory=None, src_dependent=False, date_cuts=None): # LBZ (pas sûre de l'utilité )
        if file is None and directory is None:
            raise ValueError("No file provided")
        elif file is not None and directory is not None:
            raise ValueError("Can only provide file or directory, but not both")
        elif file is not None:
            if "h5" not in file:
                raise ValueError("No hdf5 file provided for LST data")
            else:
                self.fname = file

        elif directory is not None:
            self.direc = directory
            self.fname = []
            for x in os.listdir(self.direc):
                rel_dir = os.path.relpath(self.direc)
                rel_file = os.path.join(rel_dir, x)
                if "h5" in rel_file:
                    self.fname.append(rel_file)
                    self.fname.sort()

        self.info = None
        self.src_dependent = src_dependent
        self.date_cuts = date_cuts # LBZ (pas sûre de l'utilité)

    def add_phases(self, pname):
        dphase = pd.read_hdf(pname, key=dl2_params_lstcam_key)
        self.info["pulsar_phase"] = dphase["pulsar_phase"]

    def read_LSTfile(self, fname, df_type="short", filter_data=True):
        if filter_data:
            if not self.src_dependent:
                df_or = pd.read_hdf(fname, key=dl2_params_lstcam_key)
                if "pulsar_phase" not in df_or:
                    df_pulsar = pd.read_hdf(fname, key="phase_info")
                    df_or["pulsar_phase"] = df_pulsar["pulsar_phase"]
                    df_or["mjd_time"] = df_pulsar["mjd_barycenter_time"]

                if "event_type" in df_or.columns:
                    df = df_or[df_or["event_type"] == 32]

                    if "theta2" not in df.columns:
                        try:
                            df_pos = pd.read_hdf(fname, "source_position")
                            df_pos = df_pos[df_or["event_type"] == 32]
                            if "theta2" in df_pos.columns:
                                logger.info(
                                    "Including theta2 column from source position table"
                                )
                                df["theta2"] = df_pos["theta2"]
                            if "theta2_on" in df_pos.columns:
                                logger.info(
                                    "Including theta2 column from source position table"
                                )
                                df["theta2"] = df_pos["theta2_on"]
                            else:
                                df["theta2"] = compute_theta2(
                                    np.array(df["reco_src_x"]),
                                    np.array(df["reco_src_y"]),
                                    np.array(df_pos["src_x"]),
                                    np.array(df_pos["src_y"]),
                                )
                        except AttributeError:
                            logger.info("No theta2 computed")
                else:
                    df = df_or

            elif self.src_dependent:
                srcindep_df = pd.read_hdf(
                    fname, key=dl2_params_lstcam_key, float_precision=20
                )
                on_df_srcdep = get_srcdep_params(fname, "on")

                if "pulsar_phase" not in srcindep_df:
                    df_pulsar = pd.read_hdf(fname, key="phase_info")
                    srcindep_df["pulsar_phase"] = df_pulsar["pulsar_phase"]
                    srcindep_df["mjd_time"] = df_pulsar["mjd_barycenter_time"]

                if "reco_energy" in srcindep_df.keys():
                    srcindep_df.drop(["reco_energy"])

                if "gammaness" in srcindep_df.keys():
                    srcindep_df.drop(["gammaness"])

                df = pd.concat([srcindep_df, on_df_srcdep], axis=1)
                df = df[df.event_type == 32]

            if df_type == "short":
                if "alpha" in df and "theta2" in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alpha",
                            "theta2",
                            "alt_tel",
                        ]
                    ]
                elif "alpha" in df and "theta2" not in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alpha",
                            "alt_tel",
                        ]
                    ]
                elif "theta2" in df and "alpha" not in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "theta2",
                            "alt_tel",
                        ]
                    ]
                else:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alt_tel",
                        ]
                    ]

                try:
                    df_filtered["energy"] = df["reco_energy"]
                except AttributeError:
                    df_filtered["energy"] = df["energy"]
            else:
                df_filtered = df
                df_filtered["energy"] = df["reco_energy"]

            df_filtered = add_delta_t_key(df_filtered)

        else:
            df_filtered = pd.read_hdf(fname, key=dl2_params_lstcam_key)
        
        ############################################################################# (LBZ) 
        # Apply date cuts PRE-FILTER at event level (for memory efficiency)
        # NOTE: LST data is HDF5 file(s) without observation-level metadata (unlike DL3)
        # So date filtering is applied at event level in read_LSTfile
        # (consistent with Fermi architecture, unlike DL3 which filters at __init__)
        # date_cuts are in Unix timestamp (seconds since 1970)
        # dragon_time for LST is already in Unix timestamp format (native column)
        if self.date_cuts is not None:
            len_before = len(df_filtered)
            if isinstance(self.date_cuts[0], (float, int)):
                df_filtered = df_filtered[df_filtered["dragon_time"] > self.date_cuts[0]]
            if isinstance(self.date_cuts[1], (float, int)):
                df_filtered = df_filtered[df_filtered["dragon_time"] < self.date_cuts[1]]
            len_after = len(df_filtered)
            logger.info(f"LST date pre-filter: {len_before} → {len_after} events")
        ############################################################################### (LBZ) 

        return df_filtered

    def calculate_tobs(self):
        dataframe = add_delta_t_key(self.info)
        return get_effective_time(dataframe)[1].value / 3600

    def save_memory(self):
        for column in self.info.columns:
            if (
                column == "dragon_time"
                or column == "delta_t"
                or column == "mjd_barycenter_time"
            ):
                continue
            if (self.info[column].dtype != "float64") & (
                self.info[column].dtype != "float128"
            ):
                continue
            self.info[column] = self.info[column].astype("float32")

    def run(self, pulsarana, df_type="short"):
        filter_data = pulsarana.filter_data
        logger.info("Reading LST-1 data file")
        if isinstance(self.fname, list):
            info_list = []
            for name in self.fname:
                logger.info("Reading file: " + name)
                try:
                    info_file = self.read_LSTfile(name, df_type, filter_data)
                    self.info = info_file
                    self.tobs = self.calculate_tobs()

                    if filter_data:
                        pulsarana.cuts.apply_fixed_cut(self)
                        if pulsarana.cuts.energy_binning_cut is not None:
                            pulsarana.cuts.apply_energydep_cuts(self)

                    self.save_memory()
                    info_list.append(self.info)
                except AttributeError:
                    raise ValueError("Failing when reading:" + str(name))

            self.info = pd.DataFrame()
            while len(info_list) > 0:
                if len(info_list) >= 10:
                    chunk_df = pd.concat(info_list[0:10])
                    info_list = info_list[10:]
                else:
                    chunk_df = pd.concat(info_list)
                    info_list = []

                self.info = pd.concat([self.info, chunk_df])
            self.tobs = self.calculate_tobs()

        else:
            self.info = self.read_LSTfile(self.fname, df_type, filter_data)
            self.tobs = self.calculate_tobs()

            if filter_data:
                pulsarana.cuts.apply_fixed_cut(self)

                if pulsarana.cuts.energy_binning_cut is not None:
                    pulsarana.cuts.apply_energydep_cuts(self)

        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h")


class ReadtxtFile:
    def __init__(self, file, format_txt):
        self.fname = file
        self.format = format_txt

    def read_file(self):
        data = pd.read_csv(self.fname, sep=" ", header=None)
        return data

    def check_format(self):
        for name in ["t", "p"]:
            if name not in self.format:
                raise ValueError("   No valid format")

    def create_df_from_info(self, df):
        for i in range(0, len(self.format)):
            if self.format[i] == "t":
                times = df.iloc[:, i]
            elif self.format[i] == "e":
                energies = df.iloc[:, i]
            elif self.format[i] == "p":
                phases = df.iloc[:, i]
            elif self.format[i] == "g":
                gammaness = df.iloc[:, i]
            elif self.format[i] == "a":
                alpha = df.iloc[:, i]
            elif self.format[i] == "t2":
                theta2 = df.iloc[:, i]
            elif self.format[i] == "at":
                alt_tel = df.iloc[:, i]

        dataframe = pd.DataFrame(
            {
                "mjd_time": times,
                "pulsar_phase": phases,
                "dragon_time": times * 3600 * 24,
                "energy": energies,
            }
        )

        try:
            dataframe["gammaness"] = gammaness
        except AttributeError:
            pass

        try:
            dataframe["alpha"] = alpha
        except AttributeError:
            pass

        try:
            dataframe["theta2"] = theta2
        except AttributeError:
            pass

        try:
            dataframe["alt_tel"] = alt_tel
        except AttributeError:
            pass

        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe

    def calculate_tobs(self):
        diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
            self.info["mjd_time"].to_list()[0:-1]
        )
        return sum(diff) * 24

    def run(self):
        data = self.read_file()
        self.check_format()
        self.create_df_from_info(data)
        self.tobs = self.calculate_tobs()

        logger.info(
            "    Finishing reading. Total time is " + str(self.tobs) + " s" + "\n"
        )


class ReadList:
    def __init__(self, phases_list, time_list=None, energy_list=None, tel="LST"):
        self.plist = phases_list
        self.tlist = time_list
        self.elist = energy_list
        self.tel = tel

    def create_df_from_info(self):
        dataframe = pd.DataFrame(
            {
                "mjd_time": self.tlist,
                "pulsar_phase": self.plist,
                "dragon_time": self.tlist * 3600 * 24,
                "energy": self.elist,
            }
        )
        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe

    def calculate_tobs(self):
        if self.tel == "LST" or self.tel == "MAGIC":
            dataframe = add_delta_t_key(self.info)
            return get_effective_time(dataframe)[1].value / 3600

        elif self.tel == "fermi":
            diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
                self.info["mjd_time"].to_list()[0:-1]
            )
            diff[diff > 5 / 24] = 0
            return sum(diff) * 24

    def run(self):
        self.create_df_from_info()
        self.tobs = self.calculate_tobs()
        logger.info(
            "    Finishing reading. Total time is " + str(self.tobs) + " s" + "\n"
        )
