import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

__all__ = ["PulsarTimeAnalysis", "function_sqrt", "function_lin"]


def function_sqrt(x, A):
    return A * np.sqrt(x)


def function_lin(x, A):
    return A * x


class PulsarTimeAnalysis:
    def __init__(self, tint=3600):
        # Define the arrays to store information
        self.Tarray = []
        self.HTime = [0]
        self.ZTime = [0]
        self.ChiTime = [0]
        self.P1sTime = [0]
        self.P1exTime = [0]
        self.P1exerror = [0]
        self.P2sTime = [0]
        self.P2exTime = [0]
        self.P2exerror = [0]
        self.P3sTime = [0]  #(LBZ) Add P3 support
        self.P3exTime = [0]  #(LBZ)
        self.P3exerror = [0]  #(LBZ)
        self.P1P2sTime = [0]
        self.P1P2exTime = [0]
        self.P1P2exerror = [0]

        # MJD date (end of bin) corresponding to each point of the time evolution
        # Aligned with self.t: bin_dates[0] = None (initial value)
        self.bin_dates = [None]

        # Define the time interval
        self.tint = tint

    ##############################################
    # EXECUTION
    #############################################

    def store_Tvalues(self, pulsar_phases):
        # Store periodicity tests stats
        self.HTime.append(pulsar_phases.stats.Htest_res[2])
        self.ZTime.append(pulsar_phases.stats.Zntest_res[2])
        self.ChiTime.append(pulsar_phases.stats.chisqr_res[2])

        # Store statistics of each existing signal region
        if pulsar_phases.regions.P1 is not None:
            self.P1sTime.append(pulsar_phases.regions.P1.sign)
            self.P1exTime.append(pulsar_phases.regions.P1.Nex)
            self.P1exerror.append(pulsar_phases.regions.P1.yerr)

        if pulsar_phases.regions.P2 is not None:
            self.P2sTime.append(pulsar_phases.regions.P2.sign)
            self.P2exTime.append(pulsar_phases.regions.P2.Nex)
            self.P2exerror.append(pulsar_phases.regions.P2.yerr)

        if pulsar_phases.regions.P3 is not None:  #(LBZ) Add P3 storage
            self.P3sTime.append(pulsar_phases.regions.P3.sign)  #(LBZ)
            self.P3exTime.append(pulsar_phases.regions.P3.Nex)  #(LBZ)
            self.P3exerror.append(pulsar_phases.regions.P3.yerr)  #(LBZ)

        if pulsar_phases.regions.P1P2 is not None:
            self.P1P2sTime.append(pulsar_phases.regions.P1P2.sign)
            self.P1P2exTime.append(pulsar_phases.regions.P1P2.Nex)
            self.P1P2exerror.append(pulsar_phases.regions.P1P2.yerr)

    def update_tinfo(self, pulsar_phases, dataframe):
        # Update info in the main object
        pulsar_phases.phases = np.array(dataframe.pulsar_phase.to_list())
        pulsar_phases.times = np.array(dataframe.dragon_time.to_list())
        pulsar_phases.info = dataframe

        try:
            pulsar_phases.mjd_times = np.array(dataframe.mjd_time.to_list())
            if len(pulsar_phases.mjd_times) > 0:
                self.bin_dates.append(float(np.max(pulsar_phases.mjd_times)))
            else:
                self.bin_dates.append(None)
        except AttributeError:
            self.bin_dates.append(None)

        if "delta_t" in pulsar_phases.info:
            diff = pulsar_phases.info.delta_t
            self.t.append(sum(diff[diff < 1]))
        else:
            diff = abs(
                pulsar_phases.info.dragon_time.values[1:]
                - pulsar_phases.info.dragon_time.values[:-1]
            )
            self.t.append(sum(diff[diff < 1]))

        pulsar_phases.tobs = self.t[-1] / 3600

        # Calculate stats
        pulsar_phases.update_info()

        # Store the results
        self.store_Tvalues(pulsar_phases)

    def run(self, pulsar_phases):
        dataframe = pulsar_phases.info
        diff = abs(dataframe.dragon_time.values[1:] - dataframe.dragon_time.values[:-1])
        s = 0
        self.t = [0]

        # Estimate from each telescope the interval of time at which we can ignore the differences of time
        if pulsar_phases.telescope == "fermi":
            self.diff_del = 3600 * 5
        else:
            self.diff_del = 3600

        # Update the information
        for i in range(0, len(diff)):
            if diff[i] < self.diff_del:
                s = s + diff[i]

                # Update information when tobs>=tint
                if s > self.tint:
                    pulse_df_partial = dataframe[
                        dataframe["dragon_time"] < dataframe.dragon_time.values[i + 1]
                    ]
                    self.update_tinfo(pulsar_phases, pulse_df_partial)
                    s = 0

                # Update last interval
                elif i == (len(diff) - 1):
                    self.update_tinfo(pulsar_phases, dataframe)

            else:
                diff[i] = 0

    def _add_month_grid(self, ax, x_values, show_labels=True):
        # Add vertical grid lines at month boundaries, labelled by month.
        # x_values : cumulative observation time in the SAME units as the axis.
        # It must be aligned with self.bin_dates (same length, same order).
        from astropy.time import Time

        valid = [
            (x_values[i], self.bin_dates[i])
            for i in range(len(self.bin_dates))
            if self.bin_dates[i] is not None and i < len(x_values)
        ]

        if len(valid) < 2:
            return

        x_vals, dates_mjd = zip(*valid)
        x_vals = np.array(x_vals)
        dates_py = Time(np.array(dates_mjd), format="mjd", scale="utc").datetime

        prev_ym = (dates_py[0].year, dates_py[0].month)
        for i in range(1, len(dates_py)):
            curr_ym = (dates_py[i].year, dates_py[i].month)
            if curr_ym != prev_ym:
                # Interpolate the cumulative time at the month boundary
                x_mid = (x_vals[i - 1] + x_vals[i]) / 2
                ax.axvline(x=x_mid, color="grey", linestyle="--", alpha=0.4, zorder=0)

                if show_labels:
                    label = dates_py[i].strftime("%b %Y")
                    ax.text(
                        x_mid,
                        0.98,
                        label,
                        ha="center",
                        va="top",
                        fontsize=7,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False,
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=1
                        ),
                    )
            prev_ym = curr_ym

    ##############################################
    # RESULTS
    #############################################

    def show_results(self):
        fig1 = self.PsigVsTime()
        fig2 = self.PexVsTime()
        fig3 = self.StatsVsTime()

        return fig1, fig2, fig3

    def compare_Peaksig(self):
        # Show all the signal regions vs Time if they exist

        fig = plt.figure()
        if len(self.P1P2sTime) > 1:
            plt.plot(
                np.array(self.t) / 3600,
                self.P1P2sTime,
                "o-",
                color="tab:red",
                label="P1+P2",
            )

        if len(self.P1sTime) > 1:
            plt.plot(
                np.array(self.t) / 3600,
                self.P1sTime,
                "o-",
                color="tab:orange",
                label="P1",
            )

        if len(self.P2sTime) > 1:
            plt.plot(
                np.array(self.t) / 3600,
                self.P2sTime,
                "o-",
                color="tab:green",
                label="P2",
            )

        self._add_month_grid(plt.gca(), np.array(self.t) / 3600, show_labels=True)

        plt.xlabel("Time of observation (h)")
        plt.ylabel("Significance ($\sigma$)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        return fig

    def PsigVsTime(self):
        xi = np.linspace(self.t[0] / 3600, self.t[-1] / 3600, 1000)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  #(LBZ) Changed to 2x2 grid for P3
        fig.set_figheight(6)  #(LBZ) Increased height for 2 rows
        fig.set_figwidth(12)

        #####################################
        # First plot: P1Sig vs Time
        if len(self.P1sTime) > 1:
            ax1.plot(
                np.array(self.t) / 3600,
                self.P1sTime,
                "o-",
                color="tab:orange",
                label="P1",
            )
            ax1.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.P1sTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax1.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:orange")

            ax1.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax1.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax1.set_xlabel("Time of observation (h)")
        ax1.set_ylabel("Significance (sigma)")
        ax1.set_title("P1")

        #####################################

        # Second plot: P2Sig vs Time

        if len(self.P2sTime) > 1:
            ax2.plot(
                np.array(self.t) / 3600,
                self.P2sTime,
                "o-",
                color="tab:green",
                label="P2",
            )
            ax2.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.P2sTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax2.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:green")

            ax2.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax2.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax2.set_xlabel("Time of observation (h)")
        ax2.set_ylabel("Significance (sigma)")
        ax2.set_title("P2")

        #####################################

        # Third plot: P1+P2 Sig vs Time

        if len(self.P1P2sTime) > 1:
            ax3.plot(
                np.array(self.t) / 3600,
                self.P1P2sTime,
                "o-",
                color="tab:red",
                label="P1+P2",
            )
            ax3.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.P1P2sTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax3.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:red")

            ax3.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax3.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax3.set_xlabel("Time of observation (h)")
        ax3.set_ylabel("Significance (sigma)")
        ax3.set_title("P1+P2")

        #####################################  #(LBZ) Add P3 plot
        # Fourth plot: P3Sig vs Time

        if len(self.P3sTime) > 1:  #(LBZ)
            ax4.plot(
                np.array(self.t) / 3600,
                self.P3sTime,
                "o-",
                color="tab:blue",
                label="P3",
            )  #(LBZ)
            ax4.grid()  #(LBZ)

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.P3sTime,
                p0=[0.1],
                maxfev=5000,
            )  #(LBZ)
            ax4.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:blue")  #(LBZ)

            ax4.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )  #(LBZ)

        else:
            # Plot not available if some signal is not valid
            ax4.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )  #(LBZ)

        ax4.set_xlabel("Time of observation (h)")  #(LBZ)
        ax4.set_ylabel("Significance (sigma)")  #(LBZ)
        ax4.set_title("P3")  #(LBZ)

        # Month date grid (vertical lines at month boundaries)
        x_vals = np.array(self.t) / 3600
        self._add_month_grid(ax1, x_vals, show_labels=False)
        self._add_month_grid(ax2, x_vals, show_labels=True)
        self._add_month_grid(ax3, x_vals, show_labels=False)
        self._add_month_grid(ax4, x_vals, show_labels=False)

        plt.tight_layout()
        plt.show()

        return fig

    def PexVsTime(self):
        xi = np.linspace(self.t[0] / 3600, self.t[-1] / 3600, 1000)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  #(LBZ) Changed to 2x2 grid for P3
        fig.set_figheight(6)  #(LBZ) Increased height for 2 rows
        fig.set_figwidth(12)

        #####################################

        # First plot: P1Ex vs Time

        if len(self.P1exTime) > 1:
            ax1.errorbar(
                np.array(self.t) / 3600,
                self.P1exTime,
                yerr=self.P1exerror,
                fmt="o-",
                color="tab:orange",
                label="P1",
            )
            ax1.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_lin,
                np.array(self.t) / 3600,
                self.P1exTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax1.plot(xi, function_lin(xi, *ppot), "--", color="tab:orange")

            ax1.annotate(
                "Nex=At"
                + f" \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})"
                + r"h$^{-1}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax1.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax1.set_xlabel("Time of observation (h)")
        ax1.set_ylabel("Number of excess events")
        ax1.set_title("P1")
        #####################################

        # Second plot: P2Ex vs time.

        if len(self.P2exTime) > 1:
            ax2.errorbar(
                np.array(self.t) / 3600,
                self.P2exTime,
                yerr=self.P2exerror,
                fmt="o-",
                color="tab:green",
                label="P2",
            )
            ax2.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_lin,
                np.array(self.t) / 3600,
                self.P2exTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax2.plot(xi, function_lin(xi, *ppot), "--", color="tab:green")

            ax2.annotate(
                "Nex=At"
                + f" \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})"
                + r"h$^{-1}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax2.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax2.set_xlabel("Time of observation (h)")
        ax2.set_ylabel("Number of excess events")
        ax2.set_title("P2")

        #####################################

        # Third plot: P1+P2 Ex vs time.

        if len(self.P1P2exTime) > 1:
            ax3.errorbar(
                np.array(self.t) / 3600,
                self.P1P2exTime,
                yerr=self.P1P2exerror,
                fmt="o-",
                color="tab:red",
                label="P1+P2",
            )
            ax3.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_lin,
                np.array(self.t) / 3600,
                self.P1P2exTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax3.plot(xi, function_lin(xi, *ppot), "--", color="tab:red")

            ax3.annotate(
                "Nex=At"
                + f" \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})"
                + r"h$^{-1}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax3.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        ax3.set_xlabel("Time of observation (h)")
        ax3.set_ylabel("Number of excess events")
        ax3.set_title("P1+P2")

        #####################################  #(LBZ) Add P3 plot
        # Fourth plot: P3Ex vs time.

        if len(self.P3exTime) > 1:  #(LBZ)
            ax4.errorbar(
                np.array(self.t) / 3600,
                self.P3exTime,
                yerr=self.P3exerror,
                fmt="o-",
                color="tab:blue",
                label="P3",
            )  #(LBZ)
            ax4.grid()  #(LBZ)

            # Fit to the points
            ppot, pcov = curve_fit(
                function_lin,
                np.array(self.t) / 3600,
                self.P3exTime,
                p0=[0.1],
                maxfev=5000,
            )  #(LBZ)
            ax4.plot(xi, function_lin(xi, *ppot), "--", color="tab:blue")  #(LBZ)

            ax4.annotate(
                "Nex=At"
                + f" \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})"
                + r"h$^{-1}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )  #(LBZ)

        else:
            # Plot not available if some signal is not valid
            ax4.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )  #(LBZ)

        ax4.set_xlabel("Time of observation (h)")  #(LBZ)
        ax4.set_ylabel("Number of excess events")  #(LBZ)
        ax4.set_title("P3")  #(LBZ)

        # Month date grid (vertical lines at month boundaries)
        x_vals = np.array(self.t) / 3600
        self._add_month_grid(ax1, x_vals, show_labels=False)
        self._add_month_grid(ax2, x_vals, show_labels=True)
        self._add_month_grid(ax3, x_vals, show_labels=False)
        self._add_month_grid(ax4, x_vals, show_labels=False)

        plt.tight_layout()
        plt.show()

        return fig

    def StatsVsTime(self):
        xi = np.linspace(self.t[0] / 3600, self.t[-1] / 3600, 1000)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)

        #####################################

        # First plot: H-test vs time.

        if (
            self.HTime is not None
            and isinstance(self.HTime[-1], float)
            and self.HTime[-1] != float("inf")
        ):
            ax1.plot(np.array(self.t) / 3600, self.HTime, "o-", color="tab:pink")
            ax1.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.HTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax1.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:pink")

            ax1.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax1.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        # Set labels
        ax1.set_xlabel("Time of observation (h)")
        ax1.set_ylabel("Significance (sigma)")
        ax1.set_title("H test")

        #####################################

        # Second plot: Zn-test vs time.

        if (
            self.ZTime is not None
            and isinstance(self.ZTime[-1], float)
            and self.ZTime[-1] != float("inf")
        ):
            ax2.plot(np.array(self.t) / 3600, self.ZTime, "o-", color="tab:cyan")
            ax2.grid()

            # Fit to the points
            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.ZTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax2.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:cyan")

            ax2.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        else:
            # Plot not available if some signal is not valid
            ax2.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        # Set labels
        ax2.set_xlabel("Time of observation (h)")
        ax2.set_ylabel("Significance (sigma)")
        ax2.set_title("Z test")

        #####################################

        # Third plot: CHi2 test vs time

        if (
            self.ChiTime is not None
            and isinstance(self.ChiTime[-1], float)
            and self.ChiTime[-1] != float("inf")
        ):
            ax3.plot(np.array(self.t) / 3600, self.ChiTime, "o-", color="tab:olive")
            ax3.set_xlabel("Time of observation (h)")
            ax3.set_ylabel("Significance (sigma)")
            ax3.set_title("Chi square test")
            ax3.grid()

            ppot, pcov = curve_fit(
                function_sqrt,
                np.array(self.t) / 3600,
                self.ChiTime,
                p0=[0.1],
                maxfev=5000,
            )
            ax3.plot(xi, function_sqrt(xi, *ppot), "--", color="tab:olive")

            ax3.annotate(
                "Sig=A"
                + r"$\sqrt{t}$"
                + f" \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})"
                + r"h$^{-1/2}$",
                xy=(0.95, 0.35),
                xytext=(0.95, 0.35),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )
        else:
            # Plot not available if some signal is not valid
            ax3.annotate(
                "Plot not available",
                xy=(0.8, 0.6),
                xytext=(0.8, 0.6),
                fontsize=10,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )

        # Set labels
        ax3.set_xlabel("Time of observation (h)")
        ax3.set_ylabel("Significance (sigma)")
        ax3.set_title("Chi square test")

        # Month date grid (vertical lines at month boundaries)
        x_vals = np.array(self.t) / 3600
        self._add_month_grid(ax1, x_vals, show_labels=False)
        self._add_month_grid(ax2, x_vals, show_labels=True)
        self._add_month_grid(ax3, x_vals, show_labels=False)

        plt.tight_layout()
        plt.show()

        return fig
