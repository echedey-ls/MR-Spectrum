"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
E_λ<λ₀/E calculation workflow from a set inputs is condensed here.
See :class:`MR_E_ratio` for more details.
"""

# Imports
from irradiance_ratios import E_lambda_over_E, LAMBDA0
from tools import day_of_year

from pvlib.spectrum import spectrl2
from pvlib.irradiance import aoi
from pvlib.location import Location
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from itertools import product
from functools import partial
from datetime import datetime
from time import time
from typing import Callable


class MR_E_ratio:
    """
    Group common workflow in the most flexible way possible to test with a wide range
    of input data, generate spectrums and integrate them.
    """

    def __init__(
        self,
        cutoff_lambda: str | float = "monosi",
        n=20,
        location: Location = None,
        datetimes: pd.DatetimeIndex = None,
        surface_tilt=31,
        surface_azimuth=180,
        ozone=0.31,
    ):
        """
        Create bench
        """
        self.reset_simulation_state()

        if isinstance(cutoff_lambda, float):
            self.cutoff_lambda = cutoff_lambda
        elif isinstance(cutoff_lambda, str):
            self.cutoff_lambda = LAMBDA0[cutoff_lambda]
        else:
            raise TypeError(
                f"Provide a valid cutoff_lambda. Valid strings are {set(LAMBDA0.keys())}"
            )

        self.n = n
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.ozone = ozone

        if location:
            self.locus = location
        else:
            self.locus = Location(
                latitude=40,
                longitude=-3.7,
                altitude=650,
                tz="Europe/Madrid",
                name="Madrid",
            )

        if datetimes is not None:
            self.datetimes = datetimes
        else:
            self.datetimes = pd.date_range(
                "2023-11-27T04",
                "2023-11-27T22",
                freq=pd.Timedelta(
                    minutes=5
                ),  # unit="s" bugs this TODO?: report to PVLIB
            )

    def reset_simulation_state(self):
        """
        Self-explanatory. Just in case.
        Simulation can be re-run after resetting.
        """
        self.solpos = None
        self.aoi = None
        self.time_params = None
        self.input_keys = None
        self.results = None
        self.processing_time = dict()

    def times_summary(self):
        print("Timing of bench methods:")
        for key, value in self.processing_time.items():
            if value:
                print(f"\t{key}: {value:.4} s")

    def simulation_prerun(self):
        """
        Calculates some values from scratch, in case they were updated from the outside
        """
        start_time = time()  # Initialize start time of block
        self.constant_params = {
            "surface_tilt": self.surface_tilt,  # degrees
            "ground_albedo": 0.25,  # concrete pavement
            # ref. assumes from 0.31 to 0.3444 atm-cm & ozone does not have much impact
            # in spectra for Silicon (c-Si, a-Si) devices, so we are excluding it
            "ozone": self.ozone,
        }
        # Time-dependant values
        self.solpos = self.locus.get_solarposition(self.datetimes)
        self.aoi = aoi(
            self.surface_tilt,
            self.surface_azimuth,
            self.solpos["apparent_zenith"],
            self.solpos["azimuth"],
        )
        self.time_params = {
            "apparent_zenith": self.solpos["apparent_zenith"].to_numpy(),
            "aoi": self.aoi.to_numpy(),
            "relative_airmass": self.locus.get_airmass(solar_position=self.solpos)[
                "airmass_relative"
            ].to_numpy(),
            "dayofyear": np.fromiter(
                map(day_of_year, self.datetimes), dtype=np.float64
            ),
        }
        self.processing_time["simulation_prerun"] = time() - start_time

    def simulate_from_product(self, **inputvals):
        """
        Process a simulation from **inputvals.

        inputvals are keyword arguments, numpy 1-d arrays.
        It must contain SPECTRL2 required parameters:
          * surface_pressure
          * precipitable_water
          * aerosol_turbidity_500nm

        And may also contain optional parameters:
          * scattering_albedo_400nm=0.945
          * alpha=1.14
          * wavelength_variation_factor=0.095
          * aerosol_asymmetry_factor=0.65

        Saves results to a dataframe with the following shape:
          ==================== ========================================================
          input/datetimes keys poa_sky_diffuse poa_ground_diffuse poa_direct poa_global
          ==================== ========================================================
            parameter values                       E_λ<λ₀/E values
          ==================== ========================================================
        """
        # Initialize needed values, in case they were changed from the outside
        self.simulation_prerun()

        # Start timer after prerun
        start_time = time()  # Initialize start time of block

        self.input_keys = (*inputvals.keys(),)

        # Simulation results dataframe
        spectrl2_input_columns = (
            *self.input_keys,
            *self.time_params.keys(),
        )
        spectrl2_output_columns = (
            "poa_sky_diffuse",
            "poa_ground_diffuse",
            "poa_direct",
            "poa_global",
        )
        n_inputvals_combinations = np.prod([len(array) for array in inputvals.values()])
        self.results = pd.DataFrame(
            # pre-allocate
            columns=spectrl2_input_columns + spectrl2_output_columns,
            dtype=np.float64,
        )

        self.results[[*self.input_keys]] = np.fromiter(
            product(*inputvals.values()),
            dtype=np.dtype((np.float64, len(self.input_keys))),
        ).repeat(len(self.datetimes), axis=0)

        self.results[[*self.time_params.keys()]] = np.tile(
            np.asarray((*self.time_params.values(),)), n_inputvals_combinations
        ).T

        spectrl2_result = spectrl2(
            **self.constant_params,
            **{col: self.results[col].to_numpy() for col in spectrl2_input_columns},
        )

        # Following partial func. only takes the spectral irradiance as argument
        wrapped_E_lambda_over_E = partial(
            E_lambda_over_E,
            self.cutoff_lambda,
            spectrl2_result["wavelength"],
        )
        for output_name in spectrl2_output_columns:
            self.results[output_name] = np.fromiter(
                map(
                    wrapped_E_lambda_over_E,
                    spectrl2_result[output_name].swapaxes(1, 0),
                ),
                dtype=np.float64,
            )

        self.processing_time["simulate_from_product"] = time() - start_time
        
        self.simulation_post()

    def simulation_post(self):
        """
        Run tasks after simulation processing
        """
        start_time = time()  # Initialize start time of block
        self.post_summary()
        self.processing_time["simulation_post"] = time() - start_time

    def post_summary(self):
        """
        Print condensed statistics to console
        """
        means = self.results.filter(regex="poa_").mean().dropna()
        stdvs = self.results.filter(regex="poa_").std().dropna()
        print("Simulation Results")
        print(f"> Cutoff wavelength: {self.cutoff_lambda} nm")
        print("> Mean E_λ<λ₀/E =")
        print(means)
        print("> Std  E_λ<λ₀/E =")
        print(stdvs)

    def plot_results(
        self, *, plot_keys: set = None, max_cols=2, savefig=True
    ) -> plt.Figure:
        """
        Generate a plot of 'E fraction' vs each input variable from
        self.simulate_from_product(...) and variable names at.
        Optionally, a set of variables can be specified via parameter ``plot_keys``.
        Defaults to plot all available and ``relative_airmass``.
        """
        start_time = time()  # Initialize start time of block
        # cast plot_keys to set of strings to plot E fraction against
        if plot_keys is None:  # default to add relative_airmass
            plot_keys = ["relative_airmass", *self.input_keys]
        elif isinstance(plot_keys, str):
            plot_keys = [plot_keys]
        elif not isinstance(plot_keys, list):
            plot_keys = list(plot_keys)

        # assume we've got an iterable of strings
        # make at most two columns
        cols = min(max_cols, len(plot_keys))
        rows = int(np.ceil(len(plot_keys) / cols))
        fig, axs = plt.subplots(ncols=cols, nrows=rows)

        if isinstance(axs, np.ndarray):  # to allow iteration in one dimension
            axs = axs.flatten()
        else:  # plt.Axes type
            axs = [axs]  # to allow iteration of just that element
        axs = iter(axs)

        fig.suptitle(
            r"$\frac{E_{λ<λ_0}}{E}$ as function of SPECTRL2 inputs"
            + f"\nλ₀={self.cutoff_lambda} nm"
        )
        fig.set_size_inches(12, 12)

        # get output & each of the variables
        ydata = self.results["poa_global"]
        xdata = self.results[plot_keys]

        # plot output against each of the variables
        for var_name, var_values in xdata.items():
            ax = next(axs)
            ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ vs. " + var_name)
            ax.scatter(var_values, ydata)

        if savefig:
            fig.savefig(
                f"E_ratio_lambda{self.cutoff_lambda:04.0f}_"
                + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                + ".png"
            )
        plt.close()

        self.processing_time["plot_results"] = time() - start_time
        return fig

    def plot_results_3d(self, plot_keys, ax=None):
        if len(plot_keys) != 2:
            raise ValueError("2 values must be provided for X&Y axes of 3D plot.")
        if ax is None:
            ax = plt.subplot(projection="3d")
        # get output & each of the variables
        z, (x, y) = self.get_1d_arrays_from(plot_keys)
        ax.scatter(x, y, z)
        ax.set_title(
            r"$\frac{E_{λ<λ_0}}{E}$ as function of "
            + plot_keys[0]
            + " & "
            + plot_keys[1]
        )
        ax.set_xlabel(plot_keys[0])
        ax.set_ylabel(plot_keys[1])
        ax.set_zlabel(r"$\frac{E_{λ<λ_0}}{E}$")

    def optimization_from_model(
        self, model: Callable = None, model_inputs: tuple = None, **kwargs
    ):
        """
        Optimize a model to fit generated data.

        Parameters
        ----------
        model : Callable
            Function with the model to be optimised.
        model_inputs : str or iterable of str
            Order and parameters of ``model``. Must be any of:
                * ``datetime``
                * ``apparent_zenith``, ``aoi``, ``relative_airmass`` or ``dayofyear``
                * any parameter name provided to ``simulate_from_product``
        **kwargs :
            Redirected to ``scipy.optimize.curve_fit``.

        Returns
        -------
        ``scipy.optimize.curve_fit``'s return values
        """
        start_time = time()  # Initialize start time of block

        # get output & each of the variables
        ydata, xdata = self.get_1d_arrays_from(model_inputs)

        curve_fit_results = curve_fit(model, xdata, ydata, nan_policy="omit", **kwargs)

        self.processing_time["optimization_from_model"] = time() - start_time
        return curve_fit_results
