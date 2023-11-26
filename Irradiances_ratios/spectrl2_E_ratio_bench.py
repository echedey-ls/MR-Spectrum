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
                print(f"\t{key}: {value} s")

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
        It must contain spectrl2 required parameters:
          * surface_pressure
          * precipitable_water
          * aerosol_turbidity_500nm

        And may also contain optional parameters:
          * scattering_albedo_400nm=0.945
          * alpha=1.14
          * wavelength_variation_factor=0.095
          * aerosol_asymmetry_factor=0.65

        Saves results to a dataframe with the following shape:
          ==================== ========================
          ... input values ... ... datetimes values ...
          ==================== ========================
            parameter values        E_λ<λ₀/E values
          ==================== ========================
        """
        # Initialize needed values, in case they were changed from the outside
        self.simulation_prerun()

        # Start timer after prerun
        start_time = time()  # Initialize start time of block

        self.input_keys = (*inputvals.keys(),)

        # Simulation results, save an entry for each of the cartesian product
        self.results = pd.DataFrame(
            # columns after *self.input_keys only represent a position in
            # self.time_params
            columns=(*self.input_keys, *self.datetimes),
            # pre-allocate to the length of the itertools.product result
            index=np.arange(np.prod([len(array) for array in inputvals.values()])),
            dtype=np.float64,
        )
        for index, product_tuple in enumerate(product(*inputvals.values())):
            product_input = dict(zip(self.input_keys, product_tuple))
            # 'wavelength', 'dni_extra', 'dhi', 'dni', 'poa_sky_diffuse',
            # 'poa_ground_diffuse', 'poa_direct', 'poa_global'
            spectrl2_result = spectrl2(
                **product_input,
                **self.constant_params,
                **self.time_params,
            )
            self.results.iloc[index] = [
                *product_tuple,
                *map(
                    partial(
                        E_lambda_over_E,
                        self.cutoff_lambda,
                        spectrl2_result["wavelength"],
                    ),
                    spectrl2_result["poa_global"].swapaxes(1, 0),
                ),
            ]

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
        n_inputs = len(self.input_keys)
        means = self.results.iloc[:, n_inputs:].mean().dropna()
        # stdvs = self.results.iloc[:, n_inputs:].std().dropna()
        print("Simulation Results")
        print(f"> Cutoff wavelength: {self.cutoff_lambda} nm")
        print(f"> Mean E_λ<λ₀/E = {means.mean()}")
        # print(f"Zenith\t Mean of avg(E_λ<λ₀/E)\n{means}")
        print(f"> Std  E_λ<λ₀/E = {means.std()}")
        # print(f"Zenith\t STD of avg(E_λ<λ₀/E)\n{stdvs}")

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
            plot_keys = {"relative_airmass", *self.input_keys}
        elif isinstance(plot_keys, str):
            plot_keys = {
                plot_keys,
            }
        elif not isinstance(plot_keys, set):
            plot_keys = set(plot_keys)

        # variable guard: only allow valid keys:
        #   * self.input_keys & self.time_params
        allowed_keys = set(self.input_keys) | self.time_params.keys() | {"datetime"}
        invalid_keys = plot_keys - allowed_keys
        if invalid_keys == {}:
            raise ValueError(
                "Incorrect key provided.\n"
                + f"Allowed keys are: {allowed_keys}\n"
                + f"Invalid keys are: {invalid_keys}"
            )
        del allowed_keys, invalid_keys

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

        # number of inputs from user: n-left-most columns
        n_inputs = len(self.input_keys)

        # for each axes, plot a relationship
        # Case: time
        for var_name in plot_keys.intersection({"datetime"}):
            ax = next(axs)
            ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ vs. " + var_name)
            x = self.datetimes if var_name == "datetime" else None
            for _, row in self.results.iloc[n_inputs:].iterrows():
                ax.scatter(x, row[n_inputs:])
            plot_keys.remove(var_name)

        # Case: time-dependant variables in plot_keys
        for var_name in plot_keys.intersection(self.time_params.keys()):
            ax = next(axs)
            ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ vs. " + var_name)
            x = self.time_params[var_name]
            for _, row in self.results.iloc[n_inputs:].iterrows():
                ax.scatter(x, row[n_inputs:])
            plot_keys.remove(var_name)

        # Case: SPECTRL2 generator input parameters
        for var_name in plot_keys:
            ax = next(axs)
            ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ vs. " + var_name)
            x = self.results[var_name]
            y_df = self.results.iloc[:, n_inputs:]
            for _, y_content in y_df.items():
                ax.scatter(x, y_content)

        if savefig:
            fig.savefig(
                f"E_ratio_lambda{self.cutoff_lambda:04.0f}_"
                + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                + ".png"
            )
        plt.close()

        self.processing_time["plot_results"] = time() - start_time

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

        if isinstance(model_inputs, str):
            model_inputs = (model_inputs,)

        # number of inputs from user: n-left-most columns
        n_inputs = len(self.input_keys)
        # Fitting data
        ydata = self.results.iloc[:, n_inputs:].to_numpy().flatten()
        # Prepare input vector
        xdata = []  # length of each value must be
        dates_len = len(self.datetimes)
        try:
            for var_name in model_inputs:
                # broadcast all inputs to match ydata
                if var_name in self.input_keys:
                    xdata.append(self.results[var_name].to_numpy().repeat(dates_len))
                elif var_name in self.time_params.keys():
                    xdata.append(
                        np.tile(self.time_params[var_name], self.results.shape[0])
                    )
                elif var_name in {"datetime"}:
                    xdata.append(
                        np.tile(self.datetimes.to_numpy(), self.results.shape[0])
                    )
                else:
                    raise ValueError(f"'{var_name}' is not a valid parameter name!")

        except TypeError:
            raise TypeError(
                "Provide a valid model input names vector. Must be iterable"
                + " of strings, and that input will be provided to 'model'"
                + f" in the same order.\nYou provided {model_inputs}"
            )

        ## This is kept here for debug purposes: check valid representation as 1D arrays
        # fig, axs = plt.subplots(len(model_inputs))
        # for i, name in enumerate(model_inputs):
        #     axs[i].set_title(name)
        #     axs[i].scatter(xdata[i], ydata)
        # plt.show()

        curve_fit_results = curve_fit(model, xdata, ydata, nan_policy="omit", **kwargs)

        self.processing_time["optimization_from_model"] = time() - start_time
        return curve_fit_results
