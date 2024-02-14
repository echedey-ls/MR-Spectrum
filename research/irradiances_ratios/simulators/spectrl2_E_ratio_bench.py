"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
E_λ<λ₀/E calculation workflow from a set inputs is condensed here.
See :class:`MR_E_ratio` for more details.
"""

# Imports
from irradiances_ratios.ratios_calculator import E_lambda_over_E, LAMBDA0
from utils.tools import day_of_year

from pvlib.spectrum import spectrl2
from pvlib.irradiance import (
    aoi,
    clearness_index,
    ghi_from_poa_driesse_2023,
)
from pvlib.location import Location
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from itertools import product
from functools import partial
from pathlib import Path
from time import time
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class MR_SPECTRL2_E_ratio_bench:
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
        self.constant_params = {
            "surface_tilt": self.surface_tilt,  # degrees
            "ground_albedo": 0.25,  # concrete pavement
            # ref. assumes from 0.31 to 0.3444 atm-cm & ozone does not have much impact
            # in spectra for Silicon (c-Si, a-Si) devices, so we are excluding it
            "ozone": self.ozone,
        }

    def reset_simulation_state(self):
        """
        Self-explanatory. Just in case.
        Simulation can be re-run after resetting.
        """
        self.solpos = None
        self.aoi = None
        self.spectrl2_time_params = None
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
        # Time-dependant values
        self.solpos = self.locus.get_solarposition(self.datetimes)
        self.aoi = aoi(
            self.surface_tilt,
            self.surface_azimuth,
            self.solpos["apparent_zenith"],
            self.solpos["azimuth"],
        )
        airmasses = self.locus.get_airmass(solar_position=self.solpos)
        self.spectrl2_time_params = {
            "apparent_zenith": self.solpos["apparent_zenith"].to_numpy(),
            "aoi": self.aoi.to_numpy(),
            "relative_airmass": airmasses["airmass_relative"].to_numpy(),
            "dayofyear": np.fromiter(
                map(day_of_year, self.datetimes), dtype=np.float64
            ),
        }
        self.other_time_params = {
            "absolute_airmass": airmasses["airmass_absolute"].to_numpy()
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
          ================ ===================== ========================
          SPECTRL2 inputs  poa_components_ratios poa_components_integrals
          ================ ===================== ========================
          Parameter values      E_λ<λ₀/E values      Irradiances in W/m^2
          ================ ===================== ========================
        """
        # Initialize needed values, in case they were changed from the outside
        self.simulation_prerun()

        # Start timer after prerun
        start_time = time()  # Initialize start time of block

        self.input_keys = (*inputvals.keys(),)

        ## Simulation results DataFrame
        # Inputs side
        spectrl2_input_columns = (
            *self.input_keys,
            *self.spectrl2_time_params.keys(),
        )
        # Things we can base our model on
        derived_values_columns = ("clearness_index",)
        # The outputs
        spectrl2_output_columns = (  # TODO: re-evaluate what to do prior to simulation
            "poa_sky_diffuse",
            "poa_ground_diffuse",
            "poa_direct",
            "poa_global",
        )
        n_inputvals_combinations = np.prod([len(array) for array in inputvals.values()])
        self.results = pd.DataFrame(
            columns=(
                spectrl2_input_columns + derived_values_columns
                # + spectrl2_output_columns
            ),
            dtype=np.float64,
        )

        ## Fill input columns from cartesian product
        self.results[[*self.input_keys]] = np.fromiter(
            product(*inputvals.values()),
            dtype=np.dtype((np.float64, len(self.input_keys))),
        ).repeat(len(self.datetimes), axis=0)
        ## Fill time-dependant values
        # SPECTRL2 inputs
        self.results[[*self.spectrl2_time_params.keys()]] = np.tile(
            np.asarray((*self.spectrl2_time_params.values(),)), n_inputvals_combinations
        ).T
        # Other data of interest
        self.results[[*self.other_time_params.keys()]] = np.tile(
            np.asarray((*self.other_time_params.values(),)),
            n_inputvals_combinations,
        ).T
        self.results["datetimes"] = np.tile(self.datetimes, n_inputvals_combinations).T
        ## Calculate spectrums from the SPECTRL2 model
        spectrl2_result = spectrl2(
            **self.constant_params,
            **{col: self.results[col].to_numpy() for col in spectrl2_input_columns},
        )

        ## Integrate and calculate spectral ratios (unitless)
        # Following partial func. only takes the spectral irradiance as argument
        wrapped_E_lambda_over_E = partial(
            E_lambda_over_E,
            self.cutoff_lambda,
            spectrl2_result["wavelength"],
        )
        for output_name in spectrl2_output_columns:
            self.results[output_name + "_ratio"] = np.fromiter(
                map(
                    wrapped_E_lambda_over_E,
                    spectrl2_result[output_name].swapaxes(1, 0),
                ),
                dtype=np.float64,
            )
        ## Integrate specific spectral components (W/m^2)
        spectrl2_integ_components = ("poa_global",)
        wavelength_integrator = partial(np.trapz, x=spectrl2_result["wavelength"])
        for col_name in spectrl2_integ_components:
            self.results[col_name + "_integ"] = np.fromiter(
                map(
                    wavelength_integrator,
                    spectrl2_result[output_name].swapaxes(1, 0),
                ),
                dtype=np.float64,
            )

        ## Derived values contain special cases
        # -- Clearness Index (the only one for now)
        # TODO: evaluate processing bottleneck ?
        extra_radiation = np.trapz(
            spectrl2_result["dni_extra"].swapaxes(1, 0), spectrl2_result["wavelength"]
        )
        # get GHI - needs reverse transposition
        solar_azimuth = np.tile(self.solpos["azimuth"], n_inputvals_combinations).T
        ghi = ghi_from_poa_driesse_2023(
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            solar_zenith=self.results["apparent_zenith"],
            solar_azimuth=solar_azimuth,
            poa_global=self.results["poa_global_integ"],
            dni_extra=extra_radiation,
            airmass=self.results["relative_airmass"],
            albedo=self.constant_params["ground_albedo"],
            xtol=0.01,
            full_output=False,
        )

        self.results["clearness_index"] = clearness_index(
            ghi=ghi,
            solar_zenith=self.results["apparent_zenith"],
            extra_radiation=extra_radiation,
        )

        self.processing_time["simulate_from_product"] = time() - start_time
        logger.info(
            "Elapsed time for 'simulate_from_product': %s s",
            self.processing_time["simulate_from_product"],
        )

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
        means = self.results.filter(regex="poa_.*_ratio").mean()
        stdvs = self.results.filter(regex="poa_.*_ratio").std()
        logger.info(">>> Simulation Results")
        logger.info("  > Mean E_λ<λ₀/E =")
        logger.info(means)
        logger.info("  > Std  E_λ<λ₀/E =")
        logger.info(stdvs)

    def plot_results(
        self,
        *,
        components: tuple = None,
        plot_keys: set = None,
        max_cols=2,
        savefig=True,
        output_dir=Path(),
    ) -> None:
        """
        Generate a plot of 'E fraction' vs each input variable from
        self.simulate_from_product(...) and variable names at.
        Optionally, a set of variables can be specified via parameter ``plot_keys``.
        Defaults to plot all available and ``relative_airmass``.
        """
        start_time = time()  # Initialize start time of block

        ## Input data validation and standardisation
        # cast output_dir to Path object if not
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        # cast components to list of columns of E fraction to plot
        if components is None:  # default to add all calculated ratios
            components = self.results.columns[
                self.results.columns.str.match(r"poa_.*_ratio")
            ]
            # components = ["poa_global_ratio"]
        elif isinstance(components, str):
            components = [components]
        elif not isinstance(plot_keys, list):
            components = list(components)

        # cast plot_keys to set of strings to plot E fraction against
        if plot_keys is None:  # default to add relative_airmass
            plot_keys = ["relative_airmass", *self.input_keys]
        elif isinstance(plot_keys, str):
            plot_keys = [plot_keys]
        elif not isinstance(plot_keys, list):
            plot_keys = list(plot_keys)

        for component in components:
            # we've got an iterable of variable keys to plot
            # make at most two columns
            cols = min(max_cols, len(plot_keys))
            rows = int(np.ceil(len(plot_keys) / cols))
            fig, axs = plt.subplots(ncols=cols, nrows=rows)

            if isinstance(axs, np.ndarray):  # to allow iteration in one dimension
                axs = axs.flatten()
            else:  # plt.Axes type, 1 axes only
                axs = [axs]  # to allow iteration of just that element
            axs = iter(axs)

            fig.suptitle(
                r"$\frac{E_{λ<λ_0}}{E}$ as function of SPECTRL2 inputs"
                + f"\nλ₀={self.cutoff_lambda} nm"
                + f"\nComponent: {component}"
            )
            fig.set_size_inches(12, 12)

            # get output & each of the variables
            ydata = self.results[component]
            xdata = self.results[plot_keys]

            # plot output against each of the variables
            for var_name, var_values in xdata.items():
                ax = next(axs)
                ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ vs. " + var_name)
                ax.scatter(var_values, ydata)

            if savefig:
                fig.savefig(
                    output_dir.joinpath(f"E_ratio_lambda_over_E_{component}.png")
                )
                logger.info("Figure saved for component %s", component)
            plt.close()

        self.processing_time["plot_results"] = time() - start_time
        logger.info(
            "Elapsed time for 'plot_results': %s s",
            self.processing_time["plot_results"],
        )
        return

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
