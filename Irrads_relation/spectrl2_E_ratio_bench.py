"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
E_λ<λ₀/E calculation workflow from a set inputs is condensed here.
See :class:`MR_E_ratio` for more details.
"""

# Imports
from irrads_relation_fracs import E_lambda_over_E, LAMBDA0
from tools import _get_optional_params

from pvlib.spectrum import spectrl2
from pvlib.irradiance import aoi
from pvlib.location import Location
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from functools import partial
from warnings import warn


class MR_E_ratio:
    """
    Group common workflow in the most flexible way possible to test with a wide range
    of input data, generate spectrums and integrate them.
    """

    def __init__(self, cutoff_lambda: str | float = 0.0, **kwargs):
        """
        Allow for initialization
        """
        params = _get_optional_params(self.init_values)
        self.init_values(
            cutoff_lambda=cutoff_lambda,
            **{
                param: (kwargs.pop(param, None))
                for param in params
                if kwargs.get(param) is not None
            },
        )
        if len(kwargs) > 0:
            warn(
                "Unused kwargs in bench!\n"
                + "\n".join(f"\t{key}: {value}" for key, value in params)
            )

    def init_values(
        self,
        cutoff_lambda: str | float,
        n=20,
        location: Location = None,
        dates: pd.DatetimeIndex = None,
        surface_tilt=31,
        surface_azimuth=180,
        ozone=0.31,
    ):
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

        if dates:
            self.datetimes = dates
        else:
            self.datetimes = pd.date_range(
                "2023-11-22T04",
                "2023-11-27T22",
                freq=pd.Timedelta(hours=0.5),  # unit="s" TODO: report to PVLIB?
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

    def simulation_prerun(self):
        """
        Calculates some values from scratch, in case they were updated from the outside
        """
        self.fixed_params = {
            "surface_tilt": self.surface_tilt,  # degrees
            "ground_albedo": 0.25,  # concrete pavement
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
            "apparent_zenith": self.solpos["apparent_zenith"],
            "aoi": self.aoi,
            "relative_airmass": self.locus.get_airmass(self.datetimes, self.solpos)[
                "airmass_relative"
            ],
            "dayofyear": None,  # np.array(map(day_of_year, datetimes)) ???
            # ref. assumes from 0.31 to 0.3444 atm-cm & ozone does not have much impact
            # in spectra, so we are excluding it for now
            "ozone": self.ozone,
        }

    def simulate_from_product(self, **inputvals):
        """
        Process a simulation from inputvals.

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
        """
        self.input_keys = inputvals.keys()
        # Initialize needed values, in case they were changed from the outside
        self.simulation_prerun()
        # Simulation results, save an entry for each of the cartesian product
        self.results = pd.DataFrame(
            columns=(*self.input_keys, *self.time_params["apparent_zenith"]),
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
                **self.fixed_params,
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

        self.simulation_post()

    def simulation_post(self):
        """
        Run tasks after simulation processing
        """
        self.post_summary()

    def post_summary(self):
        """
        Print condensed statistics to console
        """
        means = self.results.iloc[:, 3:].mean().dropna()
        # stdvs = self.results.iloc[:, 3:].std().dropna()
        print("Simulation Results")
        print(f"> Cutoff wavelength: {self.cutoff_lambda}")
        print(f"> Mean E_λ<λ₀/E = {means.mean()}")
        # print(f"Zenith\t Mean of avg(E_λ<λ₀/E)\n{means}")
        print(f"> Std  E_λ<λ₀/E = {means.std()}")
        # print(f"Zenith\t STD of avg(E_λ<λ₀/E)\n{stdvs}")

    def plot_results(
        self, *, plot_keys=None, max_cols=2
    ) -> plt.Figure:  # TODO: work on this
        """
        Generate a plot of 'E fraction' vs each input variable from
        self.simulate_from_product(...) and 'apparent_zenith'.
        Optionally, a set of variables can be specified via parameter 'plot_keys'.
        Defaults to plot all available.
        """
        if plot_keys is None:  # default to add apparent zenith
            plot_keys = ("apparent_zenith", *self.input_keys)
        elif isinstance(plot_keys, str):
            plot_keys = (plot_keys,)  # cast to tuple

        # assume we've got an iterable of strings
        # make at most two columns
        cols = len(plot_keys) % max_cols + 1
        rows = np.ceil(len(plot_keys) / cols)
        fig, axs = plt.subplots(ncols=cols, nrows=rows)

        # for each axes, plot a relationship
        for ax, var_name in zip(axs, plot_keys, strict=True):
            self.results[var_name].plot(kind="scatter", ax=ax)
