"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

# %% Initialization
from research.irradiances_ratios.simulators.spectrl2_E_ratio_bench import (
    MR_SPECTRL2_E_ratio_bench,
)
from research.irradiances_ratios.ratios_calculator import LAMBDA0
from research.irradiances_ratios.models import mr_alike
from research.utils.tools import get_all_params_names

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Code environment
# Set output folder for current run based on timestamp
base_output_folder = Path("outputs/" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
if not base_output_folder.exists():
    base_output_folder.mkdir()
logging.basicConfig(
    filename=base_output_folder.joinpath("log.txt"),
    encoding="utf-8",
    level=logging.INFO,
)
logger.info("Output will be written to folder: %s", base_output_folder)

try:
    mpl.use("TkAgg")
except ImportError:
    logger.warn("Matplotlib backend can't be set to TkAgg. Output will not be shown.")

# %%
# Bench initialization
# --------------------
# Test values for the bench and define input data for the SPECTRL2 spectral model

# Matrix of values to test
# Atmosphere characterization required params
N = 8
spectrl2_generator_input = {
    # SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
    "precipitable_water": np.linspace(1.0, 2.5, N),
    # SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
    "aerosol_turbidity_500nm": np.linspace(0.08, 0.30, N),
}
lambdas = np.unique(np.fromiter(LAMBDA0.values(), dtype=float))

constant_params = {"surface_pressure": 1013100.0}

# bench instance with example time input
bench = MR_SPECTRL2_E_ratio_bench(
    datetimes=pd.date_range(
        "2023-11-27T00", "2023-11-28T00", freq=pd.Timedelta(minutes=5)
    )
)

logger.info(">>> Global simulation parameters")
logger.info("    Cutoff wavelengths, λ₀ (n = %s) = %s nm", len(lambdas), lambdas)
logger.info("    SPECTRL2 inputs: (linspace values = %s)", N)
for key, val in spectrl2_generator_input.items():
    logger.info("     * '%s': %s", key, val)
logger.info("")

# %%
# Simulation and model fitting
# ----------------------------
# Runs the SPECTRL2 bench and fits model(s) for each cutoff wavelength

# what do we want to plot E_λ<λ₀/E against? (None = default behaviour)
plot_keys = ["clearness_index", "absolute_airmass", *spectrl2_generator_input.keys()]
for cutoff_lambda in lambdas:
    # Set output folder per run
    output_folder = base_output_folder.joinpath(f"{cutoff_lambda:04.0f}nm/")
    output_folder.mkdir()
    # Initialization
    bench.reset_simulation_state()
    bench.cutoff_lambda = cutoff_lambda

    bench.constant_params.update(constant_params)
    bench.simulate_from_product(**spectrl2_generator_input)
    bench.plot_ratios_vs_parameters(plot_keys=plot_keys, output_dir=output_folder)
    bench.plot_ratios_vs_components(max_cols=2, output_dir=output_folder)
    bench.plot_usable_vs_full_integrals(max_cols=2, output_dir=output_folder)

    # # test Kt calculation
    # plt.scatter(bench.results["datetimes"], bench.results["clearness_index"])
    # plt.savefig(output_folder.joinpath("kt_over_time.png"))

    ## Fitting model
    logger.info(">>> MODEL FITTING BEGIN")
    logger.info("    Lambda0, λ₀ = %snm", cutoff_lambda)
    model = mr_alike  # E_λ<λ₀/E = f(Kt, am_abs) = c·exp(a(Kt-0.74)+b(am_abs-1.5))
    model_inputs = ["clearness_index", "absolute_airmass"]
    logger.info("    Equation   : %s", model.__doc__)
    logger.info("    Parameters : %s", get_all_params_names(model))
    p0 = (0.75, 1, 1)
    logger.info("    Initial guess p₀ = %s", p0)
    # Fit for each irradiance component
    for irradiance in (
        "poa_global_ratio",
        "poa_direct_ratio",
        "poa_sky_diffuse_ratio",
        "poa_ground_diffuse_ratio",
    ):
        logger.info("  > Fitting parameters for %s", irradiance)
        # Get fitting data
        regressand = bench.results[irradiance]
        regressors = bench.results[model_inputs].to_numpy().T
        # Fit model and get perr metric (see curve_fit docs)
        try:
            popt, pcov = curve_fit(
                model, regressors, regressand, nan_policy="omit", p0=p0
            )
        except RuntimeError:
            logger.error("    curve_fit failed at finding appropiate parameters!")
        else:
            perr = np.sqrt(np.diag(pcov))
            logger.info("    Optimal values = %s", popt)
            logger.info("    Error values   = %s", perr)
    logger.info(">>> MODEL FITTING END")

    continue
    # 3D plot of original and model(s)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ as function of " + ", ".join(model_inputs))
    ax.scatter(regressors[0], regressors[1], regressand, label="Spectrum ratio")
    # ax.scatter(
    #     regressors[0], regressors[1], model(regressors, *popt), label=model.__name__
    # )
    ax.set_xlabel(model_inputs[0])
    ax.set_ylabel(model_inputs[1])
    ax.set_zlabel(r"$\frac{E_{λ<λ_0}}{E}$")
    ax.legend()
    fig.savefig(output_folder.joinpath("Models_3D_lambda.png"))
    plt.show()
