"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

# %% Initialization
from Ratios_generators.spectrl2_E_ratio_bench import MR_SPECTRL2_E_ratio_bench
from irradiance_ratios import LAMBDA0
from Models.models_clearness_index_and_airmass_absolute import mr_alike
from utils.tools import get_all_params_names

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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

# Matrix of values to test
# Atmosphere characterization required params
N = 2
spectrl2_generator_input = {
    # SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
    "precipitable_water": np.linspace(1.0, 2.5, N),
    # SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
    "aerosol_turbidity_500nm": np.linspace(0.08, 0.30, N),
}

constant_params = {"surface_pressure": 1013100.0}

# what do we want to plot E_λ<λ₀/E against? (None = default behaviour)
plot_keys = ["clearness_index", "absolute_airmass", *spectrl2_generator_input.keys()]

# bench instance with example time input
bench = MR_SPECTRL2_E_ratio_bench(
    datetimes=pd.date_range(
        "2023-11-27T00", "2023-11-28T00", freq=pd.Timedelta(minutes=5)
    )
)

# %%
for cutoff_lambda in np.unique(np.fromiter(LAMBDA0.values(), dtype=float)):
    # Set output folder per run
    output_folder = base_output_folder.joinpath(f"{cutoff_lambda:04.0f}nm/")
    output_folder.mkdir()
    # Initialization
    bench.reset_simulation_state()
    bench.cutoff_lambda = cutoff_lambda

    bench.constant_params.update(constant_params)
    bench.simulate_from_product(**spectrl2_generator_input)
    bench.plot_results(plot_keys=plot_keys, output_dir=output_folder)
    bench.times_summary()

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
    # Get fitting data
    regressand = bench.results["poa_global_ratio"]
    regressors = bench.results[model_inputs].to_numpy().T
    # Fit model and get perr metric (see curve_fit docs)
    try:
        popt, pcov = curve_fit(model, regressors, regressand, nan_policy="omit", p0=p0)
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

# with open(
#     "outputs/"
#     + "model_fitting_results"
#     + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#     + ".pkl",
#     "wb",
# ) as handle:
#     pickle.dump(saved_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
