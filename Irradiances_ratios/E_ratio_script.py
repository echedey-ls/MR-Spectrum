"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

# %% Initialization
from spectrl2_E_ratio_bench import MR_E_ratio
from irradiance_ratios import LAMBDA0
from Models import MODELS_BY_PARAMS

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pickle

# Matrix of values to test
# Atmosphere characterization required params
N = 5
spectrl2_generator_input = {
    # SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
    "precipitable_water": np.linspace(1.0, 2.5, N),
    # SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
    "aerosol_turbidity_500nm": np.linspace(0.08, 0.30, N),
}

constant_params = {"surface_pressure": 1013100.0}

# what do we want to plot E_λ<λ₀/E against? (None = default behaviour)
plot_keys = None

# model function
p0 = None  # [0.2, 0.1]
model_inputs = ("relative_airmass", "aerosol_turbidity_500nm")
bench = MR_E_ratio(
    datetimes=pd.date_range(
        "2023-11-27T00", "2023-11-28T00", freq=pd.Timedelta(minutes=15)
    )
)

# %%
# Test matrix & save values
#   * For different cutoff wavelengths / materials
#   * For different models

saved_results_dict: dict[
    str,  # lambda0
    dict[
        str,  # model name
        tuple[
            tuple[float, ...],  # popt coeffs
            tuple[float, ...],  # perr coeffs
        ],
    ],
] = {}

for cutoff_lambda in np.unique(np.fromiter(LAMBDA0.values(), dtype=float)):
    # Initialization
    bench.reset_simulation_state()
    bench.cutoff_lambda = cutoff_lambda
    saved_results_dict[cutoff_lambda] = {}

    bench.constant_params.update(constant_params)
    bench.simulate_from_product(**spectrl2_generator_input)
    bench.plot_results(plot_keys=plot_keys)

    # Get fitting data
    regressand = bench.results["poa_global"]
    regressors = bench.results[list(model_inputs)].to_numpy().T
    bench.times_summary()

    # 3D plot of original and models
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(r"$\frac{E_{λ<λ_0}}{E}$ as function of " + ", ".join(model_inputs))
    ax.scatter(regressors[0], regressors[1], regressand, label="Spectrum ratio")
    for model in MODELS_BY_PARAMS[", ".join(model_inputs)]:
        popt, pcov = curve_fit(model, regressors, regressand, nan_policy="omit")
        perr = np.sqrt(np.diag(pcov))
        saved_results_dict[cutoff_lambda][model.__name__] = (popt, perr)
        ax.scatter(
            regressors[0], regressors[1], model(regressors, *popt), label=model.__name__
        )
    ax.set_xlabel(model_inputs[0])
    ax.set_ylabel(model_inputs[1])
    ax.set_zlabel(r"$\frac{E_{λ<λ_0}}{E}$")
    ax.legend()
    fig.savefig(
        f"Models_3D_lambda{cutoff_lambda:04.0f}_"
        + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        + ".png"
    )
    plt.show()

with open(
    "model_fitting_results" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".pkl",
    "wb",
) as handle:
    pickle.dump(saved_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
