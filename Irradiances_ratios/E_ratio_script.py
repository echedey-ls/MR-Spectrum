"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

# %% Initialization
from spectrl2_E_ratio_bench import MR_E_ratio
from irradiance_ratios import LAMBDA0

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Matrix of values to test
# Atmosphere characterization required params
N = 3
spectrl2_generator_input = {
    # Wikipedia cites range 870 to 1085 hPa
    "surface_pressure": np.linspace(870.0, 1085.0, N) * 100,  # Pa
    # SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
    "precipitable_water": np.linspace(1.0, 2.5, N),
    # SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
    "aerosol_turbidity_500nm": np.linspace(0.08, 0.30, N),
}

# what do we want to plot E_λ<λ₀/E against? (None = default behaviour)
plot_keys = None

# model function
model_inputs = ["relative_airmass", "aerosol_turbidity_500nm"]
p0 = None  # [0.2, 0.1]


def model(xdata, c0, c1):  # use this func as model template
    r_am, aod500 = xdata
    return c0 * r_am + c1 * aod500


bench = MR_E_ratio(
    datetimes=pd.date_range(
        "2023-11-27T00", "2023-11-28T00", freq=pd.Timedelta(minutes=15)
    )
)

# %%
# Test with monosi/polysi cutoff wavelength
bench.cutoff_lambda = LAMBDA0["monosi"]  # == polysi
bench.simulate_from_product(**spectrl2_generator_input)
bench.plot_results(plot_keys=plot_keys)
optim_result = bench.optimization_from_model(
    model=model, model_inputs=model_inputs, p0=p0
)
bench.times_summary()

# TODO: PLOT RESULTS & MODEL PREDICTION

# %%
# Test with asi cutoff wavelength
bench.reset_simulation_state()
bench.cutoff_lambda = LAMBDA0["asi"]
bench.simulate_from_product(**spectrl2_generator_input)
bench.plot_results(plot_keys=plot_keys)
bench.optimization_from_model(model=model, model_inputs=model_inputs, p0=p0)
bench.times_summary()

# %%
# bench.results.to_csv(
#     f"E_ratio_lambda{bench.cutoff_lambda:04.0f}_"
#     + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#     + ".csv"
# )
