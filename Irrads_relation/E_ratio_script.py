"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

# %% Initialization
import numpy as np

from spectrl2_E_ratio_bench import MR_E_ratio
from irrads_relation_fracs import LAMBDA0

# Matrix of values to test
# Atmosphere characterization required params
N = 4
spectrl2_generator_input = {
    # Wikipedia cites range 870 to 1085 hPa
    "surface_pressure": np.linspace(870.0, 1085.0, N) * 100,  # Pa
    # SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
    "precipitable_water": np.linspace(1.0, 2.5, N),
    # SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
    "aerosol_turbidity_500nm": np.linspace(0.08, 0.30, N),
}

bench = MR_E_ratio()  # default values for a start

# %%
# Test with monosi/polysi cutoff wavelength
bench.cutoff_lambda = LAMBDA0["monosi"]  # == polysi
bench.simulate_from_product(**spectrl2_generator_input)

# %%
# Test with asi cutoff wavelength
bench.cutoff_lambda = LAMBDA0["asi"]
bench.simulate_from_product(**spectrl2_generator_input)
