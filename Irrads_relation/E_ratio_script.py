"""
Ratio of usable / broadband spectrum irradiance against distinct variables
==========================================================================
Script to bench the generator algorithm against known cutoff wavelength of
Silicon-based PV cell technologies.
"""

import numpy as np

from spectrl2_E_ratio_bench import MR_E_ratio
from irrads_relation_fracs import LAMBDA0


# Atmosphere characterization required params
N = 4
# Wikipedia cites range 870 to 1085 hPa
surface_pressure = np.linspace(870.0, 1085.0, N) * 100  # Pa
# SPECTRL2 paper Fig 4-6: 1.0 to 2.5 cm
precipitable_water = np.linspace(1.0, 2.5, N)
# SPECTRL2 paper Fig 4-5: 0.08 to 0.30 [unitless]
aerosol_turbidity_500nm = np.linspace(0.08, 0.30, N)


bench = MR_E_ratio()  # default values for a start

# Test for each cutoff wavelength we consider
for unique_cutoff_lambda in set(LAMBDA0.values()):
    bench.cutoff_lambda = unique_cutoff_lambda
    # Preliminarily test SPECTRL2 required params
    bench.simulate_from_product(
        surface_pressure=surface_pressure,
        precipitable_water=precipitable_water,
        aerosol_turbidity_500nm=aerosol_turbidity_500nm,
    )
