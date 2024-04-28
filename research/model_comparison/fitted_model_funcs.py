"""Instances of fitted models"""

from research.irradiances_ratios.ratios_calculator import LAMBDA0

import numpy as np
import pandas as pd


def mr_alike_inst(
    clearness_index, airmass_absolute, module_type:str=None, component_select=None
):
    # components
    IRRAD_COMPONENTS = (
        "poa_global", "poa_direct", "poa_sky_diffuse", "poa_ground_diffuse",
    )

    ECHEDEY_PARAMS = pd.DataFrame(
        index=pd.MultiIndex.from_product([(800.0, 1100.0), IRRAD_COMPONENTS]),
        columns=("c", "a", "b"),
        data=[  # [ c | a | b ]
            # λ₀ = 800nm (asi)
            [0.44529369, 0.149695, -0.01518124],  # poa_global
            [4.72919551e-20, .00000354e+00, .00000000e+00],  # poa_direct
            [0.96450068, .15149651, .01127807],  # poa_sky_diffuse
            [4.74678806e-01, 6.12130156e-01, 1.97556457e-04],  # poa_ground_diffuse
            # λ₀ = 1100nm (monosi/polysi)
            [0.72195231, 0.07684853, 0.02505114],  # poa_global
            [1.99765180e-16, .00670533e+00, .00000395e+00],  # poa_direct
            [1.02297009, .43289747, .00263201],  # poa_sky_diffuse
            [0.76003905, 0.10538212, 0.00317642],  # poa_ground_diffuse
        ],
    )

    if component_select is None:
        component_select = IRRAD_COMPONENTS
    elif isinstance(component_select, str):
        component_select = (component_select,)
    if not all(component in IRRAD_COMPONENTS for component in component_select):
        raise ValueError(
            "Unknown component in 'component_select'."
            f"Must be any of {IRRAD_COMPONENTS} or an iterable."
        )

    _lambda_0 = LAMBDA0[module_type.lower()]
    _params = ECHEDEY_PARAMS.loc[_lambda_0]

    # Compute difference
    kt_delta = clearness_index - 0.74
    am_delta = airmass_absolute - 1.5

    # Output values
    modifiers = pd.DataFrame(columns=IRRAD_COMPONENTS)

    # Calculate mismatch modifier for each irradiation
    for irrad_type in component_select:
        _coeffs = _params.loc[irrad_type]
        modifier = _coeffs["c"] * np.exp(
            _coeffs["a"] * kt_delta + _coeffs["b"] * am_delta
        )
        modifiers[irrad_type] = modifier

    return modifiers
