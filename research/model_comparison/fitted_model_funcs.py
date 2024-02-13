"""Instances of fitted models"""

from irradiances_ratios.ratios_calculator import LAMBDA0

import numpy as np
import pandas as pd


def mr_alike_inst(
    clearness_index, airmass_absolute, module_type:str=None, component_select=None
):
    # components
    IRRAD_COMPONENTS = (
        "poa_global",  # "poa_direct", # "poa_sky_diffuse", # "poa_ground_diffuse",
    )

    ECHEDEY_PARAMS = pd.DataFrame(
        index=pd.MultiIndex.from_product([(1100.0, 800.0), IRRAD_COMPONENTS]),
        columns=("c", "a", "b"),
        data=[  # [c | a | b]
            # λ₀ = 1100nm (monosi/polysi)
            [0.72379846, -0.06487852, -0.02485946],  # Global
            # λ₀ = 800nm (asi)
            [0.45153287, 0.21800605, -0.0138813],  # Global
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

    # Output values initialization depending on scalar/vector input
    if np.isscalar(clearness_index) and np.isscalar(airmass_absolute):
        modifiers = dict(zip(IRRAD_COMPONENTS, (np.nan,) * 3))
    else:
        modifiers = pd.DataFrame(columns=IRRAD_COMPONENTS)
    # Calculate mismatch modifier for each irradiation
    for irrad_type in component_select:
        _coeffs = _params.loc[irrad_type]
        modifier = _coeffs["c"] * np.exp(
            _coeffs["a"] * kt_delta + _coeffs["b"] * am_delta
        )
        modifiers[irrad_type] = modifier

    return modifiers
