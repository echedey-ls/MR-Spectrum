"""
This file adds the Sef / bar{Sef} model developed by Nuria Martín and José
María Ruiz.
This was proposed to PVLIB in PR#1658,
https://github.com/pvlib/pvlib-python/pull/1658
"""

import numpy as np
import pandas as pd


def martin_ruiz(clearness_index, airmass_absolute, module_type=None,
                model_parameters=None):
    r"""
    Calculate spectral mismatch modifiers for POA direct, sky diffuse and
    ground diffuse irradiances using the clearness index and the absolute
    airmass.

    .. warning::
        Included model parameters for ``monosi``, ``polysi`` and ``asi`` were
        estimated using the airmass model ``kasten1966`` [1]_.
        The same airmass model *must* be used to calculate the airmass input
        values to this function in order to not introduce errors.
        See :py:func:`~pvlib.atmosphere.get_relative_airmass`.

    Parameters
    ----------
    clearness_index : numeric
        Clearness index of the sky.

    airmass_absolute : numeric
        Absolute airmass. ``kasten1966`` airmass algorithm must be used
        for default parameters of ``monosi``, ``polysi`` and ``asi``,
        see [1]_.

    module_type : string, optional
        Specifies material of the cell in order to infer model parameters.
        Allowed types are ``monosi``, ``polysi`` and ``asi``, either lower or
        upper case. If not specified, ``model_parameters`` has to be provided.

    model_parameters : dict-like, optional
        Provide either a dict or a ``pd.DataFrame`` as follows:

        .. code-block:: python

            # Using a dict
            # Return keys are the same as specifying 'module_type'
            model_parameters = {
                'poa_direct': {'c': c1, 'a': a1, 'b': b1},
                'poa_sky_diffuse': {'c': c2, 'a': a2, 'b': b2},
                'poa_ground_diffuse': {'c': c3, 'a': a3, 'b': b3}
            }
            # Using a pd.DataFrame
            model_parameters = pd.DataFrame({
                'poa_direct': [c1, a1, b1],
                'poa_sky_diffuse':  [c2, a2, b2],
                'poa_ground_diffuse': [c3, a3, b3]},
                index=('c', 'a', 'b'))

        ``c``, ``a`` and ``b`` must be scalar.

        Unspecified parameters for an irradiance component (`'poa_direct'`,
        `'poa_sky_diffuse'`, or `'poa_ground_diffuse'`) will cause ``np.nan``
        to be returned in the corresponding result.

    Returns
    -------
    Modifiers : pd.DataFrame (iterable input) or dict (scalar input) of numeric
        Mismatch modifiers for direct, sky diffuse and ground diffuse
        irradiances, with indexes `'poa_direct'`, `'poa_sky_diffuse'`,
        `'poa_ground_diffuse'`.
        Each mismatch modifier should be multiplied by its corresponding
        POA component.

    Raises
    ------
    ValueError
        If ``model_parameters`` is not suitable. See examples given above.
    ValueError
        If neither ``module_type`` nor ``model_parameters`` are given.
    ValueError
        If both ``module_type`` and ``model_parameters`` are provided.
    NotImplementedError
        If ``module_type`` is not found in internal table of parameters.

    Notes
    -----
    The mismatch modifier is defined as

    .. math:: M = c \cdot \exp( a \cdot (K_t - 0.74) + b \cdot (AM - 1.5) )

    where :math:`c`, :math:`a` and :math:`b` are the model parameters,
    different for each irradiance component.

    References
    ----------
    .. [1] Martín, N. and Ruiz, J.M. (1999), A new method for the spectral
       characterisation of PV modules. Prog. Photovolt: Res. Appl., 7: 299-310.
       :doi:`10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0`

    See Also
    --------
    pvlib.irradiance.clearness_index
    pvlib.atmosphere.get_relative_airmass
    pvlib.atmosphere.get_absolute_airmass
    pvlib.atmosphere.first_solar
    """
    # Note tests for this function are prefixed with test_martin_ruiz_mm_*

    IRRAD_COMPONENTS = ('poa_direct', 'poa_sky_diffuse', 'poa_ground_diffuse')
    # Fitting parameters directly from [1]_
    MARTIN_RUIZ_PARAMS = pd.DataFrame(
        index=('monosi', 'polysi', 'asi'),
        columns=pd.MultiIndex.from_product([IRRAD_COMPONENTS,
                                           ('c', 'a', 'b')]),
        data=[  # Direct(c,a,b)  | Sky diffuse(c,a,b) | Ground diffuse(c,a,b)
            [1.029, -.313, 524e-5, .764, -.882, -.0204, .970, -.244, .0129],
            [1.029, -.311, 626e-5, .764, -.929, -.0192, .970, -.270, .0158],
            [1.024, -.222, 920e-5, .840, -.728, -.0183, .989, -.219, .0179],
        ])

    # Argument validation and choose components and model parameters
    if module_type is not None and model_parameters is None:
        # Infer parameters from cell material
        module_type_lower = module_type.lower()
        if module_type_lower in MARTIN_RUIZ_PARAMS.index:
            _params = MARTIN_RUIZ_PARAMS.loc[module_type_lower]
        else:
            raise NotImplementedError('Cell type parameters not defined in '
                                      'algorithm. Allowed types are '
                                      f'{tuple(MARTIN_RUIZ_PARAMS.index)}')
    elif model_parameters is not None and module_type is None:
        # Use user-defined model parameters
        # Validate 'model_parameters' sub-dicts keys
        if any([{'a', 'b', 'c'} != set(model_parameters[component].keys())
                for component in model_parameters.keys()]):
            raise ValueError("You must specify model parameters with keys "
                             "'a','b','c' for each irradiation component.")
        _params = model_parameters
    elif module_type is None and model_parameters is None:
        raise ValueError('You must pass at least "module_type" '
                         'or "model_parameters" as arguments.')
    elif model_parameters is not None and module_type is not None:
        raise ValueError('Cannot resolve input: must supply only one of '
                         '"module_type" or "model_parameters"')

    if np.isscalar(clearness_index) and np.isscalar(airmass_absolute):
        modifiers = dict(zip(IRRAD_COMPONENTS, (np.nan,)*3))
    else:
        modifiers = pd.DataFrame(columns=IRRAD_COMPONENTS)

    # Compute difference here to avoid recalculating inside loop
    kt_delta = clearness_index - 0.74
    am_delta = airmass_absolute - 1.5

    # Calculate mismatch modifier for each irradiation
    for irrad_type in IRRAD_COMPONENTS:
        # Skip irradiations not specified in 'model_params'
        if irrad_type not in _params.keys():
            continue
        # Else, calculate the mismatch modifier
        _coeffs = _params[irrad_type]
        modifier = _coeffs['c'] * np.exp(_coeffs['a'] * kt_delta
                                         + _coeffs['b'] * am_delta)
        modifiers[irrad_type] = modifier

    return modifiers
