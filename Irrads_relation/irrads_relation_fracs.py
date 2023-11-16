"""
Correlation of usable / broadband spectrum irradiance against distinct variables
================================================================================
"""

from pvlib.spectrum import get_am15g, spectrl2
import numba
import numpy as np

import functools


lambda0 = {  # nm
    "monosi": 1100,
    "polysi": 1100,
    "asi": 800,
}


@numba.njit
def calc_irrad_integrals(
    wavelengths: np.array, irradiances: np.array, cutoff_wavelength: float
):
    r"""
    From a spectrum of wavelengths and their irradiances, calculate
    :math:`E_{\lambda_0 < \lambda} = \int_{\lambda_min}^{\lambda_0} E(\lambda) d\lambda`
    and :math:`E = \int_{\lambda_min}^{+\inf} E(\lambda) d\lambda`.
    This function is optimized by numba.

    Parameters
    ----------
    wavelengths : np.array
        Indexes at which irradiances values are known, usually in nanometers, :math:`nm`.
    irradiances : np.array
        Irradiances values in :math:`\frac{W}{m^2 \cdot nm}`
    cutoff_wavelength : float
        Top wavelength where a PV material has non-zero effectiveness.

    Returns
    -------
    E_entire : float
        :math:`E`
    E_usable : float
        :math:`E_{\lambda_0 < \lambda}`
    """
    # TODO: check behaviour -> should be prev or next index?
    cutoff_lambda_index = np.searchsorted(wavelengths, cutoff_wavelength)
    E_lambda_le = np.trapz(
        irradiances[:cutoff_lambda_index], wavelengths[:cutoff_lambda_index]
    )  # E_{\lambda_0 <= \lambda}
    E_lambda_gt = np.trapz(
        irradiances[cutoff_lambda_index:], wavelengths[cutoff_lambda_index:]
    )  # E_{\lambda_0 > \lambda}
    return (E_lambda_le + E_lambda_gt, E_lambda_le)


@functools.lru_cache(maxsize=10, typed=False)
def G_over_G_lambda(cutoff_nm: float):
    am15g = get_am15g()
    stc_irradiances = am15g.array.to_numpy()
    stc_nanometers = am15g.index.to_numpy()
    stc_complete_integ, stc_usable_integ = calc_irrad_integrals(
        stc_nanometers, stc_irradiances, cutoff_nm
    )
    return (stc_complete_integ, stc_usable_integ)


def E_lambda_over_E(cutoff_nm: float):
    spectrl2()
    pass
