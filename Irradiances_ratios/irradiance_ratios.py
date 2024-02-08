"""
Ratio of usable / broadband spectrum irradiance numeric tools
=============================================================
This file includes:
 * Integrate a spectrum. `calc_irrad_integrals`
 * Get each of N. Martín et all ratios: `G_over_G_lambda` & `E_lambda_over_E`
"""

from pvlib.spectrum import get_am15g
import numpy as np

import functools


LAMBDA0 = {  # Materials different cutoff wavelengths, [nm]
    "monosi": 1100.0,
    "polysi": 1100.0,
    "asi": 800.0,
}


def calc_irrad_integrals(
    wavelengths: np.ndarray, irradiances: np.ndarray, cutoff_wavelength: float
):
    r"""
    From a spectrum of wavelengths and their irradiances, calculate
    :math:`E_{\lambda_0 < \lambda} = \int_{\lambda_min}^{\lambda_0} E(\lambda) d\lambda`
    and :math:`E = \int_{\lambda_min}^{+\infty} E(\lambda) d\lambda`.
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
    # TODO: optimize with numba? Currently can't do trapz(non-uniform dimensions)
    # TODO: check behaviour -> maybe should be prev or next index?
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
    r"""
    Given a cutoff wavelength, get standard spectrum's ratio
    :math:`\frac{\bar{G}}{\bar{G}_{\lambda_0 < \lambda}}`, given
    :math:`\bar{G}_{\lambda_0 < \lambda} = \int_{\lambda_{min}}^{\lambda_0} G(\lambda)
    d\lambda` and :math:`\bar{G} = \int_{\lambda_{min}}^{+\infty} G(\lambda) d\lambda`.

    This function is cached by default.

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
    G_fraction : float
        :math:`\frac{\bar{G}}{\bar{G}_{\lambda_0 < \lambda}}`
    """
    am15g = get_am15g()
    G_irradiances = am15g.array.to_numpy()
    G_nanometers = am15g.index.to_numpy()
    G_complete_integ, G_usable_integ = calc_irrad_integrals(
        G_nanometers, G_irradiances, cutoff_nm
    )
    return G_complete_integ / G_usable_integ


def E_lambda_over_E(cutoff_nm: float, wavelengths: np.ndarray, irradiances: np.ndarray):
    """
    Same as above, but for the :math:`\frac{E_{\lambda_0 < \lambda}}{E}` ratio.
    """  # E_λ<λ₀/E
    E_complete_integ, E_usable_integ = calc_irrad_integrals(
        wavelengths, irradiances, cutoff_nm
    )
    return E_usable_integ / E_complete_integ
