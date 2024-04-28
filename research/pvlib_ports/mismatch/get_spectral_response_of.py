import pandas as pd
from pathlib import Path

def get_spectral_response_of_material(material, wavelengths=None):
    """
    Read spectral response curve of a material from the database [1]_,
    optionally interpolated to the specified wavelength(s).

    Parameters
    ----------
    material : str
        The material name of the spectral response curve to read.
        The names are ``'monosi'``, ``'polysi'``, ``'hit'`` and ``'all'``.
    wavelengths : numeric, optional
        The wavelengths at which the spectral response is interpolated.
        By default the 181 available wavelengths are returned. :math:`[nm]`

    Returns
    -------
    spectral_response : pandas.Series or pandas.DataFrame
        The returned spectral response curve, indexed by ``wavelengths`` if
        provided. If not, the curve is indexed by the 181 default wavelengths.

    Notes
    -----
    The original dataset is available in [1]_.

    Wavelengths outside of the range :math:`[300, 1200]` will result in
    NaN values.

    Examples
    --------
    >>> print(get_spectral_response_of_material("monosi").head())
    >>> wavelength
    >>> 300    0.000000
    >>> 305    0.046272
    >>> 310    0.088575
    >>> 315    0.127410
    >>> 320    0.163710
    >>> Name: monosi, dtype: float64

    References
    ----------
    .. [1] A. Driesse, M. Theristis, and J. Stein, “PV module spectral response
       measurements - Data and Resources.” EMN-DURMAT (EMN-DuraMAT); Sandia
       National Laboratories (SNL-NM), Albuquerque, NM (United States), 2023.
       :doi:`10.21948/2204677`.
       Available: https://www.osti.gov/servlets/purl/2204677/
    """

    sr_dataset_path = Path(__file__).parent.joinpath(
        "data", "duramat_spectral_responses.csv"
    )
    dataset = pd.read_csv(sr_dataset_path, index_col=0)
    dataset.name = "spectral_response"

    if material in dataset.columns:
        dataset = dataset[material]
    elif material != "all":  # if material=="all", return the whole dataset
        raise ValueError(
            f"Material '{material}' not found in dataset.\n"
            + f"Available materials are {', '.join((*dataset.columns, 'all'))}"
        )

    if wavelengths is not None:
        dataset = (
            # fill with NaNs where we want to interpolate
            dataset.reindex(wavelengths, method=None)
            .interpolate(  # interpolate those NaN values
                method="linear",
                limit_area="inside",
            )
            .loc[wavelengths]  # keep only the requested wavelengths
            .fillna(0.0)  # fill out-of-bounds NaNs with zeros
        )

    return dataset
