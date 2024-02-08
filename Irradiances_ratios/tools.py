import numpy as np
import numba

from datetime import date
import functools
import inspect


@functools.lru_cache(maxsize=2, typed=False)
@numba.njit
def check_evenly_spaced_numpy_array(array: np.array):
    np.all(np.diff(array) == np.diff(array)[0])


def day_of_year(d: date, ref_year=None):
    # https://stackoverflow.com/a/25852628
    # https://docs.python.org/3/library/datetime.html#datetime.datetime.timetuple
    yday = d.toordinal() - date(ref_year or d.year, 1, 1).toordinal() + 1
    return yday


def _get_optional_params(func):
    """
    Get optional parameters of a callable objects

    Parameters
    ----------
    funcs : callable
    Returns
    -------
    optional_params : set
    """
    params = inspect.signature(func).parameters
    return {
        param_name
        for param_name, param in params.items()
        if param.default is not inspect.Parameter.empty
    }
