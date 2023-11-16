import numpy as np
import numba

from datetime import date
import functools

@functools.lru_cache(maxsize=2, typed=False)
@numba.njit
def check_evenly_spaced_numpy_array(array: np.array):
    np.all(np.diff(array) == np.diff(array)[0])


def day_of_year(d: date):
    # https://stackoverflow.com/a/25852628
    # https://docs.python.org/3/library/datetime.html#datetime.datetime.timetuple
    yday = d.toordinal() - date(d.year, 1, 1).toordinal() + 1
    return yday
