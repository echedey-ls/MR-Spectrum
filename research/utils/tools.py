import numpy as np

from datetime import date
import functools
import inspect


@functools.lru_cache(maxsize=2, typed=False)
def check_evenly_spaced_numpy_array(array: np.array):
    np.all(np.diff(array) == np.diff(array)[0])


def day_of_year(d: date, ref_year=None):
    # https://stackoverflow.com/a/25852628
    # https://docs.python.org/3/library/datetime.html#datetime.datetime.timetuple
    yday = d.toordinal() - date(ref_year or d.year, 1, 1).toordinal() + 1
    return yday


def get_all_params_names(func):
    """
    Get parameter names of a callable object.

    Parameters
    ----------
    func : callable

    Returns
    -------
    parameter_names : set
    """
    params = inspect.signature(func).parameters
    return tuple(params.keys())


def get_required_params_names(func):
    """
    Get required parameter names of a callable object.

    Parameters
    ----------
    func : callable

    Returns
    -------
    optional_params : set
    """
    params = inspect.signature(func).parameters
    return (
        param_name
        for param_name, param in params.items()
        if param.default is not inspect.Parameter.empty
    )


def get_optional_params_names(func):
    """
    Get optional parameters names of a callable object

    Parameters
    ----------
    func : callable

    Returns
    -------
    optional_params : set
    """
    params = inspect.signature(func).parameters
    return (
        param_name
        for param_name, param in params.items()
        if param.default is not inspect.Parameter.empty
    )


def get_keyword_params(func):
    """
    Get keyword parameters of a callable object. Excludes ``POSITIONAL_ONLY``
    arguments.

    Parameters
    ----------
    func : callable

    Returns
    -------
    keyword_params : set of parameter names
    """
    params = inspect.signature(func).parameters
    return (
        param_name
        for param_name, param in params.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )


def has_variable_positional_arguments(func):
    """
    Check if ``func`` admits variable positional arguments. Usually ``*args``.

    Parameters
    ----------
    func : callable

    Returns
    -------
    has_var_positional_args : bool
    """
    params = inspect.signature(func).parameters
    return any(
        param.kind is inspect.Parameter.VAR_POSITIONAL for param in params.values()
    )


def has_variable_keyword_arguments(func):
    """
    Check if ``func`` admits variable keyword arguments. Usually ``**kwargs``.

    Parameters
    ----------
    func : callable

    Returns
    -------
    has_var_keyword_args : bool
    """
    params = inspect.signature(func).parameters
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())


COMPONENTS_TRANSLATION_DICT: dict = {
    "global": "G",
    "direct": "B",
    "sky": "D",
    "ground": "A"
}

def component_name_to_character(name: str):
    for comp_name, comp_char in COMPONENTS_TRANSLATION_DICT.items():
        if comp_name in name:
            return comp_char
