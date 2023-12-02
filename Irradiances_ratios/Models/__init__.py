"""
E_λ<λ₀/E model database
=======================
The idea is to iterate over them, with the parameters as a key.
All models should start with "model" to be included in ``MODELS_BY_PARAMS``.
``MODELS_BY_PARAMS`` is a ``dict`` in the form ``dict[str, tuple[Callable, ...]]``,
where the key represents the variables required by the model and the value is a tuple
made of each of the models that take those variables as arguments.
"""
from . import models_relative_am
from . import models_relative_am_and_aod500

from inspect import getmembers, isfunction
from typing import Callable


def _get_models_and_inputs_from_modules(*modules):
    models_by_params = {}
    for module in modules:
        models_by_params[", ".join(module.INPUTS)] = tuple(
            [
                f_obj
                for _, f_obj in getmembers(
                    module, lambda f: isfunction(f) and f.__name__.startswith("model")
                )
            ]
        )
    return models_by_params


MODELS_BY_PARAMS: dict[str, tuple[Callable, ...]] = _get_models_and_inputs_from_modules(
    models_relative_am, models_relative_am_and_aod500
)
