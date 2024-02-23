from .ratios_calculator import calc_irrad_integrals

import numpy as np
from numpy.testing import assert_approx_equal

from pytest import fixture

@fixture
def spectrum_integral_example():
    points = 50
    lambdas = np.linspace(0, 1000, points)  # nm
    wvPower = np.repeat(1, points)  # W/(nm·m^2)
    integ_total = 1 * (1000 - 0)
    return lambdas, wvPower, integ_total


def test_calc_irrad_integrals(spectrum_integral_example):
    cutoff_lambda = 1000
    lambdas, wvPower, integ_total = spectrum_integral_example
    integ_all, integ_lower = calc_irrad_integrals(lambdas, wvPower, cutoff_lambda)
    assert_approx_equal(integ_all, integ_total, 0)
    assert_approx_equal(integ_lower, integ_total*cutoff_lambda/1000, 0)

    cutoff_lambda = 500
    integ_all, integ_lower = calc_irrad_integrals(lambdas, wvPower, cutoff_lambda)
    assert_approx_equal(integ_all, integ_total, 0)
    assert_approx_equal(integ_lower, integ_total*cutoff_lambda/1000, 0)
