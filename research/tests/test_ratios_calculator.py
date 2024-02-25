from research.irradiances_ratios.ratios_calculator import (
    LAMBDA0,
    calc_irrad_integrals,
    G_over_G_lambda,
    spectrum_integrals_and_ratio,
)

import numpy as np
from numpy.testing import assert_approx_equal

import pytest


@pytest.fixture
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
    assert_approx_equal(integ_lower, integ_total * cutoff_lambda / 1000, 0)

    cutoff_lambda = 500
    integ_all, integ_lower = calc_irrad_integrals(lambdas, wvPower, cutoff_lambda)
    assert_approx_equal(integ_all, integ_total, 0)
    assert_approx_equal(integ_lower, integ_total * cutoff_lambda / 1000, 0)


def test_G_over_G_lambda():
    for _, cutoff_lambda in LAMBDA0.items():
        assert G_over_G_lambda(cutoff_lambda) > 1.0


@pytest.mark.parametrize("cutoff_lambda", (1000, 500, 250))
def test_spectrum_integrals_and_ratio(cutoff_lambda, spectrum_integral_example):
    wavelengths, wvPower, integ_total = spectrum_integral_example
    integ_all, integ_usable, ratio = spectrum_integrals_and_ratio(
        cutoff_lambda, wavelengths, wvPower
    )
    assert_approx_equal(integ_all, integ_total, 0)
    assert_approx_equal(integ_usable, integ_total * cutoff_lambda / 1000, 0)
    # checks equality only of first decimal -> the example data is too rough
    assert_approx_equal(ratio, cutoff_lambda/integ_total, 1)
