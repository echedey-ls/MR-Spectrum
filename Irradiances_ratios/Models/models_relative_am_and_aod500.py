import numpy as np

INPUTS = ("relative_airmass", "aerosol_turbidity_500nm")


def model0(xdata, c0, c1, c2):
    "c0 * r_am + c1 * aod500"
    r_am, aod500 = xdata
    return c0 * r_am + c1 * aod500 + c2


def model1(xdata, c0, c1, c2, c3):
    "c0 * np.exp(c1 * r_am + c2 * aod500)"
    r_am, aod500 = xdata
    return c0 * np.exp(c1 * r_am + c2 * aod500 + c3)


def model2(xdata, c0, c1, c2, c3, c4):
    "c0 * np.exp(c1 * r_am) + c2 * np.exp(c3 * aod500)"
    r_am, aod500 = xdata
    return c0 * np.exp(c1 * r_am) + c2 * np.exp(c3 * aod500) + c4
