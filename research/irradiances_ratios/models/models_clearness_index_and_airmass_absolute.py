import numpy as np

INPUTS = ("clearness_index", "airmass_absolute")

def mr_alike(xdata, c, a, b):
    "E_λ<λ₀/E = f(Kt, am_abs) = c·exp(a(Kt-0.74)+b(am_abs-1.5))"
    Kt, am = xdata
    return c * np.exp(a * (Kt - 0.74) + b * (am - 1.5))
