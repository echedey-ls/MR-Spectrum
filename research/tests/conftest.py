import numpy as np


def assert_ascending(array):
    assert (np.diff(array) >= 0).all()
