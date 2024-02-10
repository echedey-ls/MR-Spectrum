INPUTS = ("relative_airmass")


def model0(xdata, c0):
    """rel_am * c0"""
    return xdata * c0
