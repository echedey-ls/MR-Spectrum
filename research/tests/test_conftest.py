from . import conftest

import pytest


def test_assert_all_ascending():
    conftest.assert_ascending([1, 1, 3, 4, 4])
    with pytest.raises(AssertionError):
        conftest.assert_ascending([1, 0, 3, 4, 5])
