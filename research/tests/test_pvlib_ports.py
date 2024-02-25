from research.pvlib_ports import martin_ruiz

import numpy as np
from numpy.testing import assert_approx_equal, assert_allclose
import pandas as pd

import pytest

@pytest.fixture
def martin_ruiz_mismatch_data():
    # Data to run tests of martin_ruiz
    kwargs = {
        'clearness_index': [0.56, 0.612, 0.664, 0.716, 0.768, 0.82],
        'airmass_absolute': [2, 1.8, 1.6, 1.4, 1.2, 1],
        'monosi_expected': {
            'dir': [1.09149, 1.07275, 1.05432, 1.03622, 1.01842, 1.00093],
            'sky': [0.88636, 0.85009, 0.81530, 0.78194, 0.74994, 0.71925],
            'gnd': [1.02011, 1.00465, 0.98943, 0.97444, 0.95967, 0.94513]},
        'polysi_expected': {
            'dir': [1.09166, 1.07280, 1.05427, 1.03606, 1.01816, 1.00058],
            'sky': [0.89443, 0.85553, 0.81832, 0.78273, 0.74868, 0.71612],
            'gnd': [1.02638, 1.00888, 0.99168, 0.97476, 0.95814, 0.94180]},
        'asi_expected': {
            'dir': [1.07066, 1.05643, 1.04238, 1.02852, 1.01485, 1.00136],
            'sky': [0.94889, 0.91699, 0.88616, 0.85637, 0.82758, 0.79976],
            'gnd': [1.03801, 1.02259, 1.00740, 0.99243, 0.97769, 0.96316]},
        'monosi_model_params_dict': {
            'poa_direct': {'c': 1.029, 'a': -3.13e-1, 'b': 5.24e-3},
            'poa_sky_diffuse': {'c': 0.764, 'a': -8.82e-1, 'b': -2.04e-2},
            'poa_ground_diffuse': {'c': 0.970, 'a': -2.44e-1, 'b': 1.29e-2}},
        'monosi_custom_params_df': pd.DataFrame({
            'poa_direct': [1.029, -0.313, 0.00524],
            'poa_sky_diffuse': [0.764, -0.882, -0.0204]},
            index=('c', 'a', 'b'))
    }
    return kwargs


def test_martin_ruiz_mm_scalar(martin_ruiz_mismatch_data):
    # test scalar input ; only module_type given
    clearness_index = martin_ruiz_mismatch_data['clearness_index'][0]
    airmass_absolute = martin_ruiz_mismatch_data['airmass_absolute'][0]
    result = martin_ruiz(clearness_index,
                                  airmass_absolute,
                                  module_type='asi')

    assert_approx_equal(result['poa_direct'],
                        martin_ruiz_mismatch_data['asi_expected']['dir'][0],
                        significant=5)
    assert_approx_equal(result['poa_sky_diffuse'],
                        martin_ruiz_mismatch_data['asi_expected']['sky'][0],
                        significant=5)
    assert_approx_equal(result['poa_ground_diffuse'],
                        martin_ruiz_mismatch_data['asi_expected']['gnd'][0],
                        significant=5)


def test_martin_ruiz_mm_series(martin_ruiz_mismatch_data):
    # test with Series input ; only module_type given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    expected = pd.DataFrame(data={
        'dir': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['dir']),
        'sky': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['sky']),
        'gnd': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['gnd'])})

    result = martin_ruiz(clearness_index, airmass_absolute,
                                  module_type='polysi')
    assert_allclose(result['poa_direct'], expected['dir'], atol=1e-5)
    assert_allclose(result['poa_sky_diffuse'], expected['sky'], atol=1e-5)
    assert_allclose(result['poa_ground_diffuse'], expected['gnd'], atol=1e-5)


def test_martin_ruiz_mm_nans(martin_ruiz_mismatch_data):
    # test NaN in, NaN out ; only module_type given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    airmass_absolute[:5] = np.nan

    result = martin_ruiz(clearness_index, airmass_absolute,
                                  module_type='monosi')
    assert np.isnan(result['poa_direct'][:5]).all()
    assert not np.isnan(result['poa_direct'][5:]).any()
    assert np.isnan(result['poa_sky_diffuse'][:5]).all()
    assert not np.isnan(result['poa_sky_diffuse'][5:]).any()
    assert np.isnan(result['poa_ground_diffuse'][:5]).all()
    assert not np.isnan(result['poa_ground_diffuse'][5:]).any()


def test_martin_ruiz_mm_model_dict(martin_ruiz_mismatch_data):
    # test results when giving 'model_parameters' as dict
    # test custom quantity of components and its names can be given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    expected = pd.DataFrame(data={
        'dir': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['dir']),
        'sky': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['sky']),
        'gnd': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['gnd'])})
    model_parameters = martin_ruiz_mismatch_data['monosi_model_params_dict']

    result = martin_ruiz(
        clearness_index,
        airmass_absolute,
        model_parameters=model_parameters)
    assert_allclose(result['poa_direct'], expected['dir'], atol=1e-5)
    assert_allclose(result['poa_sky_diffuse'], expected['sky'], atol=1e-5)
    assert_allclose(result['poa_ground_diffuse'], expected['gnd'], atol=1e-5)


def test_martin_ruiz_mm_model_df(martin_ruiz_mismatch_data):
    # test results when giving 'model_parameters' as DataFrame
    # test custom quantity of components and its names can be given
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = martin_ruiz_mismatch_data['monosi_custom_params_df']
    expected = pd.DataFrame(data={
        'dir': np.array(martin_ruiz_mismatch_data['monosi_expected']['dir']),
        'sky': np.array(martin_ruiz_mismatch_data['monosi_expected']['sky'])})

    result = martin_ruiz(
        clearness_index,
        airmass_absolute,
        model_parameters=model_parameters)
    assert_allclose(result['poa_direct'], expected['dir'], atol=1e-5)
    assert_allclose(result['poa_sky_diffuse'], expected['sky'], atol=1e-5)
    assert result['poa_ground_diffuse'].isna().all()


def test_martin_ruiz_mm_error_notimplemented(martin_ruiz_mismatch_data):
    # test exception is raised when module_type does not exist in algorithm
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])

    with pytest.raises(NotImplementedError,
                       match='Cell type parameters not defined in algorithm.'):
        _ = martin_ruiz(clearness_index, airmass_absolute,
                                 module_type='')


def test_martin_ruiz_mm_error_model_keys(martin_ruiz_mismatch_data):
    # test exception is raised when  in params keys
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = {
        'component_example': {'z': 0.970, 'x': -2.44e-1, 'y': 1.29e-2}}
    with pytest.raises(ValueError,
                       match="You must specify model parameters with keys "
                             "'a','b','c' for each irradiation component."):
        _ = martin_ruiz(clearness_index, airmass_absolute,
                                 model_parameters=model_parameters)


def test_martin_ruiz_mm_error_missing_params(martin_ruiz_mismatch_data):
    # test exception is raised when missing module_type and model_parameters
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])

    with pytest.raises(ValueError,
                       match='You must pass at least "module_type" '
                             'or "model_parameters" as arguments.'):
        _ = martin_ruiz(clearness_index, airmass_absolute)


def test_martin_ruiz_mm_error_too_many_arguments(martin_ruiz_mismatch_data):
    # test warning is raised with both 'module_type' and 'model_parameters'
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = martin_ruiz_mismatch_data['monosi_model_params_dict']

    with pytest.raises(ValueError,
                       match='Cannot resolve input: must supply only one of '
                             '"module_type" or "model_parameters"'):
        _ = martin_ruiz(clearness_index, airmass_absolute,
                                 module_type='asi',
                                 model_parameters=model_parameters)
