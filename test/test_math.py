import numpy as np
import hypothesis.strategies as strat
from hypothesis import given, assume, note
import utils
import spyct._math as smath
from scipy.special import expit


@given(utils.pair_of_same_size_real_vectors())
def test_vector_dot_vector(vectors):
    vec1, vec2 = vectors
    assert smath.vector_dot_vector(vec1, vec2) == np.dot(vec1, vec2)


@given(utils.pair_of_same_size_real_vectors())
def test_component_sum(vectors):
    vec1, vec2 = vectors
    result = np.empty_like(vec1)
    smath.component_sum(vec1, vec2, result)
    assert np.array_equal(vec1 + vec2, result)


@given(utils.pair_of_same_size_real_vectors())
def test_component_diff(vectors):
    vec1, vec2 = vectors
    result = np.empty_like(vec1)
    smath.component_diff(vec1, vec2, result)
    assert np.array_equal(vec1 - vec2, result)


@given(utils.pair_of_same_size_real_vectors())
def test_component_prod(vectors):
    vec1, vec2 = vectors
    result = np.empty_like(vec1)
    smath.component_prod(vec1, vec2, result)
    assert np.array_equal(vec1 * vec2, result)


@given(utils.pair_of_same_size_real_vectors())
def test_component_div(vectors):
    vec1, vec2 = vectors
    assume(np.all(vec2 != 0))
    result = np.empty_like(vec1)
    smath.component_div(vec1, vec2, result)
    assert np.array_equal(vec1 / vec2, result)


@given(utils.real_ndarrays(utils.random_vector_shapes()), strat.floats(allow_nan=False, allow_infinity=False, width=32))
def test_reset_vector(vector, value):
    smath.reset_vector(vector, value)
    assert np.all(vector == value)


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_vector_sum(vector):
    note('vector shape: {}'.format(vector.shape))
    assert utils.values_almost_equal(vector.sum(), smath.vector_sum(vector))


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_vector_mean(vector):
    note('vector shape: {}'.format(vector.shape))
    assert utils.values_almost_equal(vector.mean(), smath.vector_mean(vector))


@given(utils.real_ndarrays(utils.random_vector_shapes()), strat.floats(allow_nan=False, width=32, min_value=-1e8, max_value=1e8))
def test_vector_scalar_prod(vector, scalar):
    orig = vector.copy()
    smath.vector_scalar_prod(vector, scalar)
    assert np.array_equal(scalar * orig, vector)


@given(utils.real_ndarrays(utils.random_vector_shapes()), strat.floats(allow_nan=False, width=32, min_value=-1e8, max_value=1e8))
def test_vector_scalar_sum(vector, scalar):
    orig = vector.copy()
    smath.vector_scalar_sum(vector, scalar)
    assert np.array_equal(scalar + orig, vector)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()), strat.floats(allow_nan=False, width=32, min_value=-1e8, max_value=1e8))
def test_impute_missing(matrix, value):
    matrix = np.asfortranarray(matrix)
    result = np.nan_to_num(matrix, nan=value)
    smath.impute_missing(matrix, value)
    assert np.array_equal(matrix, result)


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_l1_norm(vector):
    note('vector shape: {}'.format(vector.shape))
    norm = np.linalg.norm(vector, ord=1)
    assert utils.values_almost_equal(norm, smath.l1_norm(vector))


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_l2_norm(vector):
    note('vector shape: {}'.format(vector.shape))
    norm = np.linalg.norm(vector, ord=2)
    assert utils.values_almost_equal(norm, smath.l2_norm(vector))


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_l05_norm(vector):
    note('vector shape: {}'.format(vector.shape))
    norm = np.sum(np.sqrt(np.abs(vector))) ** 2
    assert utils.values_almost_equal(norm, smath.l05_norm(vector))


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_l1_normalize(vector):
    norm = np.linalg.norm(vector, ord=1)
    assume(norm != 0)
    result = vector / norm
    smath.l1_normalize(vector)
    note('vector shape: {}'.format(vector.shape))
    note('target: {}'.format(result))
    note('actual: {}'.format(vector))
    note('diff: {}'.format(np.abs(vector - result)))
    assert utils.arrays_almost_equal(vector, result)


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2)
    assume(norm != 0)
    result = vector / norm
    smath.l2_normalize(vector)
    note('vector shape: {}'.format(vector.shape))
    note('target: {}'.format(result))
    note('actual: {}'.format(vector))
    note('diff: {}'.format(np.abs(vector - result)))
    assert utils.arrays_almost_equal(vector, result)


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_fuzzy_split_hinge(vector):
    large = vector > 1
    small = vector < -1
    selection_derivative_result = 0.5 * np.ones_like(vector)
    selection_derivative_result[large] = 0
    selection_derivative_result[small] = 0

    right_selection_result = vector.copy()
    right_selection_result[large] = 1
    right_selection_result[small] = -1
    right_selection_result = (right_selection_result + 1) / 2
    left_selection_result = 1 - right_selection_result

    right_selection = np.empty_like(vector)
    left_selection = np.empty_like(vector)
    selection_derivative = np.empty_like(vector)
    smath.fuzzy_split_hinge(vector, left_selection, right_selection, selection_derivative)

    note('right_selection: {} vs {}'.format(right_selection, right_selection_result))
    note('left_selection: {} vs {}'.format(left_selection, left_selection_result))
    note('selection_derivative: {} vs {}'.format(selection_derivative, selection_derivative_result))

    assert utils.arrays_almost_equal(right_selection_result, right_selection)
    assert utils.arrays_almost_equal(left_selection_result, left_selection)
    assert utils.arrays_almost_equal(selection_derivative_result, selection_derivative)


@given(utils.real_ndarrays(utils.random_vector_shapes()))
def test_fuzzy_split_sigmoid(vector):
    right_selection_result = expit(vector)
    left_selection_result = 1 - right_selection_result
    selection_derivative_result = right_selection_result * left_selection_result

    right_selection = np.empty_like(vector)
    left_selection = np.empty_like(vector)
    selection_derivative = np.empty_like(vector)
    smath.fuzzy_split_sigmoid(vector, left_selection, right_selection, selection_derivative)

    note('right_selection: {} vs {}'.format(right_selection, right_selection_result))
    note('left_selection: {} vs {}'.format(left_selection, left_selection_result))
    note('selection_derivative: {} vs {}'.format(selection_derivative, selection_derivative_result))

    assert utils.arrays_almost_equal(right_selection_result, right_selection)
    assert utils.arrays_almost_equal(left_selection_result, left_selection)
    assert utils.arrays_almost_equal(selection_derivative_result, selection_derivative)

