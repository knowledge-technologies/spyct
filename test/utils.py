import numpy as np
import scipy.sparse as sp
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as strat
import warnings

max_dimensions = 100


def clip_matrix_values(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        m = np.nanmax(np.abs(matrix))
    if m > 1e6:
        matrix /= m
    return matrix


def make_csr(matrix):
    clip_matrix_values(matrix)
    matrix -= matrix[0, 0]
    return sp.csr_matrix(matrix)


def arrays_almost_equal(arr1, arr2, eps=1e-4):

    arr1[np.isinf(arr1)] = np.nan
    arr2[np.isinf(arr2)] = np.nan

    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)
    if not np.array_equal(nan1, nan2):
        return False

    arr1[nan1] = 0
    arr2[nan2] = 0
    diff_norm = np.linalg.norm(arr1 - arr2)
    return diff_norm < eps or diff_norm / (np.linalg.norm(arr1) + np.linalg.norm(arr2) + eps) < eps


def values_almost_equal(val1, val2, eps=1e-4):
    return abs(val1 - val2) < eps or abs(val1 - val2) / (abs(val1) + abs(val2) + eps) < eps


def random_vector_shapes():
    return hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=max_dimensions)


def random_matrix_shapes():
    return hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=max_dimensions)


def fixed_vector_shapes(size=100):
    return strat.just((size, ))


def index_ndarrays(shapes):
    return hnp.arrays(dtype=np.dtype(np.intp), shape=shapes, unique=True,
                      elements=strat.integers(min_value=0, max_value=max_dimensions))


def real_ndarrays(shapes):
    return hnp.arrays(dtype=np.dtype(np.float32), shape=shapes,
                      elements=strat.floats(min_value=-1e6, max_value=1e6, width=32, allow_nan=False))


def real_ndarrays_nan(shapes):
    return hnp.arrays(dtype=np.dtype(np.float32), shape=shapes,
                      elements=strat.floats(width=32, allow_nan=True, allow_infinity=False))


@strat.composite
def pair_of_same_size_real_vectors(draw):
    shape = draw(random_vector_shapes())
    v1 = draw(real_ndarrays(shape))
    v2 = draw(real_ndarrays(shape))
    return v1, v2


@strat.composite
def matrix_vector_pair(draw):
    shape = draw(random_matrix_shapes())
    matrix = draw(real_ndarrays(shape))
    vector = draw(real_ndarrays((shape[1], )))
    return matrix, vector


@strat.composite
def vector_matrix_pair(draw):
    shape = draw(random_matrix_shapes())
    matrix = draw(real_ndarrays(shape))
    vector = draw(real_ndarrays((shape[0], )))
    return vector, matrix


@strat.composite
def matrix_slice(draw, dim):
    shape = draw(random_matrix_shapes())
    matrix = draw(real_ndarrays_nan(shape))
    indices = draw(hnp.arrays(dtype=np.dtype(np.intp), shape=(shape[dim], ), unique=True,
                              elements=strat.integers(min_value=0, max_value=shape[dim]-1)))
    indices.sort()
    return matrix, indices
