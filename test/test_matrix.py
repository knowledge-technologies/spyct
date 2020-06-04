import pytest
import numpy as np
from hypothesis import given, assume, note, settings, HealthCheck
import utils
import spyct._matrix as smatrix


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_ndarray_to_DMatrix_to_ndarray(matrix):
    result = smatrix.ndarray_to_DMatrix(matrix).to_ndarray()
    note('original: {}'.format(matrix))
    note('result: {}'.format(result))
    np.testing.assert_equal(result, matrix)


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_csr_to_SMatrix_to_csr(matrix):
    csr = utils.make_csr(matrix)
    result = smatrix.csr_to_SMatrix(csr).to_csr()
    np.testing.assert_equal(result.data, csr.data)
    np.testing.assert_equal(result.indptr, csr.indptr)
    np.testing.assert_equal(result.indices, csr.indices)


@given(utils.matrix_vector_pair())
def test_multiply_sparse_sparse(pair):
    matrix, vector = pair
    n = vector.shape[0]
    indices = np.random.choice(n, size=n//2, replace=False).astype(np.intp)
    indices.sort()
    smat = utils.make_csr(matrix)
    svec = np.zeros_like(vector)
    svec[indices] = vector[indices]
    target = np.matmul(smat.A, svec)
    note('matrix: {}'.format(smat.A))
    note('vector: {}'.format(vector))
    note('indices: {}'.format(indices))
    note('target: {}'.format(target))
    scores = np.empty(matrix.shape[0], dtype='f')
    smatrix.multiply_sparse_sparse(smatrix.csr_to_SMatrix(smat),
                                   smatrix.memview_to_SMatrix(vector[indices], n, indices),
                                   scores)
    note('scores: {}'.format(scores))
    assert utils.arrays_almost_equal(target, scores)


### DMatrix tests ################################################################

@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_copy(matrix):
    copy = smatrix.ndarray_to_DMatrix(matrix).copy().to_ndarray()
    np.testing.assert_equal(matrix, copy)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(utils.matrix_slice(0))
def test_DMatrix_take_rows(matrix_rows):
    matrix, rows = matrix_rows
    result = smatrix.ndarray_to_DMatrix(matrix).take_rows(rows).to_ndarray()
    note('target: {}'.format(matrix[rows]))
    note('result: {}'.format(result))
    np.testing.assert_equal(matrix[rows], result)


@given(utils.matrix_slice(1))
def test_DMatrix_take_columns(matrix_columns):
    matrix, columns = matrix_columns
    result = smatrix.ndarray_to_DMatrix(matrix).take_columns(columns).to_ndarray()
    note('target: {}'.format(matrix[:, columns]))
    note('result: {}'.format(result))
    np.testing.assert_equal(matrix[:, columns], result)


@given(utils.matrix_vector_pair())
def test_DMatrix_self_dot_vector(pair):
    matrix, vector = pair
    result = np.empty(shape=matrix.shape[0], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).self_dot_vector(vector, result)
    note('result: {} target:{}'.format(result, matrix.dot(vector)))
    assert utils.arrays_almost_equal(result, matrix.dot(vector))


@given(utils.vector_matrix_pair())
def test_DMatrix_vector_dot_self(pair):
    vector, matrix = pair
    result = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).vector_dot_self(vector, result)
    note('result: {} target:{}'.format(result, vector.dot(matrix)))
    assert utils.arrays_almost_equal(result, vector.dot(matrix))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_column_means_nan(matrix):
    matrix = utils.clip_matrix_values(matrix)
    means = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).column_means_nan(means)
    matrix = matrix.astype('d')
    target_means = np.nan_to_num(np.nanmean(matrix, axis=0))
    note('real: {}  target: {}'.format(means, target_means))
    assert utils.arrays_almost_equal(means, target_means, eps=1e-2)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_column_means(matrix):
    means = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).column_means(means)
    matrix = matrix.astype('d')
    target_means = np.mean(matrix, axis=0)
    note('real: {}  target: {}'.format(means, target_means))
    assert utils.arrays_almost_equal(means, target_means, eps=1e-2)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_column_stds_nan(matrix):
    matrix = utils.clip_matrix_values(matrix)
    means = np.empty(shape=matrix.shape[1], dtype='f')
    stds = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).column_stds_nan(0, means, stds)
    matrix = matrix.astype('d')
    target_means = np.nan_to_num(np.nanmean(matrix, axis=0))
    target_stds = np.nan_to_num(np.nanstd(matrix, axis=0))
    note('means real: {}  target: {}'.format(means, target_means))
    note('stds real: {}  target: {}'.format(stds, target_stds))
    assert utils.arrays_almost_equal(means, target_means, eps=1e-2)
    assert utils.arrays_almost_equal(stds, target_stds, eps=1e-2)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_column_stds(matrix):
    means = np.empty(shape=matrix.shape[1], dtype='f')
    stds = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).column_stds(0, means, stds)
    matrix = matrix.astype('d')
    note('means real: {}  target: {}'.format(means, np.mean(matrix, axis=0)))
    note('stds real: {}  target: {}'.format(stds, np.std(matrix, axis=0)))
    assert utils.arrays_almost_equal(means, np.mean(matrix, axis=0), eps=1e-2)
    assert utils.arrays_almost_equal(stds, np.std(matrix, axis=0), eps=1e-2)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_impute_missing(matrix):
    target = np.nan_to_num(matrix, nan=10)
    result = smatrix.ndarray_to_DMatrix(matrix)
    result.impute_missing(10)
    result = result.to_ndarray()
    assert utils.arrays_almost_equal(target, result)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_nonmissing_matrix(matrix):
    target = (~np.isnan(matrix)).astype('f')
    result = smatrix.ndarray_to_DMatrix(matrix).nonmissing_matrix().to_ndarray()
    assert utils.arrays_almost_equal(target, result)


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_unstandardize_inverts_standardize(matrix):
    means = np.empty(shape=matrix.shape[1], dtype='f')
    stds = np.empty(shape=matrix.shape[1], dtype='f')
    m = smatrix.ndarray_to_DMatrix(matrix)
    m.column_stds(1, means, stds)
    m.standardize_columns(means, stds)
    m.unstandardize_columns(means, stds)
    result = m.to_ndarray()
    note('result: {}'.format(result))
    assert utils.arrays_almost_equal(matrix, result)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_min_nonnan_in_column(matrix):
    result = smatrix.ndarray_to_DMatrix(matrix).min_nonnan_in_column()
    target = matrix.shape[0] - np.isnan(matrix).sum(axis=0).max()
    note('result: {} target: {}'.format(result, target))
    assert utils.values_almost_equal(result, target)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_DMatrix_row_vector(matrix):
    m = smatrix.ndarray_to_DMatrix(matrix)
    for row in range(matrix.shape[0]):
        assert utils.arrays_almost_equal(np.asarray(m.row_vector(row)), matrix[row])


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_cluster_rows_mse(matrix):
    assume(matrix.shape[0] >= 2)
    c0 = matrix[0]
    c1 = matrix[1]

    d0 = np.sqrt(np.sum((matrix - c0) * (matrix - c0), axis=1))
    d1 = np.sqrt(np.sum((matrix - c1) * (matrix - c1), axis=1))
    target = (d1 <= d0).astype('f')

    result = np.empty(matrix.shape[0], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).cluster_rows_mse(c0, c1, result)

    note('result: {}'.format(result))
    note('target: {}'.format(target))
    assert utils.arrays_almost_equal(target, result)


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_cluster_rows_dot(matrix):
    assume(matrix.shape[0] >= 2)
    c0 = matrix[0]
    c1 = matrix[1]

    s0 = matrix.dot(c0) / (np.linalg.norm(c0) + 1e-8)
    s1 = matrix.dot(c1) / (np.linalg.norm(c1) + 1e-8)
    target = (s1 >= s0).astype('f')
    note(str(s0))
    note(str(s1))

    result = np.empty(matrix.shape[0], dtype='f')
    smatrix.ndarray_to_DMatrix(matrix).cluster_rows_dot(c0, c1, result, 1e-4, np.empty(matrix.shape[0], dtype='f'))

    note('result: {}'.format(result))
    note('target: {}'.format(target))
    assert utils.arrays_almost_equal(target, result)


### SMatrix tests ################################################################


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_SMatrix_copy(matrix):
    csr = utils.make_csr(matrix)
    copy = smatrix.csr_to_SMatrix(csr).copy().to_csr()
    np.testing.assert_equal(csr.A, copy.A)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(utils.matrix_slice(0))
def test_SMatrix_take_rows(matrix_rows):
    matrix, rows = matrix_rows
    matrix = utils.make_csr(matrix)
    result = smatrix.csr_to_SMatrix(matrix).take_rows(rows).to_csr()
    note('target: {}'.format(matrix[rows].A))
    note('result: {}'.format(result.A))
    np.testing.assert_equal(matrix[rows].A, result.A)


@given(utils.matrix_slice(1))
def test_SMatrix_take_columns(matrix_columns):
    matrix, columns = matrix_columns
    matrix = utils.make_csr(matrix)
    result = smatrix.csr_to_SMatrix(matrix).take_columns(columns).to_csr()
    note('target: {}'.format(matrix[:, columns].A))
    note('result: {}'.format(result.A))
    np.testing.assert_equal(matrix[:, columns].A, result.A)


@given(utils.matrix_vector_pair())
def test_SMatrix_self_dot_vector(pair):
    matrix, vector = pair
    matrix = utils.make_csr(matrix)
    result = np.empty(shape=matrix.shape[0], dtype='f')
    smatrix.csr_to_SMatrix(matrix).self_dot_vector(vector, result)
    target = matrix.A.dot(vector)
    note('result: {} target:{}'.format(result, target))
    assert utils.arrays_almost_equal(result, target)


@given(utils.vector_matrix_pair())
def test_SMatrix_vector_dot_self(pair):
    vector, matrix = pair
    matrix = utils.make_csr(matrix)
    result = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.csr_to_SMatrix(matrix).vector_dot_self(vector, result)
    target = vector.dot(matrix.A)
    note('result: {} target:{}'.format(result, target))
    assert utils.arrays_almost_equal(result, target)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_SMatrix_column_means(matrix):
    matrix = utils.make_csr(matrix)
    means = np.empty(shape=matrix.shape[1], dtype='f')
    smatrix.csr_to_SMatrix(matrix).column_means(means)
    matrix = matrix.A.astype('d')
    target_means = np.mean(matrix, axis=0)
    note('real: {}  target: {}'.format(means, target_means))
    assert utils.arrays_almost_equal(means, target_means, eps=1e-2)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_SMatrix_column_stds(matrix):
    means = np.empty(shape=matrix.shape[1], dtype='f')
    stds = np.empty(shape=matrix.shape[1], dtype='f')
    matrix = utils.make_csr(matrix)
    smatrix.csr_to_SMatrix(matrix).column_stds(0, means, stds)
    matrix = matrix.A.astype('d')
    note('means real: {}  target: {}'.format(means, np.mean(matrix, axis=0)))
    note('stds real: {}  target: {}'.format(stds, np.std(matrix, axis=0)))
    assert utils.arrays_almost_equal(means, np.mean(matrix, axis=0), eps=1e-2)
    assert utils.arrays_almost_equal(stds, np.std(matrix, axis=0), eps=1e-2)


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_SMatrix_unstandardize_inverts_standardize(matrix):
    matrix = utils.make_csr(matrix)
    means = np.empty(shape=matrix.shape[1], dtype='f')
    stds = np.empty(shape=matrix.shape[1], dtype='f')
    m = smatrix.csr_to_SMatrix(matrix)
    m.column_stds(1, means, stds)
    m.standardize_columns(means, stds)
    m.unstandardize_columns(means, stds)
    result = m.to_csr()
    note('result: {}'.format(result.A))
    assert utils.arrays_almost_equal(matrix.A, result.A)


@given(utils.real_ndarrays_nan(utils.random_matrix_shapes()))
def test_SMatrix_row_vector(matrix):
    matrix = utils.make_csr(matrix)
    dense = matrix.A
    m = smatrix.csr_to_SMatrix(matrix)
    for row in range(matrix.shape[0]):
        assert utils.arrays_almost_equal(np.asarray(m.row_vector(row)), dense[row])


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_SMatrix_cluster_rows_mse(matrix):
    assume(matrix.shape[0] >= 2)
    matrix = utils.make_csr(matrix)
    dense = matrix.A
    c0 = dense[0]
    c1 = dense[1]

    d0 = np.sqrt(np.sum((dense - c0) * (dense - c0), axis=1))
    d1 = np.sqrt(np.sum((dense - c1) * (dense - c1), axis=1))
    target = (d1 <= d0).astype('f')

    result = np.empty(matrix.shape[0], dtype='f')
    smatrix.csr_to_SMatrix(matrix).cluster_rows_mse(c0, c1, result)

    note('result: {}'.format(result))
    note('target: {}'.format(target))
    assert utils.arrays_almost_equal(target, result)


@given(utils.real_ndarrays(utils.random_matrix_shapes()))
def test_DMatrix_cluster_rows_dot(matrix):
    assume(matrix.shape[0] >= 2)
    matrix = utils.make_csr(matrix)
    dense = matrix.A
    c0 = dense[0]
    c1 = dense[1]

    s0 = dense.dot(c0) / (np.linalg.norm(c0) + 1e-8)
    s1 = dense.dot(c1) / (np.linalg.norm(c1) + 1e-8)
    target = (s1 >= s0).astype('f')
    note(str(s0))
    note(str(s1))

    result = np.empty(matrix.shape[0], dtype='f')
    smatrix.csr_to_SMatrix(matrix).cluster_rows_dot(c0, c1, result, 1e-4, np.empty(matrix.shape[0], dtype='f'))

    note('result: {}'.format(result))
    note('target: {}'.format(target))
    assert utils.arrays_almost_equal(target, result)


