from libc.math cimport isnan
from spyct._matrix cimport *
from spyct._math cimport *


cpdef DTYPE[::1] kmeans(Matrix data, DTYPE[::1] centroid0, DTYPE[::1] centroid1,
                        index max_iter, DTYPE tol, DTYPE eps, int cluster_distance, bint missing_data):
    cdef:
        DTYPE entropy, new_entropy
        DTYPE[::1] left_or_right, temp
        index i

    if cluster_distance == 2:
        temp = create_real_vector(data.n_rows)
        if missing_data:
            data.impute_missing(0)

    left_or_right = create_real_vector(data.n_rows)
    if cluster_distance == 1:
        if missing_data:
            entropy = data.cluster_rows_mse_nan(centroid0, centroid1, left_or_right)
        else:
            entropy = data.cluster_rows_mse(centroid0, centroid1, left_or_right)
    else:
        entropy = data.cluster_rows_dot(centroid0, centroid1, left_or_right, eps, temp)

    # optimization iterations
    for i in range(max_iter):

        update_centroids(data, left_or_right, centroid0, centroid1, missing_data)

        if cluster_distance == 1:
            if missing_data:
                new_entropy = data.cluster_rows_mse_nan(centroid0, centroid1, left_or_right)
            else:
                new_entropy = data.cluster_rows_mse(centroid0, centroid1, left_or_right)
            if new_entropy == 0 or new_entropy / entropy > 1 - tol:
                break
            else:
                entropy = new_entropy
        else:
            new_entropy = data.cluster_rows_dot(centroid0, centroid1, left_or_right, eps, temp)
            if new_entropy / entropy < 1 + tol:
                break
            else:
                entropy = new_entropy

    return left_or_right

cpdef void update_centroids(Matrix data, DTYPE[::1] left_or_right, DTYPE[::1] centroid0, DTYPE[::1] centroid1,
                            bint missing_data):
    cdef index row, col, num0, num1
    cdef DMatrix ddata
    cdef DTYPE lor, v, sum0, sum1
    cdef DTYPE[::1] temp0, temp1

    if not missing_data:
        temp0 = create_real_vector(data.n_rows)
        temp1 = create_real_vector(data.n_rows)
        sum0 = 0
        sum1 = 0
        for row in range(data.n_rows):
            lor = left_or_right[row]
            if lor == 0:
                sum0 += 1
                temp0[row] = 1
                temp1[row] = 0
            elif lor == 1:
                sum1 += 1
                temp1[row] = 1
                temp0[row] = 0
            else:
                temp0[row] = 0
                temp1[row] = 0

        data.vector_dot_self(temp0, centroid0)
        vector_scalar_prod(centroid0, 1/sum0)

        data.vector_dot_self(temp1, centroid1)
        vector_scalar_prod(centroid1, 1/sum1)
    else:
        ddata = <DMatrix> data
        # Matrix is not sparse, because sparse matrices do not support missing data (yet).
        for col in range(ddata.n_cols):
            centroid0[col] = 0
            centroid1[col] = 0
            num0 = 0
            num1 = 0
            for row in range(ddata.n_rows):
                lor = left_or_right[row]
                v = ddata.data[row, col]
                if not isnan(v):
                    if lor == 1:
                        centroid1[col] += v
                        num1 += 1
                    elif lor == 0:
                        centroid0[col] += v
                        num0 += 1
            if num0 > 0:
                centroid0[col] /= num0
            if num1 > 0:
                centroid1[col] /= num1





