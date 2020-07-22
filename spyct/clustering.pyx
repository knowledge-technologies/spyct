from spyct._matrix cimport *
from spyct._math cimport *
import numpy as np


cpdef DTYPE[::1] kmeans(Matrix data, DTYPE[::1] centroid0, DTYPE[::1] centroid1, DTYPE[::1] tiebraker,
                        index max_iter, DTYPE tol, DTYPE eps, int cluster_distance):
    cdef:
        DTYPE entropy, new_entropy, sum
        DTYPE[::1] left_or_right, temp
        index i

    if cluster_distance == 2:
        temp = create_real_vector(data.n_rows)

    left_or_right = create_real_vector(data.n_rows)
    if cluster_distance == 1:
        entropy = data.cluster_rows_mse(centroid0, centroid1, left_or_right, tiebraker)
    else:
        entropy = data.cluster_rows_dot(centroid0, centroid1, left_or_right, eps, temp)

    # optimization iterations
    for i in range(max_iter):
        data.vector_dot_self(left_or_right, centroid1)
        sum = vector_sum(left_or_right)
        vector_scalar_prod(centroid1, 1/sum)

        vector_scalar_prod(left_or_right, -1)
        vector_scalar_sum(left_or_right, 1)
        data.vector_dot_self(left_or_right, centroid0)
        vector_scalar_prod(centroid0, 1/(left_or_right.shape[0] - sum))

        if cluster_distance == 1:
            new_entropy = data.cluster_rows_mse(centroid0, centroid1, left_or_right, tiebraker)
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
