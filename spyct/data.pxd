from spyct._matrix cimport *
from spyct._math cimport *

cpdef Matrix matrix_from_np_or_sp(np_or_sp)
cpdef bint missing_values_in_matrix(np_or_sp)
cpdef DTYPE relative_impurity(Data data1, Data data2)


cdef class Data:
    cdef:
        readonly index n, d, c, t
        readonly Matrix descriptive_data, clustering_data, target_data
        readonly bint missing_descriptive, missing_clustering, missing_target
        readonly DTYPE[::1] impurities

    cpdef Data take_rows(self, index[::1] rows)
    cpdef void calc_impurity(self, DTYPE eps)
    cpdef index min_labelled(self)
