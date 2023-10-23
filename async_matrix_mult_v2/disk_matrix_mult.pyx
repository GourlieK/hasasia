import numpy as np
import ast
cimport cython
cimport numpy as cnp

cnp.import_array()
DTYPE = np.double  

ctypedef cnp.double_t DTYPE_t
cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

@cython.boundscheck(False)
@cython.wraparound(False)


def fast_for_loop(str filename_1, str filename_2, cnp.ndarray[DTYPE_t, ndim=2] result):
    cdef double[:] cline_1, cline_2
    cdef int n,m
    with open(filename_1, 'r') as file_1, open(filename_2, 'r') as file_2:
        n = -1
        for line_1 in file_1:
            m = -1
            file_2.seek(0)  # resets the text pointer for the second file
            line_1 = np.asarray(ast.literal_eval((line_1.strip('\n'))))
            cline_1 = line_1
            n = n + 1
            print(n)
            for line_2 in file_2:
                line_2 = np.array(ast.literal_eval((line_2.strip('\n'))))
                cline_2 = line_2
                m = m + 1 
                result[n, m] = ddot(cline_1.shape[0], &cline_1[0], 1, &cline_2[0], 1) 
        return result








     