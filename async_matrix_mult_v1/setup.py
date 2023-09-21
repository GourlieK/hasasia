from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy



setup(
    ext_modules=cythonize(
        Extension(
            'fast_for_loop',
            sources=['disk_matrix_mult.pyx'],
            language='c',
            include_dirs=[numpy.get_include()],
            library_dirs=[],  
            libraries=[':libcblas.so.3'],  
            extra_link_args=[]
        )
    )
)
