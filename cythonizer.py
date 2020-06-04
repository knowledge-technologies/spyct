import os
from distutils.core import Extension
from Cython.Build import cythonize


cython_directives = {
    'language_level': 3,
    'cdivision': True,
    'wraparound': False,
    'boundscheck': False,
    'initializedcheck': False,
    'profile': False,
}

math = Extension('spyct._math', sources=[os.path.join('spyct', '_math.pyx')])
matrix = Extension('spyct._matrix', sources=[os.path.join('spyct', '_matrix.pyx')])
data = Extension('spyct.data', sources=[os.path.join('spyct', 'data.pyx')])
grad_splitter = Extension('spyct.grad_split', sources=[os.path.join('spyct', 'grad_split.pyx')])
clustering = Extension('spyct.clustering', sources=[os.path.join('spyct', 'clustering.pyx')])

cythonize([math, matrix, data, grad_splitter, clustering],
          compiler_directives=cython_directives,
          force=True, annotate=True, build_dir='cythonized')

