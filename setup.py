import os
from distutils.core import setup, Extension


with open("README.md", "r") as fh:
    long_description = fh.read()

math = Extension('spyct._math', sources=[os.path.join('cythonized', 'spyct', '_math.c')])
matrix = Extension('spyct._matrix', sources=[os.path.join('cythonized', 'spyct', '_matrix.c')])
data = Extension('spyct.data', sources=[os.path.join('cythonized', 'spyct', 'data.c')])
grad_splitter = Extension('spyct.grad_split', sources=[os.path.join('cythonized', 'spyct', 'grad_split.c')])
clustering = Extension('spyct.clustering', sources=[os.path.join('cythonized', 'spyct', 'clustering.c')])
node = Extension('spyct.node', sources=[os.path.join('cythonized', 'spyct', 'node.c')])

setup(
    name="spyct-tstepi",
    version="1.0",
    author="Tomaž Stepišnik",
    author_email="tomaz.stepi@gmail.com",
    description="An implementation of multivariate predictive clustering trees.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/TStepi/spyct",
    packages=['spyct'],
    ext_modules=[math, matrix, grad_splitter, data, clustering, node],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'joblib', 'scikit-learn'],
)
