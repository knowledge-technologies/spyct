import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spyct-tstepi",
    version="0.1",
    author="Tomaž Stepišnik",
    author_email="tomaz.stepi@gmail.com",
    description="An implementation of multivariate predictive clustering trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/TStepi/spyct",
    packages=setuptools.find_packages(exclude=['playgtround', 'example']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'joblib'],
)
