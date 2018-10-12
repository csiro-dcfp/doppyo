from setuptools import setup

setup(
    name='doppyo',
    version='0.0.1',
    author='D. Squire',
    author_email='dougie.squire@csiro.au',
    packages=['doppyo', 'doppyo.test'],
    # url=''
    license='LICENSE.txt',
    description='Diagnostics and verification package.',
    long_description=open('README.md').read(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "dask >= 0.17.1",
        "matplotlib >= 2.2.2",
        "bokeh >= 0.12.13",
        "netcdf4 >= 1.3.1",
        "numpy >= 1.13.3",
        "pandas >= 0.22.0",
        "pytest >= 3.5.0",
        "xarray >= 0.10.7",
        "bottleneck >= 1.2.1",
        "cartopy >= 0.16.0",
        "scipy >= 1.0.0",
	"pyspharm >= 1.0.9",
	"windspharm >= 1.6.0",    
    ],
)
