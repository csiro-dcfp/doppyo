from setuptools import setup
#from distutils.core import setup

setup(
    name='pyLatte_cafe',
    version='0.0.5',
    author='D. Squire',
    author_email='dougie.squire@csiro.au',
    packages=['pylatte', 'pylatte.test'],
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
	"windspharm >= 1.6.0",    
    ],
)
