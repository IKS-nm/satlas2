# SATLAS2 -- Statistical Analysis Toolbox for Laser Spectroscopy Version 2

![alt text](https://img.shields.io/pypi/v/satlas2?label=PyPI%20version 'PyPI version')
![alt text](https://img.shields.io/pypi/pyversions/satlas2?label=Python%20version&logo=python&logoColor=white 'Python version')
![alt text](https://img.shields.io/pypi/l/satlas2?color=blue&label=License 'License')

![alt text](https://img.shields.io/badge/Tested_on-Windows/Linux-green.svg 'Supported platform')
![alt text](https://img.shields.io/badge/Not_tested_on-Mac-red.svg 'Unsupported platform')

![alt text](https://img.shields.io/pypi/dm/satlas2?label=Downloads 'PyPI - Downloads')

## Purpose

Contributors:

* Ruben de Groote (ruben.degroote@kuleuven.be)
* Wouter Gins (wouter.gins@kuleuven.be)
* Bram van den Borne (bram.vandenborne@kuleuven.be)

An updated version of the [satlas](http://github.com/woutergins/satlas/) package. A different architecture of the code is used, resulting in a speedup of roughly 2 orders of magnitude in fitting, with increased support for simultaneous fitting and summing models. Documentation can be found [here](https://iks-nm.github.io/satlas2/).

## Dependencies

This package makes use of the following packages:

* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)
* [SciPy](http://www.scipy.org/)
* [h5py](http://docs.h5py.org/en/latest/index.html)
* [emcee](http://dan.iel.fm/emcee/current/)
* [sympy](http://www.sympy.org/)
* [LMFIT](http://lmfit.github.io/lmfit-py/index.html)
* [numdifftools](http://numdifftools.readthedocs.io/en/latest/)
* [uncertainties](https://pythonhosted.org/uncertainties/)
* [tqdm](https://github.com/tqdm/tqdm)
* [pandas](https://pandas.pydata.org/)

Only Python 3.x is supported! Parts of the code have been based on other resources; this is signaled in the documentation when this is the case. Inspiration has been drawn from the `triangle.py` script, written by Dan Foreman-Mackey et al., for the correlation plot code.

## Installation

A package is available on PyPi, so 'pip install satlas2' should provide a working environment.
