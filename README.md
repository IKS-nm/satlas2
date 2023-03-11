# SATLAS2 -- Statistical Analysis Toolbox for Laser Spectroscopy Version 2

==========================================================================

![alt text](https://img.shields.io/badge/License-MIT-blue.svg 'License')
![alt text](https://img.shields.io/badge/Python-3.x-green.svg 'Python version')
![alt text](https://img.shields.io/badge/Tested_on-Windows-green.svg 'Supported platform')

## Purpose

----------
Contributors:

* Wouter Gins (wouter.gins@kuleuven.be)
* Bram van den Borne (bram.vandenborne@kuleuven.be)

An updated version of the [satlas](http://github.com/woutergins/satlas/) package. A different architecture of the code is used, resulting in a speedup of roughly 2 orders of magnitude in fitting, with increased support for simultaneous fitting and summing models.

## Dependencies

----------
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

----------

A package is available on PyPi, so 'pip install satlas2' should provide a working environment.
