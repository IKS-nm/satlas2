SATLAS2 -- Statistical Analysis Toolbox for Laser Spectroscopy, version 2
=========================================================================

.. image:: https://img.shields.io/pypi/v/satlas2?label=PyPI%20version
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/satlas2?label=Python%20version&logo=python&logoColor=white
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/l/satlas2?color=blue&label=License
   :alt: PyPI - License

\

.. image:: https://img.shields.io/badge/Tested_on-Windows/Linux-green.svg
    :alt: Supported Platform

.. image:: https://img.shields.io/badge/Not_tested_on-Mac-red.svg
    :alt: Unsupported platform

\

.. image:: https://img.shields.io/pypi/dm/satlas2?label=Downloads
   :alt: PyPI - Downloads

\

.. image:: https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1016%2Fj.cpc.2023.109053-blue
   :target: https://doi.org/10.1016/j.cpc.2023.109053


Purpose
=======

.. sidebar:: Contributors

    - Bram van den Borne
    - Wouter Gins

This Python package has been created with the goal of creating an easier interface for the analysis of data gathered from laser spectroscopy experiments. Support for fitting the spectra, using both :math:`\chi^2`-fitting and Maximum Likelihood Estimation routines, are present.

Dependencies
============
This package has the following dependencies:

    * `NumPy <http://www.numpy.org/>`_
    * `Matplotlib <http://matplotlib.org/>`_
    * `SciPy <http://www.scipy.org/>`_
    * `h5py <http://docs.h5py.org/en/latest/index.html>`_
    * `emcee <http://dan.iel.fm/emcee/current/>`_
    * `sympy <http://www.sympy.org/>`_
    * `LMFIT <http://lmfit.github.io/lmfit-py/index.html>`_
    * `numdifftools <http://numdifftools.readthedocs.io/en/latest/>`_
    * `uncertainties <https://pythonhosted.org/uncertainties/>`_
    * `tqdm <https://github.com/tqdm/tqdm>`_
    * `pandas <https://pandas.pydata.org/>`_

Contents
========
.. toctree::
   :maxdepth: 1

   architecture/index
   tutorials/index
   api/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
