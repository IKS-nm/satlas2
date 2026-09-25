API reference
=============

Core module summary
-------------------

.. currentmodule:: satlas2.core

.. autosummary::

   ~Fitter
   ~Source
   ~Model

Models module summary
---------------------

.. currentmodule:: satlas2.models

.. autosummary::

   ~models.ExponentialDecay
   ~models.Polynomial
   ~models.SkewedVoigt
   ~models.PiecewiseConstant
   ~models.Voigt
   ~hfsModel.HFS

Lineshapes module summary
-------------------------

.. currentmodule:: satlas2.lineshapes

.. autosummary::

   ~gaussian
   ~lorentzian
   ~voigt
   ~skew
   ~voigtFWHM

Interface module summary
------------------------

.. currentmodule:: satlas2.interface

.. autosummary::

   ~HFSModel
   ~SumModel
   ~chisquare_fit

Plotting module summary
-----------------------

.. currentmodule:: satlas2.plotting

.. autosummary::

   ~generateCorrelationPlot
   ~generateWalkPlot

Utilities module summary
------------------------

.. currentmodule:: satlas2.utilities

.. autosummary::

   ~generateSpectrum
   ~poissonInterval
   ~weightedAverage

Subpages
--------

.. toctree::
   :maxdepth: 1

   summaries/core
   summaries/models
   summaries/lineshapes
   summaries/interface
   summaries/plotting
   summaries/utilities