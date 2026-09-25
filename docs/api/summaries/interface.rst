API Interface
=============

.. deprecated:: 0.4.0
    The interface module only exists to ease the migration from satlas and will
    be removed in a future version. Use :class:`~satlas2.models.hfsModel.HFS`
    with :class:`~satlas2.core.Source` and :class:`~satlas2.core.Fitter` instead.

Interface summaries
-------------------

.. currentmodule:: satlas2.interface

.. autoclass:: HFSModel
    :noindex:

    .. rubric:: Methods

    .. autosummary::
   
        ~HFSModel.set_expr
        ~HFSModel.set_variation
        ~HFSModel.f
        ~HFSModel.chisquare_fit
        ~HFSModel.display_chisquare_fit
        ~HFSModel.get_result
        ~HFSModel.get_result_dict
        ~HFSModel.get_result_frame

.. autoclass:: SumModel
    :noindex:

    .. rubric:: Methods

    .. autosummary::
   
        ~SumModel.f
        ~SumModel.chisquare_fit
        ~SumModel.display_chisquare_fit
        ~SumModel.get_result
        ~SumModel.get_result_dict
        ~SumModel.get_result_frame

.. autofunction:: chisquare_fit
    :noindex:

Extensive interface
-------------------

.. inheritance-diagram:: satlas2.interface

.. automodule:: satlas2.interface
   :members:
   :undoc-members:
   :show-inheritance:
