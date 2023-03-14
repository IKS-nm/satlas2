
Architecture of SATLAS2
=======================

The SATLAS2 architecture has been streamlined and differs enormously from SATLAS. For a way to mostly reuse old SATLAS code, see the :doc:`interface tutorial<../tutorials/interface/index>`. Help

In SATLAS2, the main object to work with is the :class:`Fitter<satlas2.core.Fitter>` object. This object can be assigned one or more :class:`Source<satlas2.core.Source>` objects through the :meth:`addSource<satlas2.core.Fitter.addSource>` method. This :class:`Source<satlas2.core.Source>` object should be seen as a source for the cost-function calculation, rather than strictly associated with a data source. Each :class:`Source<satlas2.core.Source>` can itself contain one or more :class:`Model<satlas2.core.Model>` objects, which calculate a response based on parameters and an input. Models that are assigned to the same Source will be added together to generate the total response.

When a :class:`Fitter<satlas2.core.Fitter>` object has multiple :class:`Source<satlas2.core.Source>` objects, the total cost function will be calculated by concatenating, essentially performing a simultaneous fit of models to different datasets. The :class:`Fitter<satlas2.core.Fitter>` object uses the LMFIT library for the fitting, for which a Parameters object containing the parameters of all the models is created. This structure allows a very easy way of using the powerful LMFIT expressions to constrain parameters to be shared. For ease of use, the :class:`Fitter<satlas2.core.Fitter>` object contains the :meth:`shareModelParams<satlas2.core.Fitter.shareModelParams>` and :meth:`shareParams<satlas2.core.Fitter.shareParams>` to set parameters to be shared across models with the same name in different Sources, or simply across all models respectively. The method :meth:`setExpr<satlas2.core.Fitter.setExpr>` can be used to set the expression of a parameter directly.

The architecture can be summarized in the following picture featuring an example:

.. raw:: html
    :file: architecture.svg

In this example, a Source with the name *scan001* contains two models: the HFS model with the name *Pb208*, and a background model named *bkg1*. Also included in the source is some data with x, y and uncertainty of y. A second source with the name *scan002* contains another HFS model with the same name, but the background has a different name. Since the HFS models have the same name, :meth:`shareModelParams<satlas2.core.Fitter.shareModelParams>` can be used on the Fitter object to link desired parameters together, such as the hyperfine parameters. When performing the fit, the Fitter will fit the models in Source 1 to the data in Source 1, and the models in Source 2 to the data in Source 2 simultaneously.

This architecture is a big deviation from the architecture in SATLAS, where the paradigm was that special SumModels and LinkedModels would be created for such occasions. Instead, in SATLAS2, by implementing that models assigned to the same source are summed together and assigning different sources causes a simultaneous fit, several bugs present in SATLAS are avoided simply by reducing the coding complexity. As a bonus, with this standardized implementation, speedups of a factor 20 to 200 can be achieved, as is shown in the :doc:`benchmark<../tutorials/benchmark/index>`.