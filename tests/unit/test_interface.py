"""The satlas v1 compatibility layer is deprecated, but must keep working."""

import numpy as np
import pytest

from satlas2.interface import HFSModel, SumModel, chisquare_fit
from satlas2.models import HFS

ABC = [100.0, 50.0, 0.0, 0.0, 0.0, 0.0]


def make_model():
    return HFSModel(1.5, [0.5, 1.5], ABC, centroid=10.0, fwhm=[30.0, 10.0])


def test_hfs_model_is_deprecated_and_points_at_the_caller():
    with pytest.warns(DeprecationWarning, match="HFSModel is deprecated") as record:
        make_model()
    assert record[0].filename == __file__


def test_sum_model_is_deprecated():
    with pytest.warns(DeprecationWarning):
        models = [make_model(), make_model()]
    with pytest.warns(DeprecationWarning, match="SumModel is deprecated"):
        SumModel(models, {"values": [1.0], "bounds": []})


def test_hfs_model_still_matches_hfs():
    with pytest.warns(DeprecationWarning):
        model = make_model()
    reference = HFS(1.5, [0.5, 1.5], A=[100.0, 50.0], df=10.0, fwhmg=30.0, fwhml=10.0)
    x = np.linspace(-300, 300, 11)
    # HFSModel adds a constant background of 0.001 by default
    assert model.f(x) == pytest.approx(reference.f(x) + 0.001)


def test_chisquare_fit_is_deprecated():
    with pytest.warns(DeprecationWarning):
        truth, model = make_model(), make_model()
    x = np.linspace(-300, 300, 80)
    y = truth.f(x)
    with pytest.warns(DeprecationWarning, match="chisquare_fit is deprecated"):
        chisquare_fit(model, x, y, yerr=np.full_like(x, 0.01))
