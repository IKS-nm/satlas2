import logging
import numpy as np
import satlas2.utilities as utils
import pytest

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_weighted_average():
    """Test to see if the weighted average is correctly calculated"""
    data = np.array([0, 1])
    sigma = np.array([1, 1])
    result = utils.weightedAverage(data, sigma)
    assert 0.5 == pytest.approx(result[0])
    assert ((2**0.5) / 2) == pytest.approx(result[1])
