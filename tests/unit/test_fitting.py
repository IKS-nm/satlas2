import logging
import numpy as np
import satlas2
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


@pytest.fixture
def simple_fitter():
    rng = np.random.default_rng(42)
    x = np.linspace(-100, 100, 50)
    y = 3.0 + 0.01 * x + rng.normal(0, 0.1, size=x.shape)
    yerr = np.full_like(x, 0.1)
    model = satlas2.Polynomial(p=[3.0, 0.0], name='BG')
    src = satlas2.Source(x, y, yerr, name='test')
    src.addModel(model)
    fitter = satlas2.Fitter()
    fitter.addSource(src)
    return fitter


def test_basic_chisq_fit(simple_fitter):
    simple_fitter.fit()
    assert simple_fitter.result.success
    assert simple_fitter.result.redchi > 0
    df = simple_fitter.createResultDataframe()
    assert len(df) > 0


def test_create_metadata_dataframe(simple_fitter):
    simple_fitter.fit()
    meta = simple_fitter.createMetadataDataframe()
    expected_cols = {'Fitting method', 'Message', 'Function evaluations',
                     'Data points', 'Variables', 'Chisquare', 'Redchi', 'Aic', 'Bic'}
    assert expected_cols.issubset(set(meta.columns))
    assert len(meta) == 1


def test_read_walk_metadata(simple_fitter, tmp_path):
    chains_file = str(tmp_path / 'chains.h5')
    simple_fitter.fit()
    simple_fitter.fit(
        method='emcee',
        llh_method='gaussian',
        nwalkers=10,
        steps=50,
        filename=chains_file,
    )
    simple_fitter.readWalk(chains_file, burnin=10)
    meta = simple_fitter.createMetadataDataframe()
    assert meta['Fitting method'].iloc[0] == 'emcee'
    assert len(meta) == 1
