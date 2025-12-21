from priorpatch.core import Ensemble
from priorpatch.utils import load_image


def test_ensemble_runs():
    img = load_image('examples/sample_input.png')
    ens = Ensemble.from_config('config/detectors.json')
    heat = ens.score_image(img, patch_size=64, stride=64)
    assert heat.min() >= 0.0 and heat.max() <= 1.0
