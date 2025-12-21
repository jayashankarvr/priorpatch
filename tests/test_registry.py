from priorpatch.detectors.registry import DETECTOR_REGISTRY


def test_registry():
    assert isinstance(DETECTOR_REGISTRY, dict)
    assert 'color_stats' in DETECTOR_REGISTRY
