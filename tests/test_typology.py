from loopstop.analysis.typology import classify, type_proportions


def test_monotone_converging():
    r = [0.1, 0.25, 0.4, 0.55, 0.65, 0.7, 0.72, 0.73, 0.735, 0.735]
    assert classify(r).label == "monotone"


def test_plateau_early():
    # 第2轮后基本不动, k < T/2
    r = [0.2, 0.7, 0.71, 0.71, 0.715, 0.71, 0.712, 0.71, 0.711, 0.71]
    assert classify(r).label == "plateau"


def test_oscillating():
    r = [0.3, 0.7, 0.35, 0.75, 0.3, 0.7, 0.72, 0.71]
    res = classify(r)
    assert res.label == "oscillating"
    assert res.details["reversals"] >= 2


def test_degrading():
    r = [0.2, 0.6, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45]
    res = classify(r)
    assert res.label == "degrading"
    assert res.peak_t <= len(r) * 2 / 3


def test_still_rising_is_other():
    # A curve still rising at the final round is classified as other.
    r = [0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85]
    assert classify(r).label == "other"


def test_proportions_sum_to_one():
    series = [
        [0.1, 0.3, 0.5, 0.6, 0.65, 0.66, 0.665, 0.665],
        [0.2, 0.6, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45],
    ]
    props = type_proportions(series)
    assert abs(sum(props.values()) - 1.0) < 1e-9
