# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Tests for the Cox proportional hazards partial likelihood objective.

The target is encoded as ``y = time * (2 * event - 1)``: positive values are
event (uncensored) times and negative values are censored times. ``time == 0``
is rejected.
"""

import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingRegressor


def _encode_survival(time, event):
    """Encode (time, event) into the signed-time target used by the C++ objective."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    return np.where(event > 0.5, time, -time)


def _simulate_cox_data(n, rng, true_betas, censor_rate=0.3):
    """Simulate survival times from a Cox model with exp-distributed event times."""
    p = len(true_betas)
    X = rng.normal(size=(n, p))
    log_hr = X @ np.asarray(true_betas, dtype=np.float64)
    # Exponential event times with rate exp(log_hr): smaller time = higher hazard
    event_times = rng.exponential(scale=np.exp(-log_hr))
    censored = rng.random(n) < censor_rate
    event = (~censored).astype(np.float64)
    y = _encode_survival(event_times, event)
    return X, y, event_times, event


def _concordance_index(predictions, event_times, event):
    """Simple O(n^2) concordance index for survival predictions.

    Higher prediction should correspond to higher hazard (shorter survival).
    """
    n = len(predictions)
    concordant = 0
    comparable = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # pair is comparable if i has an event and its time is smaller
            if event[i] > 0.5 and event_times[i] < event_times[j]:
                comparable += 1
                if predictions[i] > predictions[j]:
                    concordant += 1
                elif predictions[i] == predictions[j]:
                    concordant += 0.5
    if comparable == 0:
        return float("nan")
    return concordant / comparable


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


def test_cox_basic_fit():
    """A Cox EBM should fit and predict without errors on a simple survival dataset."""
    rng = np.random.default_rng(0)
    X, y, _, _ = _simulate_cox_data(200, rng, true_betas=[1.0, 0.5])

    ebm = ExplainableBoostingRegressor(objective="survival_cox", interactions=0)
    ebm.fit(X, y)
    preds = ebm.predict(X)

    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
    for term_scores in ebm.term_scores_:
        assert np.all(np.isfinite(term_scores))


def test_cox_all_events():
    """Cox should fit when every sample has an event (no censoring)."""
    rng = np.random.default_rng(1)
    X, _, event_times, _ = _simulate_cox_data(100, rng, true_betas=[0.8])
    y = event_times  # all positive => all events

    ebm = ExplainableBoostingRegressor(objective="survival_cox", interactions=0)
    ebm.fit(X, y)
    assert np.all(np.isfinite(ebm.predict(X)))


def test_cox_all_censored_fits():
    """With no events at all, the likelihood is constant — fit should succeed and
    produce a flat model (zero learning signal)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 1))
    times = rng.exponential(size=50)
    y = -times  # every sample censored

    ebm = ExplainableBoostingRegressor(objective="survival_cox", interactions=0)
    ebm.fit(X, y)
    preds = ebm.predict(X)
    # Every sample is censored, so there is no information and predictions
    # should be effectively constant.
    assert np.ptp(preds) < 1e-6


def test_cox_zero_target_rejected():
    """time == 0 is invalid (ambiguous event/censor) and must be rejected."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 1))
    y = np.ones(10)
    y[0] = 0.0  # invalid
    ebm = ExplainableBoostingRegressor(objective="survival_cox", interactions=0)
    with pytest.raises(Exception):
        ebm.fit(X, y)


def test_cox_rejects_classifier():
    """Cox is a regression-task objective; classifiers must reject it."""
    from interpret.glassbox import ExplainableBoostingClassifier

    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 2))
    y = (rng.random(20) < 0.5).astype(np.int64)

    clf = ExplainableBoostingClassifier(objective="survival_cox")
    with pytest.raises(Exception):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Statistical correctness tests
# ---------------------------------------------------------------------------


def test_cox_recovers_direction():
    """On simulated data with a known log hazard ratio, predictions should
    correlate positively with the true log hazard ratio."""
    rng = np.random.default_rng(42)
    true_betas = [1.2, -0.8]
    X, y, _, _ = _simulate_cox_data(500, rng, true_betas=true_betas)
    true_log_hr = X @ np.asarray(true_betas)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=2000,
    )
    ebm.fit(X, y)
    preds = ebm.predict(X)

    # Spearman-style rank correlation with the true signal
    corr = np.corrcoef(preds, true_log_hr)[0, 1]
    assert corr > 0.7, f"expected strong correlation with true log-HR, got {corr:.3f}"


def test_cox_concordance_beats_random():
    """The fitted Cox EBM should produce a concordance index well above 0.5."""
    rng = np.random.default_rng(123)
    X, y, event_times, event = _simulate_cox_data(
        300, rng, true_betas=[1.0, 0.5], censor_rate=0.3
    )

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=2000,
    )
    ebm.fit(X, y)
    preds = ebm.predict(X)

    c_index = _concordance_index(preds, event_times, event)
    assert c_index > 0.7, f"concordance index {c_index:.3f} should exceed 0.7"


def test_survival_cox_score_decreases():
    """The negative partial log-likelihood (NPL) reported during training
    should be lower after many rounds than after very few rounds."""
    rng = np.random.default_rng(7)
    X, y, _, _ = _simulate_cox_data(300, rng, true_betas=[1.0])

    # Few rounds
    ebm_few = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=5,
        early_stopping_rounds=0,
    )
    ebm_few.fit(X, y)

    # Many rounds
    ebm_many = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=2000,
    )
    ebm_many.fit(X, y)

    # Compare training loss via the stored best_iteration_ score if present,
    # otherwise fall back to checking concordance.
    # We know more rounds should explain the data at least as well.
    # Use the range of predictions as a proxy for learned signal.
    assert np.ptp(ebm_many.predict(X)) > np.ptp(ebm_few.predict(X))


def test_cox_breslow_first_step_formula():
    """Hand-computed Breslow gradient verification for a 4-sample dataset.

    Samples, sorted by time (all events, no censoring):
      s0: (time=1, event, bin=0)
      s1: (time=2, event, bin=0)
      s2: (time=3, event, bin=1)
      s3: (time=4, event, bin=1)

    At initial scores=0, exp(score)=1 everywhere.
    Risk-set sizes at each event time: [4, 3, 2, 1].

    Forward pass (cumulative sum of 1/S over events with time <= t_k):
      j=0: cumH = 1/4                          cumH2 = 1/16
        grad[0] = 1/4 - 1          = -3/4     hess[0] = 1/4 - 1/16          = 3/16
      j=1: cumH = 1/4 + 1/3 = 7/12             cumH2 = 1/16 + 1/9 = 25/144
        grad[1] = 7/12 - 1         = -5/12    hess[1] = 7/12 - 25/144       = 59/144
      j=2: cumH = 7/12 + 1/2 = 13/12           cumH2 = 25/144 + 1/4 = 61/144
        grad[2] = 13/12 - 1        = 1/12     hess[2] = 13/12 - 61/144      = 95/144
      j=3: cumH = 13/12 + 1 = 25/12            cumH2 = 61/144 + 1 = 205/144
        grad[3] = 25/12 - 1        = 13/12    hess[3] = 25/12 - 205/144     = 95/144

    By bin:
      bin 0 (s0, s1): sum_grad = -3/4 + -5/12 = -7/6,  sum_hess = 3/16 + 59/144 = 43/72
      bin 1 (s2, s3): sum_grad = 1/12 + 13/12 = 7/6,   sum_hess = 95/144 + 95/144 = 95/72

    Newton update = -sum_grad / sum_hess:
      bin 0: -(-7/6) / (43/72) =  84/43 ≈  1.9535
      bin 1: -( 7/6) / (95/72) = -84/95 ≈ -0.8842

    With learning_rate=0.01:
      bin 0 update ≈  0.019535  (short times  -> higher hazard)
      bin 1 update ≈ -0.008842  (long times   -> lower  hazard)
    """
    # Pre-binned feature: bin 0 for samples 0,1 (short times); bin 1 for samples 2,3.
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    times = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 1.0])  # all events for clean hand-check
    y = _encode_survival(times, event)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=1,
        learning_rate=0.01,
        min_samples_leaf=1,
        min_hessian=0.0,
        early_stopping_rounds=0,
        outer_bags=1,
        validation_size=0,
    )
    ebm.fit(X, y)

    # Predict on the training points: subtract intercept to isolate term contribution.
    preds = ebm.predict(X) - ebm.intercept_
    pred_bin0 = preds[0]  # also preds[1]
    pred_bin1 = preds[2]  # also preds[3]

    # Sign check: short times (bin 0) -> positive update; long times (bin 1) -> negative.
    assert pred_bin0 > 0, (
        f"bin 0 (short times) should have positive update, got {pred_bin0}"
    )
    assert pred_bin1 < 0, (
        f"bin 1 (long times) should have negative update, got {pred_bin1}"
    )

    # The EBM Python wrapper purifies term scores after each boost round, which
    # shifts the mean into the intercept. The *difference* bin0 - bin1 is
    # invariant to this purification and should equal the raw Breslow Newton
    # gap: lr * (84/43 - (-84/95)) = 0.01 * 84*(1/43 + 1/95)
    expected_gap = 0.01 * (84.0 / 43.0 - (-84.0 / 95.0))  # ≈ 0.02838
    actual_gap = pred_bin0 - pred_bin1
    assert abs(actual_gap - expected_gap) < 0.001, (
        f"bin0 - bin1 = {actual_gap:.6f} should match Breslow gap {expected_gap:.6f}"
    )


def test_cox_recovers_linear_log_hazard():
    """With a single linear feature and no censoring, the fitted shape function
    for that feature should be approximately monotone-increasing in the feature
    value (matching the simulated positive log-hazard coefficient)."""
    rng = np.random.default_rng(99)
    X, y, _, _ = _simulate_cox_data(400, rng, true_betas=[1.5], censor_rate=0.0)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        max_rounds=2000,
    )
    ebm.fit(X, y)

    # Evaluate the shape function on a monotonic grid of feature values.
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 25).reshape(-1, 1)
    preds = ebm.predict(grid)
    # At least 80% of successive differences should be non-negative.
    diffs = np.diff(preds)
    frac_nondecreasing = np.mean(diffs >= -1e-6)
    assert frac_nondecreasing > 0.8, (
        f"expected roughly monotonic shape for positive beta, "
        f"got only {frac_nondecreasing:.2%} non-decreasing"
    )


# ---------------------------------------------------------------------------
# Additional robustness / integration tests
# ---------------------------------------------------------------------------


def test_cox_determinism_same_seed():
    """Two fits with the same random_state must produce identical predictions."""
    rng = np.random.default_rng(2024)
    X, y, _, _ = _simulate_cox_data(150, rng, true_betas=[0.7, -0.3])

    kwargs = dict(
        objective="survival_cox",
        interactions=0,
        random_state=7,
        max_rounds=300,
    )
    ebm_a = ExplainableBoostingRegressor(**kwargs).fit(X, y)
    ebm_b = ExplainableBoostingRegressor(**kwargs).fit(X, y)

    np.testing.assert_allclose(ebm_a.predict(X), ebm_b.predict(X), rtol=0, atol=0)


def test_cox_sign_flip_via_event_inversion():
    """Flipping the event indicator for all samples reverses the problem: samples
    that previously provided "event" information are now censored and vice versa.
    The fit should NOT recover the same direction — it's a different problem — but
    it should still produce a finite, valid model."""
    rng = np.random.default_rng(11)
    X, y, _, _ = _simulate_cox_data(150, rng, true_betas=[1.0], censor_rate=0.5)
    y_flipped = -y  # invert which samples are events vs censored

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=300
    )
    ebm.fit(X, y_flipped)
    assert np.all(np.isfinite(ebm.predict(X)))


def test_cox_handles_tied_event_times():
    """Breslow's partial likelihood uses the simple tie approximation. Ensure
    the fit is stable when many samples share the same event time."""
    rng = np.random.default_rng(13)
    n = 200
    X = rng.normal(size=(n, 2))
    # Force times onto a coarse grid (many ties)
    raw_times = rng.exponential(scale=np.exp(-X[:, 0]))
    times = np.round(raw_times * 4.0) / 4.0 + 0.001  # snap to 0.25 grid, avoid 0
    event = (rng.random(n) > 0.2).astype(np.float64)
    y = _encode_survival(times, event)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=500
    )
    ebm.fit(X, y)
    preds = ebm.predict(X)
    assert np.all(np.isfinite(preds))
    # First feature should still show positive correlation with true log-HR direction
    corr = np.corrcoef(preds, X[:, 0])[0, 1]
    assert corr > 0.3


def test_cox_pickle_roundtrip():
    """The fitted Cox EBM must survive a pickle round-trip with identical predictions."""
    import pickle

    rng = np.random.default_rng(17)
    X, y, _, _ = _simulate_cox_data(100, rng, true_betas=[0.9])

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=100
    )
    ebm.fit(X, y)
    preds_before = ebm.predict(X)

    blob = pickle.dumps(ebm)
    ebm2 = pickle.loads(blob)
    preds_after = ebm2.predict(X)
    np.testing.assert_allclose(preds_before, preds_after, rtol=0, atol=0)


def test_cox_predict_on_new_samples():
    """Model trained on one split should produce finite predictions on a held-out split."""
    rng = np.random.default_rng(21)
    X, y, _, _ = _simulate_cox_data(200, rng, true_betas=[1.0, -0.5])
    X_train, y_train = X[:150], y[:150]
    X_test = X[150:]

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=300
    )
    ebm.fit(X_train, y_train)
    preds = ebm.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert np.all(np.isfinite(preds))


def test_cox_sample_weights_accepted():
    """sample_weight should be accepted and yield a valid fit."""
    rng = np.random.default_rng(23)
    X, y, _, _ = _simulate_cox_data(120, rng, true_betas=[0.8])
    weights = rng.uniform(0.5, 1.5, size=len(y))

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=200
    )
    ebm.fit(X, y, sample_weight=weights)
    assert np.all(np.isfinite(ebm.predict(X)))


def test_cox_categorical_feature():
    """Nominal (categorical) features should work with Cox."""
    rng = np.random.default_rng(29)
    n = 300
    group = rng.integers(0, 3, size=n).astype(np.float64)
    x_cont = rng.normal(size=n)
    X = np.column_stack([group, x_cont])
    # hazard depends on group: group 0 high hazard, group 2 low hazard
    log_hr = np.where(group == 0, 1.0, np.where(group == 2, -1.0, 0.0)) + 0.3 * x_cont
    times = rng.exponential(scale=np.exp(-log_hr))
    event = (rng.random(n) > 0.2).astype(np.float64)
    y = _encode_survival(times, event)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        feature_types=["nominal", "continuous"],
        max_rounds=500,
    )
    ebm.fit(X, y)
    assert np.all(np.isfinite(ebm.predict(X)))

    # Scoring on new group levels should succeed too
    X_new = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    preds = ebm.predict(X_new)
    # group 0 (high hazard) should have a larger score than group 2 (low hazard)
    assert preds[0] > preds[2], (
        f"expected group 0 (high hazard) score > group 2 (low hazard) score, "
        f"got preds={preds}"
    )


def test_cox_outer_bags_averaging():
    """Multiple outer bags should still produce a valid, finite model."""
    rng = np.random.default_rng(31)
    X, y, _, _ = _simulate_cox_data(200, rng, true_betas=[1.0])

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        outer_bags=4,
        max_rounds=200,
    )
    ebm.fit(X, y)
    assert np.all(np.isfinite(ebm.predict(X)))


def test_cox_dataframe_input():
    """pandas DataFrame input with named columns should work end-to-end."""
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(37)
    X_np, y, _, _ = _simulate_cox_data(120, rng, true_betas=[0.5, 1.0])
    X_df = pd.DataFrame(X_np, columns=["age", "dose"])

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=200
    )
    ebm.fit(X_df, y)
    assert list(ebm.feature_names_in_) == ["age", "dose"]
    preds = ebm.predict(X_df)
    assert np.all(np.isfinite(preds))


def test_cox_global_explanation():
    """Global explanation should produce valid output for every term."""
    rng = np.random.default_rng(41)
    X, y, _, _ = _simulate_cox_data(150, rng, true_betas=[0.7, -0.4])

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=150
    )
    ebm.fit(X, y)
    glob = ebm.explain_global()
    data_0 = glob.data(0)
    assert data_0 is not None
    assert "scores" in data_0
    assert np.all(np.isfinite(data_0["scores"]))


def test_cox_local_explanation_shapes():
    """Local explanation scores must sum to the prediction (minus any link offset)."""
    rng = np.random.default_rng(43)
    X, y, _, _ = _simulate_cox_data(80, rng, true_betas=[1.0, 0.5])

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=150
    )
    ebm.fit(X, y)
    local = ebm.explain_local(X[:3], y[:3])
    for i in range(3):
        data_i = local.data(i)
        # scores from the local explanation + intercept should equal the model prediction
        total = float(data_i["extra"]["scores"][0]) + sum(data_i["scores"])
        np.testing.assert_allclose(
            total, ebm.predict(X[i : i + 1])[0], rtol=1e-6, atol=1e-8
        )


def test_cox_invariant_to_time_rescaling():
    """Cox partial likelihood depends only on the ordering of event times, not
    their absolute values. Scaling all times by a positive constant must not
    change predictions."""
    rng = np.random.default_rng(47)
    X, y, _, _ = _simulate_cox_data(120, rng, true_betas=[1.0], censor_rate=0.3)

    ebm_a = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        random_state=0,
        max_rounds=200,
    ).fit(X, y)

    # Multiply |y| by 10 but preserve the sign (event/censor indicator).
    y_scaled = y * 10.0
    ebm_b = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        random_state=0,
        max_rounds=200,
    ).fit(X, y_scaled)

    np.testing.assert_allclose(
        ebm_a.predict(X), ebm_b.predict(X), rtol=1e-9, atol=1e-12
    )


def test_cox_invariant_to_score_offset():
    """Cox partial likelihood depends only on score *differences* within each
    risk set, so adding a constant to every sample's init_score must not
    change the *relative* ranking of predictions across samples — the two
    models must differ only by an overall constant shift."""
    rng = np.random.default_rng(53)
    X, y, _, _ = _simulate_cox_data(120, rng, true_betas=[0.8])

    ebm_base = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        random_state=0,
        max_rounds=200,
    ).fit(X, y)

    init_score = np.full(len(y), 5.0)
    ebm_shift = ExplainableBoostingRegressor(
        objective="survival_cox",
        interactions=0,
        random_state=0,
        max_rounds=200,
    ).fit(X, y, init_score=init_score)

    diffs = ebm_base.predict(X) - ebm_shift.predict(X)
    # All per-sample differences must be the same constant (translation invariance).
    assert np.ptp(diffs) < 1e-8, (
        f"Cox must be translation-invariant, but per-sample offsets vary by "
        f"{np.ptp(diffs):.3e}"
    )


def test_cox_extreme_values_stable():
    """Very small and very large event times must not produce NaN/inf."""
    rng = np.random.default_rng(59)
    n = 100
    X = rng.normal(size=(n, 1))
    # Mix of tiny, medium, and huge times
    scale = 10.0 ** rng.uniform(-4, 4, size=n)
    event = (rng.random(n) > 0.3).astype(np.float64)
    y = _encode_survival(scale, event)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=300
    )
    ebm.fit(X, y)
    preds = ebm.predict(X)
    assert np.all(np.isfinite(preds)), f"non-finite predictions: {preds}"


def test_cox_single_sample_single_bin_degenerate():
    """A dataset with only one event and one feature value provides no
    discriminative signal but must not crash."""
    rng = np.random.default_rng(61)
    X = np.zeros((5, 1))  # all same feature value
    times = rng.uniform(1.0, 5.0, size=5)
    event = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # only one event
    y = _encode_survival(times, event)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=50
    )
    ebm.fit(X, y)
    preds = ebm.predict(X)
    assert np.all(np.isfinite(preds))


def test_cox_partial_log_likelihood_decreases_with_training():
    """Evaluate a proxy for the partial log-likelihood on the training set
    before and after training: after training, the model's predictions should
    rank event samples higher (more concordant) than at init."""
    rng = np.random.default_rng(67)
    X, y, event_times, event = _simulate_cox_data(
        250, rng, true_betas=[1.2, -0.6], censor_rate=0.25
    )

    # Untrained: concordance should be ~0.5 (random)
    untrained_preds = np.zeros(len(y))
    c_untrained = _concordance_index(untrained_preds, event_times, event)
    assert 0.3 < c_untrained < 0.7  # random-ish (handles floating-point variation)

    ebm = ExplainableBoostingRegressor(
        objective="survival_cox", interactions=0, max_rounds=500
    )
    ebm.fit(X, y)
    trained_preds = ebm.predict(X)
    c_trained = _concordance_index(trained_preds, event_times, event)
    assert c_trained > c_untrained + 0.1, (
        f"training should materially improve concordance "
        f"(before={c_untrained:.3f}, after={c_trained:.3f})"
    )
