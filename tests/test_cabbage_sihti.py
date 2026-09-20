import json
from pathlib import Path

import numpy as np

from cabbage_sihti import (
    compare_images,
    fit_model,
    gradient_anisotropy,
    load_model,
    save_model,
    spectral_measure,
    synthesize,
    synthetic_reference,
)


def test_spectral_measure_is_probability_distribution():
    x = np.random.default_rng(0).standard_normal((32, 40))
    bins = spectral_measure(x, max_bins=128)
    assert len(bins) <= 128
    assert abs(sum(b["p"] for b in bins) - 1.0) < 1e-10
    assert all(b["p"] > 0 for b in bins)


def test_model_round_trip(tmp_path: Path):
    ref = synthetic_reference(48, seed=2)
    model = fit_model(ref, max_bins=256)
    path = tmp_path / "model.json"
    save_model(path, model)
    loaded = load_model(path)
    assert loaded["format"] == "cabbage-farm-sihti-v0"
    assert loaded["reference_shape"] == [48, 48]
    assert len(loaded["components"]) == 3


def test_coordinate_addressing_has_exact_overlap():
    ref = synthetic_reference(48, seed=3)
    model = fit_model(ref, max_bins=256)

    whole = synthesize(model, 72, 40, origin_x=100, origin_y=-20, seed=11, modes=128)
    left = synthesize(model, 48, 40, origin_x=100, origin_y=-20, seed=11, modes=128)
    right = synthesize(model, 48, 40, origin_x=124, origin_y=-20, seed=11, modes=128)

    assert np.array_equal(whole[:, :48], left)
    assert np.array_equal(whole[:, 24:72], right)
    assert np.array_equal(left[:, 24:48], right[:, :24])


def test_new_seed_changes_world_not_law():
    ref = synthetic_reference(48, seed=4)
    model = fit_model(ref, max_bins=256)
    a = synthesize(model, 48, 48, seed=1, modes=192)
    b = synthesize(model, 48, 48, seed=2, modes=192)
    assert not np.array_equal(a, b)
    assert abs(gradient_anisotropy(a) - gradient_anisotropy(b)) < 20.0


def test_directional_reference_transfers_directional_bias():
    ref = synthetic_reference(64, seed=5)
    model = fit_model(ref, max_bins=512)
    out = synthesize(model, 96, 96, seed=7, modes=384)
    # Synthetic reference deliberately has pronounced directional smoothing.
    assert gradient_anisotropy(ref) > 1.2
    assert gradient_anisotropy(out) > 1.1


def test_compare_reports_control_metrics():
    ref = synthetic_reference(40, seed=6)
    model = fit_model(ref, max_bins=256)
    out = synthesize(model, 40, 40, seed=9, modes=192)
    result = compare_images(ref, out)
    for key in (
        "log_radial_psd_rmse",
        "rgb_covariance_relative_error",
        "gaussian_residue_energy_rmse",
    ):
        assert np.isfinite(result[key])
        assert result[key] >= 0
