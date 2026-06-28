import io

import pandas as pd

from core.data import generate_fem_like_validation_dataset, load_fem_validation_csv
from core.evaluator import evaluate_splits
from core.model import MaglevModel


def test_fem_like_dataset_has_reference_force_and_cases():
    bundle = generate_fem_like_validation_dataset(num_magnets=3, steps_per_case=24, window=5, beam_points=33)

    assert bundle.source_type == "fem_like"
    assert bundle.force_ref is not None
    assert bundle.force_ref.shape == bundle.target_gap.shape
    assert bundle.case_ids.shape[0] == bundle.target_gap.shape[0]
    assert set(bundle.split_indices) == {"train", "val", "test"}
    assert len(bundle.case_ids.unique()) == 3

    model = MaglevModel(num_magnets=3, feature_dim=len(bundle.feature_columns), window=5, latent_dim=12)
    metrics = evaluate_splits(model, bundle)
    assert set(metrics) == {"train", "val", "test"}
    assert all("MSE" in split_metrics for split_metrics in metrics.values())


def test_long_form_fem_csv_loader():
    rows = []
    for case_id in [10, 20]:
        for step in range(8):
            for magnet_id, x in enumerate([0.0, 0.5, 1.0], start=1):
                current = 2.0 + 0.1 * step + 0.05 * magnet_id
                gap = 0.45 + 0.01 * magnet_id
                potential = current / (gap + 0.05)
                rows.append(
                    {
                        "case_id": case_id,
                        "time": step * 0.1,
                        "magnet_id": magnet_id,
                        "x": x,
                        "current": current,
                        "gap": gap,
                        "flux_density": potential * 0.8,
                        "field_potential": potential,
                        "force_ref": current ** 2 / (gap + 0.05) ** 2,
                        "beam_disp": 0.001 * step,
                    }
                )
    csv_buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    bundle = load_fem_validation_csv(csv_buffer, window=4, beam_points=17)

    assert bundle.source_type == "fem_csv"
    assert bundle.seq.shape[2] == 3
    assert bundle.x_grid.shape == (17,)
    assert bundle.force_ref.shape == bundle.target_gap.shape
    assert len(bundle.case_ids.unique()) == 2
