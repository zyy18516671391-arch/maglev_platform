import ast
from pathlib import Path

import torch

from core.data import FEATURE_COLUMNS, generate_multimagnet_dataset
from core.baselines import compare_with_newmark, newmark_beta_baseline
from core.evaluator import evaluate_splits
from core.experiments import save_experiment_record
from core.model import MaglevModel
from core.physics import (
    compute_beam_pde_loss,
    compute_magnet_balance_loss,
    compute_magnetic_field_loss,
    compute_physics_loss,
    project_magnet_load_to_beam_grid,
)
from services.maglev_service import MaglevService


PYTHON_FILES = [
    "core/data.py",
    "core/baselines.py",
    "core/evaluator.py",
    "core/experiments.py",
    "core/model.py",
    "core/physics.py",
    "services/maglev_service.py",
    "training/trainer.py",
    "ui/app.py",
]


def check_ast():
    for file_name in PYTHON_FILES:
        path = Path(file_name)
        ast.parse(path.read_text(encoding="utf-8"), filename=file_name)
        print(f"AST_OK {file_name}")


def check_smoke():
    torch.manual_seed(7)
    bundle = generate_multimagnet_dataset(
        num_magnets=3,
        total_steps=40,
        window=6,
        noise_level=0.01,
        sample_ratio=0.8,
        seed=7,
        beam_points=33,
    )
    assert bundle.seq.shape[1:] == (6, 3, len(FEATURE_COLUMNS))
    assert bundle.target_gap.shape[1] == 3
    assert bundle.x_grid.shape == (33,)
    assert bundle.beam_field.shape[-1] == 33
    assert set(bundle.split_indices) == {"train", "val", "test"}

    params = {
        "m": 1.0,
        "g": 9.81,
        "c": 0.45,
        "k": 2.0,
        "epsilon": 0.05,
        "kc": 1.2,
        "physics_scale": 1e-3,
        "lambda_physics": 0.1,
        "lambda_reconstruction": 0.02,
        "use_structural_loss": True,
        "structural_weight": 0.1,
        "field_weight": 0.05,
        "field_div_scale": 1e-4,
        "field_flux_scale": 1e-3,
        "field_boundary_scale": 1e-3,
        "field_force_blend": 0.25,
        "field_force_scale": 0.1,
        "initial_weight": 1.0,
        "beam_ei": 8.0,
        "min_gap": 0.02,
        "force_limit": 100.0,
        "lr": 0.001,
    }

    model = MaglevModel(num_magnets=3, feature_dim=len(FEATURE_COLUMNS), window=6, latent_dim=16)
    t = bundle.t.detach().clone().requires_grad_(True)
    output = model(bundle.seq, t, bundle.currents, return_aux=True)
    assert output["gap"].shape == bundle.target_gap.shape
    assert output["reconstruction"].shape == bundle.seq.shape
    assert output["spatial_attention"].shape[-2:] == (3, 3)
    assert output["temporal_attention"].shape[-2:] == (6, 6)
    physics_loss = compute_physics_loss(
        output["gap"],
        t,
        bundle.currents,
        params,
        initial_gap=bundle.target_gap[0],
        beam_disp=bundle.beam_disp,
        beam_field=bundle.beam_field,
        x_grid=bundle.x_grid,
        magnet_positions=bundle.magnet_positions,
        field_potential=bundle.field_potential,
        flux_density=bundle.flux_density,
        boundary=bundle.boundary,
    )
    assert torch.isfinite(physics_loss)
    balance_loss, electromagnetic_force = compute_magnet_balance_loss(
        output["gap"],
        t,
        bundle.currents,
        params,
        x_grid=bundle.x_grid,
        magnet_positions=bundle.magnet_positions,
    )
    projected_load = project_magnet_load_to_beam_grid(electromagnetic_force, bundle.magnet_positions, bundle.x_grid)
    assert projected_load.shape == bundle.beam_field.shape
    beam_loss = compute_beam_pde_loss(
        bundle.beam_field,
        t,
        bundle.x_grid,
        electromagnetic_force,
        params,
        magnet_positions=bundle.magnet_positions,
    )
    field_loss = compute_magnetic_field_loss(
        output["gap"],
        bundle.currents,
        bundle.x_grid,
        bundle.flux_density,
        bundle.boundary,
        params,
        magnet_positions=bundle.magnet_positions,
    )
    assert torch.isfinite(balance_loss)
    assert torch.isfinite(beam_loss)
    assert torch.isfinite(field_loss)
    assert beam_loss.item() > 0.0

    baseline, elapsed_ms = newmark_beta_baseline(bundle.t, bundle.currents, params, initial_gap=bundle.target_gap[0])
    assert baseline.shape == bundle.target_gap.shape
    assert elapsed_ms >= 0.0
    baseline_info = compare_with_newmark(output["gap"].detach(), bundle.target_gap, bundle.t, bundle.currents, params)
    assert baseline_info["method"] == "nonlinear_newmark"

    for mode in ["data_only", "pinn", "ae_pinn", "self_supervised"]:
        service = MaglevService(params, num_magnets=3, feature_dim=len(FEATURE_COLUMNS), window=6, latent_dim=16)
        history = service.train(
            bundle.seq,
            bundle.t,
            bundle.currents,
            bundle.target_gap,
            beam_disp=bundle.beam_disp,
            beam_field=bundle.beam_field,
            x_grid=bundle.x_grid,
            magnet_positions=bundle.magnet_positions,
            field_potential=bundle.field_potential,
            flux_density=bundle.flux_density,
            boundary=bundle.boundary,
            train_indices=bundle.split_indices["train"],
            mode=mode,
            epochs=2,
        )
        pred = service.predict(bundle.seq, bundle.t, bundle.currents)
        assert pred.shape == bundle.target_gap.shape
        assert len(history["total"]) == 2
        assert all(torch.isfinite(torch.tensor(values[-1])) for values in history.values())
        if mode == "self_supervised":
            assert history["data"][-1] == 0.0
        split_metrics = evaluate_splits(service.model, bundle)
        assert set(split_metrics) == {"train", "val", "test"}
        assert all("MSE" in metrics for metrics in split_metrics.values())
        print(f"SMOKE_OK {mode} total={history['total'][-1]:.6f}")

    record_paths = save_experiment_record(
        {
            "mode": "smoke",
            "elapsed_ms": elapsed_ms,
            "final_loss": history["total"][-1],
            "split_metrics": split_metrics,
            "params": params,
        }
    )
    assert Path(record_paths["json"]).exists()
    assert Path(record_paths["csv"]).exists()


if __name__ == "__main__":
    check_ast()
    check_smoke()
    print("ALL_OK")
