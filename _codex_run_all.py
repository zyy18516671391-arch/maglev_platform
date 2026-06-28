import ast
from pathlib import Path

import torch

from core.data import FEATURE_COLUMNS, generate_multimagnet_dataset
from core.baselines import newmark_beta_baseline
from core.model import MaglevModel
from core.physics import compute_magnetic_field_loss, compute_beam_pde_loss, compute_physics_loss
from services.maglev_service import MaglevService


PYTHON_FILES = [
    "core/data.py",
    "core/baselines.py",
    "core/evaluator.py",
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
    )
    assert bundle.seq.shape[1:] == (6, 3, len(FEATURE_COLUMNS))
    assert bundle.target_gap.shape[1] == 3

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
        "initial_weight": 1.0,
        "beam_ei": 8.0,
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
        field_potential=bundle.field_potential,
        flux_density=bundle.flux_density,
        boundary=bundle.boundary,
    )
    assert torch.isfinite(physics_loss)
    beam_loss = compute_beam_pde_loss(bundle.beam_field, t, bundle.x_grid, output["gap"], params)
    field_loss = compute_magnetic_field_loss(
        output["gap"],
        bundle.currents,
        bundle.x_grid,
        bundle.flux_density,
        bundle.boundary,
        params,
    )
    assert torch.isfinite(beam_loss)
    assert torch.isfinite(field_loss)

    baseline, elapsed_ms = newmark_beta_baseline(bundle.t, bundle.currents, params, initial_gap=bundle.target_gap[0])
    assert baseline.shape == bundle.target_gap.shape
    assert elapsed_ms >= 0.0

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
            field_potential=bundle.field_potential,
            flux_density=bundle.flux_density,
            boundary=bundle.boundary,
            mode=mode,
            epochs=2,
        )
        pred = service.predict(bundle.seq, bundle.t, bundle.currents)
        assert pred.shape == bundle.target_gap.shape
        assert len(history["total"]) == 2
        if mode == "self_supervised":
            assert history["data"][-1] == 0.0
        print(f"SMOKE_OK {mode} total={history['total'][-1]:.6f}")


if __name__ == "__main__":
    check_ast()
    check_smoke()
    print("ALL_OK")
