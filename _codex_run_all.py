import ast
from pathlib import Path

import torch

from core.data import FEATURE_COLUMNS, generate_multimagnet_dataset
from core.model import MaglevModel
from core.physics import compute_physics_loss
from services.maglev_service import MaglevService


PYTHON_FILES = [
    "core/data.py",
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
        "initial_weight": 1.0,
        "lr": 0.001,
    }

    model = MaglevModel(num_magnets=3, feature_dim=len(FEATURE_COLUMNS), window=6, latent_dim=16)
    t = bundle.t.detach().clone().requires_grad_(True)
    output = model(bundle.seq, t, bundle.currents, return_aux=True)
    assert output["gap"].shape == bundle.target_gap.shape
    assert output["reconstruction"].shape == bundle.seq.shape
    physics_loss = compute_physics_loss(
        output["gap"],
        t,
        bundle.currents,
        params,
        initial_gap=bundle.target_gap[0],
        beam_disp=bundle.beam_disp,
    )
    assert torch.isfinite(physics_loss)

    for mode in ["data_only", "pinn", "ae_pinn"]:
        service = MaglevService(params, num_magnets=3, feature_dim=len(FEATURE_COLUMNS), window=6, latent_dim=16)
        history = service.train(
            bundle.seq,
            bundle.t,
            bundle.currents,
            bundle.target_gap,
            beam_disp=bundle.beam_disp,
            mode=mode,
            epochs=2,
        )
        pred = service.predict(bundle.seq, bundle.t, bundle.currents)
        assert pred.shape == bundle.target_gap.shape
        assert len(history["total"]) == 2
        print(f"SMOKE_OK {mode} total={history['total'][-1]:.6f}")


if __name__ == "__main__":
    check_ast()
    check_smoke()
    print("ALL_OK")
