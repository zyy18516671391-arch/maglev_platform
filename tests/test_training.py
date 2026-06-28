import torch

from core.data import FEATURE_COLUMNS, generate_multimagnet_dataset
from services.maglev_service import MaglevService


def test_all_training_modes_smoke_two_epochs():
    bundle = generate_multimagnet_dataset(
        num_magnets=3,
        total_steps=36,
        window=6,
        noise_level=0.01,
        sample_ratio=0.9,
        seed=7,
        beam_points=33,
    )
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
        "field_force_model": "blended",
        "field_force_blend": 0.3,
        "maxwell_force_scale": 1e-5,
        "initial_weight": 1.0,
        "beam_ei": 8.0,
        "min_gap": 0.02,
        "force_limit": 100.0,
        "lr": 0.001,
    }

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
