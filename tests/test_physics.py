import torch

from core.data import generate_multimagnet_dataset
from core.physics import (
    beam_ei_profile,
    compute_beam_pde_loss,
    compute_magnet_balance_loss,
    compute_magnetic_field_loss,
    slot_force_profile,
)


def base_params():
    return {
        "m": 1.0,
        "g": 9.81,
        "c": 0.45,
        "k": 2.0,
        "epsilon": 0.05,
        "kc": 1.2,
        "physics_scale": 1e-3,
        "use_structural_loss": True,
        "beam_ei": 8.0,
        "slot_amp": 0.03,
        "slot_harmonics": 3,
        "field_force_blend": 0.3,
        "maxwell_force_scale": 1e-5,
        "force_limit": 100.0,
    }


def test_field_force_models_are_trainable_scalars():
    bundle = generate_multimagnet_dataset(num_magnets=3, total_steps=32, window=5, beam_points=33)
    pred = bundle.target_gap.detach().clone().requires_grad_(True)
    t = bundle.t.detach().clone().requires_grad_(True)
    losses = {}

    for mode in ["engineering", "legacy_gradient", "maxwell_stress", "blended"]:
        loss, force, components = compute_magnet_balance_loss(
            pred,
            t,
            bundle.currents,
            {**base_params(), "field_force_model": mode},
            magnet_positions=bundle.magnet_positions,
            field_potential=bundle.field_potential,
        )
        losses[mode] = loss
        assert loss.ndim == 0
        assert force.shape == bundle.target_gap.shape
        assert torch.isfinite(force).all()
        assert "selected_force" in components

    assert not torch.allclose(losses["engineering"], losses["maxwell_stress"])


def test_magnetic_field_loss_returns_maxwell_components():
    bundle = generate_multimagnet_dataset(num_magnets=4, total_steps=28, window=5, beam_points=33)
    loss, components = compute_magnetic_field_loss(
        bundle.target_gap,
        bundle.currents,
        bundle.x_grid,
        bundle.flux_density,
        bundle.boundary,
        base_params(),
        magnet_positions=bundle.magnet_positions,
        field_potential=bundle.field_potential,
        return_components=True,
    )

    assert torch.isfinite(loss)
    assert components["maxwell_force"].shape == bundle.target_gap.shape
    assert torch.all(components["pressure"] >= 0)


def test_segmented_ei_and_cogging_harmonics_drive_beam_residual():
    bundle = generate_multimagnet_dataset(num_magnets=3, total_steps=32, window=5, beam_points=33)
    params = base_params()
    ei = beam_ei_profile(bundle.x_grid, params)
    slot = slot_force_profile(bundle.x_grid, params, device=bundle.t.device, dtype=bundle.t.dtype)
    load = torch.ones_like(bundle.target_gap)
    loss = compute_beam_pde_loss(
        bundle.beam_field,
        bundle.t.detach().clone().requires_grad_(True),
        bundle.x_grid,
        load,
        params,
        magnet_positions=bundle.magnet_positions,
    )

    assert len(torch.unique(ei)) >= 3
    assert slot.shape == (1, bundle.x_grid.shape[0])
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
