import torch

from core.field import compute_field_quantities, select_field_force


def test_maxwell_force_is_finite_nonnegative_and_limited():
    gaps = torch.full((6, 4), 0.45)
    currents = torch.linspace(1.8, 2.4, 24).reshape(6, 4)
    x_grid = torch.linspace(0.0, 1.0, 4)
    params = {"force_limit": 0.5, "maxwell_force_scale": 1e-5}

    quantities = compute_field_quantities(gaps, currents, x_grid, params)

    assert quantities["flux_density"].shape == gaps.shape
    assert quantities["maxwell_force"].shape == gaps.shape
    assert torch.isfinite(quantities["pressure"]).all()
    assert torch.all(quantities["pressure"] >= 0)
    assert torch.all(quantities["maxwell_force"] <= params["force_limit"] + 1e-6)


def test_field_force_modes_are_selectable():
    gaps = torch.full((5, 3), 0.5)
    currents = torch.tensor(
        [
            [2.0, 2.2, 2.4],
            [2.1, 2.3, 2.5],
            [2.2, 2.4, 2.6],
            [2.3, 2.5, 2.7],
            [2.4, 2.6, 2.8],
        ]
    )
    x_grid = torch.linspace(0.0, 1.0, 3)
    engineering = torch.ones_like(gaps) * 7.0
    params = {"field_force_blend": 0.3, "maxwell_force_scale": 1e-5, "force_limit": 100.0}

    forces = {}
    for mode in ["engineering", "legacy_gradient", "maxwell_stress", "blended"]:
        selected, components = select_field_force(
            gaps,
            currents,
            x_grid,
            {**params, "field_force_model": mode},
            engineering_force=engineering,
        )
        forces[mode] = selected
        assert selected.shape == gaps.shape
        assert torch.isfinite(selected).all()
        assert "maxwell_force" in components

    assert torch.allclose(forces["engineering"], engineering)
    assert not torch.allclose(forces["engineering"], forces["maxwell_stress"])
    assert not torch.allclose(forces["legacy_gradient"], forces["blended"])
