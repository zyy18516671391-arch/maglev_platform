import math

import torch


def spatial_first(value, x_grid):
    """First-order spatial derivative on an approximately uniform 1D grid."""
    if value is None or value.shape[1] < 2:
        return torch.zeros_like(value)
    grid = x_grid.to(device=value.device, dtype=value.dtype)
    dx = torch.mean(grid[1:] - grid[:-1])
    first = torch.zeros_like(value)
    first[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / (2 * dx + 1e-8)
    first[:, 0] = (value[:, 1] - value[:, 0]) / (dx + 1e-8)
    first[:, -1] = (value[:, -1] - value[:, -2]) / (dx + 1e-8)
    return first


def scalar_potential_from_gap(gaps, currents, params):
    """Quasi-1D scalar magnetic potential proxy A=I/(gap+epsilon)."""
    epsilon = params.get("epsilon", 0.05)
    return currents.reshape_as(gaps) / (torch.clamp(gaps, min=1e-4) + epsilon)


def flux_density_from_potential(potential, x_grid):
    """Proxy flux density B=-dA/dx."""
    return -spatial_first(potential, x_grid)


def maxwell_pressure_from_flux(flux_density, params):
    """Normal magnetic pressure p=B^2/(2*mu0)."""
    mu0 = params.get("mu0", 4 * math.pi * 1e-7)
    return flux_density ** 2 / (2.0 * mu0 + 1e-12)


def maxwell_force_from_flux(flux_density, params):
    """Map Maxwell pressure to bounded magnet force."""
    pressure = maxwell_pressure_from_flux(flux_density, params)
    area = params.get("pole_area", 0.01)
    normal = params.get("force_normal", 1.0)
    scale = params.get("maxwell_force_scale", 1e-5)
    limit = params.get("force_limit", 100.0)
    force = pressure * area * normal * scale
    return torch.clamp(force, min=0.0, max=limit)


def legacy_gradient_force(gaps, currents, x_grid, params):
    """Backward-compatible gradient-squared field-force proxy."""
    potential = scalar_potential_from_gap(gaps, currents, params)
    flux = flux_density_from_potential(potential, x_grid)
    return params.get("field_force_scale", 0.1) * flux ** 2


def compute_field_quantities(gaps, currents, x_grid, params, field_potential=None):
    """Return potential, flux density, magnetic pressure, and Maxwell force."""
    grid = x_grid.to(device=gaps.device, dtype=gaps.dtype)
    if field_potential is None:
        potential = scalar_potential_from_gap(gaps, currents, params)
    else:
        potential = field_potential.to(device=gaps.device, dtype=gaps.dtype).reshape_as(gaps)
    flux = flux_density_from_potential(potential, grid)
    pressure = maxwell_pressure_from_flux(flux, params)
    force = maxwell_force_from_flux(flux, params)
    return {
        "potential": potential,
        "flux_density": flux,
        "pressure": pressure,
        "maxwell_force": force,
    }


def select_field_force(gaps, currents, x_grid, params, engineering_force=None, field_potential=None):
    """Select engineering, legacy, Maxwell, or blended electromagnetic force."""
    if x_grid is None:
        return engineering_force if engineering_force is not None else torch.zeros_like(gaps), {}

    model = params.get("field_force_model", "maxwell_stress")
    grid = x_grid.to(device=gaps.device, dtype=gaps.dtype)
    quantities = compute_field_quantities(gaps, currents, grid, params, field_potential=field_potential)
    maxwell_force = quantities["maxwell_force"]

    if engineering_force is None:
        engineering_force = torch.zeros_like(gaps)
    if model == "engineering":
        selected = engineering_force
    elif model == "legacy_gradient":
        selected = legacy_gradient_force(gaps, currents, grid, params)
    elif model == "blended":
        blend = torch.clamp(
            torch.tensor(params.get("field_force_blend", 0.3), device=gaps.device, dtype=gaps.dtype),
            0.0,
            1.0,
        )
        selected = (1.0 - blend) * engineering_force + blend * maxwell_force
    else:
        selected = maxwell_force

    quantities["selected_force"] = torch.clamp(selected, min=0.0, max=params.get("force_limit", 100.0))
    quantities["engineering_force"] = engineering_force
    return quantities["selected_force"], quantities
