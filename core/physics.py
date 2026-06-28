import torch


def _safe_grad(value, t):
    """Return d(value)/dt, or zeros when value is not connected to t."""
    if not value.requires_grad:
        return torch.zeros_like(value)
    grad = torch.autograd.grad(
        value,
        t,
        torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(value)
    return grad


def _adjacent_coupling(gaps, stiffness):
    """Nearest-neighbor coupling force for a row of electromagnets."""
    coupling = torch.zeros_like(gaps)
    if gaps.shape[1] <= 1:
        return coupling
    coupling[:, 1:] += stiffness * (gaps[:, 1:] - gaps[:, :-1])
    coupling[:, :-1] += stiffness * (gaps[:, :-1] - gaps[:, 1:])
    return coupling


def _spatial_second(value, x_grid):
    if value is None or value.shape[1] < 3:
        return torch.zeros_like(value)
    dx = torch.mean(x_grid[1:] - x_grid[:-1]).to(device=value.device, dtype=value.dtype)
    second = torch.zeros_like(value)
    second[:, 1:-1] = (value[:, 2:] - 2 * value[:, 1:-1] + value[:, :-2]) / (dx ** 2 + 1e-8)
    return second


def _spatial_fourth(value, x_grid):
    if value is None or value.shape[1] < 5:
        return torch.zeros_like(value)
    return _spatial_second(_spatial_second(value, x_grid), x_grid)


def _spatial_first(value, x_grid):
    if value is None or value.shape[1] < 2:
        return torch.zeros_like(value)
    dx = torch.mean(x_grid[1:] - x_grid[:-1]).to(device=value.device, dtype=value.dtype)
    first = torch.zeros_like(value)
    first[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / (2 * dx + 1e-8)
    first[:, 0] = (value[:, 1] - value[:, 0]) / (dx + 1e-8)
    first[:, -1] = (value[:, -1] - value[:, -2]) / (dx + 1e-8)
    return first


def compute_magnet_balance_loss(pred, t, currents, params):
    """Multi-magnet force-balance residual."""
    gaps = torch.clamp(pred, min=1e-4)
    currents = currents.reshape_as(gaps)

    m = params.get("m", 1.0)
    g = params.get("g", 9.81)
    c = params.get("c", 0.5)
    k = params.get("k", 2.0)
    epsilon = params.get("epsilon", 0.05)
    kc = params.get("kc", 1.5)

    velocities = []
    accelerations = []
    for i in range(gaps.shape[1]):
        velocity_i = _safe_grad(gaps[:, i:i + 1], t)
        accel_i = _safe_grad(velocity_i, t)
        velocities.append(velocity_i)
        accelerations.append(accel_i)
    velocity = torch.cat(velocities, dim=1)
    acceleration = torch.cat(accelerations, dim=1)

    electromagnetic_force = k * currents ** 2 / (gaps + epsilon) ** 2
    coupling_force = _adjacent_coupling(gaps, kc)
    balance = m * acceleration + c * velocity + coupling_force - (electromagnetic_force - m * g)
    loss = torch.mean(balance ** 2) * params.get("physics_scale", 1e-3)
    return loss, electromagnetic_force


def compute_beam_pde_loss(beam_field, t, x_grid, electromagnetic_load, params):
    """Euler-Bernoulli beam PDE residual driven by electromagnetic load."""
    if beam_field is None or x_grid is None or not params.get("use_structural_loss", False):
        return torch.tensor(0.0, device=t.device, dtype=t.dtype)

    beam_velocity = _safe_grad(beam_field, t)
    beam_accel = _safe_grad(beam_velocity, t)
    bending = _spatial_fourth(beam_field, x_grid)
    rho_a = params.get("rho_a", 1.0)
    damping = params.get("beam_c", 0.08)
    ei = params.get("beam_ei", params.get("beam_k", 8.0))
    slot_amp = params.get("slot_amp", 0.03)
    slot_pitch = params.get("slot_pitch", 0.6)
    slot_force = slot_amp * torch.sin(2 * torch.pi * x_grid.to(t.device).reshape(1, -1) / max(slot_pitch, 1e-6))
    load = electromagnetic_load - electromagnetic_load.mean(dim=1, keepdim=True)

    residual = rho_a * beam_accel + damping * beam_velocity + ei * bending - load - slot_force
    return torch.mean(residual ** 2)


def compute_magnetic_field_loss(pred, currents, x_grid, flux_density, boundary, params):
    """Quasi-1D scalar magnetic-potential residual.

    This is a lightweight Maxwell-style proxy: predicted scalar magnetic
    potential is derived from current and gap, its spatial derivative defines a
    proxy flux density, and both flux consistency plus divergence smoothness are
    penalized.
    """
    if x_grid is None:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    epsilon = params.get("epsilon", 0.05)
    predicted_potential = currents.reshape_as(pred) / (torch.clamp(pred, min=1e-4) + epsilon)
    predicted_flux = -_spatial_first(predicted_potential, x_grid)
    divergence = _spatial_first(predicted_flux, x_grid)
    divergence_loss = torch.mean(divergence ** 2)

    if flux_density is None:
        flux_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        flux_ref = flux_density.to(device=pred.device, dtype=pred.dtype)
        flux_loss = torch.mean((predicted_flux - flux_ref) ** 2) * params.get("field_flux_scale", 1e-3)

    if boundary is None or "field_left" not in boundary or "field_right" not in boundary:
        boundary_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        left = boundary["field_left"][-pred.shape[0]:].to(device=pred.device, dtype=pred.dtype)
        right = boundary["field_right"][-pred.shape[0]:].to(device=pred.device, dtype=pred.dtype)
        boundary_loss = torch.mean((predicted_potential[:, 0:1] - left) ** 2)
        boundary_loss = boundary_loss + torch.mean((predicted_potential[:, -1:] - right) ** 2)
        boundary_loss = boundary_loss * params.get("field_boundary_scale", 1e-3)

    return divergence_loss * params.get("field_div_scale", 1e-4) + flux_loss + boundary_loss


def compute_initial_boundary_loss(pred, initial_gap=None, beam_field=None, boundary=None):
    """Initial gap and structural boundary-condition residual."""
    device = pred.device
    dtype = pred.dtype
    loss = torch.tensor(0.0, device=device, dtype=dtype)
    if initial_gap is not None:
        initial_ref = initial_gap.reshape(1, -1).to(device=device, dtype=dtype)
        loss = loss + torch.mean((pred[:1] - initial_ref) ** 2)

    if beam_field is not None and boundary is not None:
        if "beam_left" in boundary:
            left = boundary["beam_left"][-beam_field.shape[0]:].to(device=device, dtype=dtype)
            loss = loss + torch.mean((beam_field[:, 0:1] - left) ** 2)
        if "beam_right" in boundary:
            right = boundary["beam_right"][-beam_field.shape[0]:].to(device=device, dtype=dtype)
            loss = loss + torch.mean((beam_field[:, -1:] - right) ** 2)
    return loss


def compute_physics_loss(
    pred,
    t,
    currents,
    params,
    initial_gap=None,
    beam_disp=None,
    beam_field=None,
    x_grid=None,
    field_potential=None,
    flux_density=None,
    boundary=None,
    return_components=False,
):
    """Compute coupled multi-magnet PINN residuals.

    `pred` and `currents` use shape [batch, num_magnets]. The balance residual
    keeps the engineering electromagnetic-force model F=kI^2/gap^2, then adds
    adjacent-magnet coupling, an optional initial-condition term, and an optional
    structural-dynamics scaffold.
    """
    if beam_field is None:
        beam_field = beam_disp

    balance_loss, electromagnetic_force = compute_magnet_balance_loss(pred, t, currents, params)
    beam_loss = compute_beam_pde_loss(beam_field, t, x_grid, electromagnetic_force, params)
    field_loss = compute_magnetic_field_loss(pred, currents, x_grid, flux_density, boundary, params)
    initial_loss = compute_initial_boundary_loss(pred, initial_gap=initial_gap, beam_field=beam_field, boundary=boundary)
    total = (
        balance_loss
        + params.get("initial_weight", 1.0) * initial_loss
        + params.get("structural_weight", 0.1) * beam_loss
        + params.get("field_weight", 0.05) * field_loss
    )

    components = {
        "balance": balance_loss,
        "initial": initial_loss,
        "structural": beam_loss,
        "field": field_loss,
        "total": total,
    }
    if return_components:
        return total, components
    return total
