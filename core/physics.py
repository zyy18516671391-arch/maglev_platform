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


def compute_structural_dynamics_loss(beam_disp, t, electromagnetic_load, params):
    """Simplified rail-beam dynamics scaffold with a slot ripple term."""
    if beam_disp is None or not params.get("use_structural_loss", False):
        return torch.tensor(0.0, device=t.device, dtype=t.dtype)

    beam_state = beam_disp.mean(dim=1, keepdim=True)
    beam_velocity = _safe_grad(beam_state, t)
    beam_accel = _safe_grad(beam_velocity, t)
    load = electromagnetic_load.mean(dim=1, keepdim=True)

    rho_a = params.get("rho_a", 1.0)
    damping = params.get("beam_c", 0.08)
    stiffness = params.get("beam_k", 8.0)
    slot_amp = params.get("slot_amp", 0.03)
    slot_pitch = params.get("slot_pitch", 0.6)
    slot_force = slot_amp * torch.sin(2 * torch.pi * t / max(slot_pitch, 1e-6))

    residual = rho_a * beam_accel + damping * beam_velocity + stiffness * beam_state - load - slot_force
    return torch.mean(residual ** 2)


def compute_physics_loss(
    pred,
    t,
    currents,
    params,
    initial_gap=None,
    beam_disp=None,
    return_components=False,
):
    """Compute coupled multi-magnet PINN residuals.

    `pred` and `currents` use shape [batch, num_magnets]. The balance residual
    keeps the engineering electromagnetic-force model F=kI^2/gap^2, then adds
    adjacent-magnet coupling, an optional initial-condition term, and an optional
    structural-dynamics scaffold.
    """
    gaps = torch.clamp(pred, min=1e-4)
    currents = currents.reshape_as(gaps)
    device = gaps.device

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
    balance_loss = torch.mean(balance ** 2) * params.get("physics_scale", 1e-3)

    if initial_gap is None:
        initial_loss = torch.tensor(0.0, device=device, dtype=gaps.dtype)
    else:
        initial_ref = initial_gap.reshape(1, -1).to(device=device, dtype=gaps.dtype)
        initial_loss = torch.mean((gaps[:1] - initial_ref) ** 2)

    structural_loss = compute_structural_dynamics_loss(beam_disp, t, electromagnetic_force, params)
    total = (
        balance_loss
        + params.get("initial_weight", 1.0) * initial_loss
        + params.get("structural_weight", 0.1) * structural_loss
    )

    components = {
        "balance": balance_loss,
        "initial": initial_loss,
        "structural": structural_loss,
        "total": total,
    }
    if return_components:
        return total, components
    return total
