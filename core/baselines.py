import time

import torch


def newmark_beta_baseline(t, currents, params, initial_gap=None):
    """Solve a compact linearized maglev response with Newmark-beta.

    This baseline is intentionally simple: it uses the same current excitation
    as the PINN pipeline, linearizes the electromagnetic force around the initial
    gap, and integrates a damped second-order system for each magnet.
    """
    start = time.perf_counter()
    t = t.reshape(-1)
    currents = currents.float()
    steps, num_magnets = currents.shape
    device = currents.device
    dtype = currents.dtype

    if initial_gap is None:
        initial_gap = torch.full((num_magnets,), 0.5, device=device, dtype=dtype)
    else:
        initial_gap = initial_gap.reshape(-1).to(device=device, dtype=dtype)

    m = params.get("m", 1.0)
    c = params.get("c", 0.45)
    k_struct = params.get("baseline_k", params.get("beam_k", 8.0))
    force_k = params.get("k", 2.0)
    epsilon = params.get("epsilon", 0.05)
    beta = params.get("newmark_beta", 0.25)
    gamma = params.get("newmark_gamma", 0.5)

    x = torch.zeros((steps, num_magnets), device=device, dtype=dtype)
    v = torch.zeros_like(x)
    a = torch.zeros_like(x)
    x[0] = initial_gap

    nominal_force = force_k * currents[0] ** 2 / (initial_gap + epsilon) ** 2
    a[0] = (nominal_force - c * v[0] - k_struct * x[0]) / m

    for i in range(steps - 1):
        dt = torch.clamp(t[i + 1] - t[i], min=1e-4).to(device=device, dtype=dtype)
        force_next = force_k * currents[i + 1] ** 2 / (initial_gap + epsilon) ** 2

        x_pred = x[i] + dt * v[i] + dt ** 2 * (0.5 - beta) * a[i]
        v_pred = v[i] + dt * (1.0 - gamma) * a[i]
        effective_mass = m + gamma * dt * c + beta * dt ** 2 * k_struct
        a_next = (force_next - c * v_pred - k_struct * x_pred) / effective_mass

        x[i + 1] = x_pred + beta * dt ** 2 * a_next
        v[i + 1] = v_pred + gamma * dt * a_next
        a[i + 1] = a_next

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return x, elapsed_ms


def compare_with_newmark(pred, target, t, currents, params):
    """Return baseline predictions and aggregate comparison metrics."""
    baseline, elapsed_ms = newmark_beta_baseline(t, currents, params, initial_gap=target[0])
    mse = torch.mean((baseline - target) ** 2).item()
    mae = torch.mean(torch.abs(baseline - target)).item()
    pred_mse = torch.mean((pred - target) ** 2).item()
    return {
        "baseline": baseline,
        "elapsed_ms": elapsed_ms,
        "newmark_mse": mse,
        "newmark_mae": mae,
        "model_mse": pred_mse,
    }
