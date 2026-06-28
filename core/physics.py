import torch


def compute_physics_loss(pred, t, I1, I2, params):
    # 1. 提取预测值
    x1 = pred[:, 0:1]
    x2 = pred[:, 1:2]

    # 2. 自动微分求解速度与加速度
    # 增加一个小常数 1e-4 提升求导稳定性
    dx1 = torch.autograd.grad(x1, t, torch.ones_like(x1), create_graph=True)[0]
    dx2 = torch.autograd.grad(x2, t, torch.ones_like(x2), create_graph=True)[0]
    ddx1 = torch.autograd.grad(dx1, t, torch.ones_like(dx1), create_graph=True)[0]
    ddx2 = torch.autograd.grad(dx2, t, torch.ones_like(dx2), create_graph=True)[0]

    m, g = params["m"], params["g"]
    k, c = params["k"], params["c"]
    kc, epsilon = params["kc"], params["epsilon"]

    # 3. 真实的电磁力模型 (增加 eps 防止分母过小导致梯度爆炸)
    # 物理经验：电磁力项往往量级非常大，是导致 Loss 炸掉的主因
    F1 = k * I1 ** 2 / (x1 + epsilon) ** 2
    F2 = k * I2 ** 2 / (x2 + epsilon) ** 2

    # 4. 动力学平衡方程残差
    res1 = m * ddx1 + c * dx1 + kc * (x1 - x2) - (F1 - m * g)
    res2 = m * ddx2 + c * dx2 + kc * (x2 - x1) - (F2 - m * g)

    # =========================================================
    # 【核心调整：物理量级平滑】
    # =========================================================
    # 理由：物理残差(牛顿)通常是 10^1 量级，而位移 MSE 是 10^-3 量级。
    # 我们将其乘以 1e-3 (或 0.001) 使其与数据损失在同一量级起步。

    loss_ph1 = torch.mean(res1 ** 2)
    loss_ph2 = torch.mean(res2 ** 2)

    # 使用 Huber 损失的思想：对极大的物理残差进行惩罚限制，防止带偏模型
    # 简单的做法是直接乘以一个微小的缩放系数
    total_phys_loss = (loss_ph1 + loss_ph2) * 0.001

    return total_phys_loss


def _safe_grad(value, t):
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
    coupling = torch.zeros_like(gaps)
    if gaps.shape[1] <= 1:
        return coupling
    coupling[:, 1:] += stiffness * (gaps[:, 1:] - gaps[:, :-1])
    coupling[:, :-1] += stiffness * (gaps[:, :-1] - gaps[:, 1:])
    return coupling


def compute_structural_dynamics_loss(beam_disp, t, electromagnetic_load, params):
    """Simplified structural dynamics scaffold for the rail beam subsystem."""
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

    pred and currents have shape [batch, num_magnets]. The balance residual
    keeps the current engineering electromagnetic-force model while adding
    adjacent magnet coupling and optional structural dynamics scaffolding.
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
