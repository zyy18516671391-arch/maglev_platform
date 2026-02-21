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