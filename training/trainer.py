import torch
from core.physics import compute_physics_loss


class Trainer:
    def __init__(self, model, params, use_physics=True):
        self.model = model
        self.params = params
        self.use_physics = use_physics
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(self, seq, t, I1, I2, target, epochs=400):
        history = {"data": [], "physics": []}

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # --- 关键修正：确保每个 epoch 的输入都是可求导的新节点 ---
            # 这样就不会因为上一轮 backward 释放了图而报错
            t_curr = t.detach().clone().requires_grad_(True)
            I1_curr = I1.detach().clone()
            I2_curr = I2.detach().clone()

            # 前向传播
            pred = self.model(seq, t_curr, I1_curr, I2_curr)

            # 1. 数据损失
            loss_data = torch.mean((pred - target) ** 2)

            if self.use_physics:
                # 2. 物理损失
                loss_phys = compute_physics_loss(pred, t_curr, I1_curr, I2_curr, self.params)

                # 3. 动态权重 (自适应调整)
                with torch.no_grad():
                    w = loss_data / (loss_phys + 1e-6)
                    w = torch.clamp(w, 0, 0.1)  # 限制物理权重不要带偏模型

                loss = loss_data + w * loss_phys
            else:
                loss_phys = torch.tensor(0.0)
                loss = loss_data

            # 执行反向传播
            loss.backward()
            self.optimizer.step()

            # 记录记录
            history["data"].append(loss_data.item())
            history["physics"].append(loss_phys.item())

        return self.model, history


class Trainer:
    """Trainer for data-only, PINN, and attention-autoencoder PINN modes."""

    def __init__(self, model, params, mode="ae_pinn"):
        self.model = model
        self.params = params
        self.mode = mode
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))

    def train(self, seq, t, currents, target, beam_disp=None, epochs=400):
        history = {
            "data": [],
            "physics": [],
            "reconstruction": [],
            "initial": [],
            "structural": [],
            "total": [],
        }
        use_physics = self.mode in {"pinn", "ae_pinn"}
        use_reconstruction = self.mode == "ae_pinn"
        initial_gap = target[0].detach()

        for _ in range(epochs):
            self.optimizer.zero_grad()
            t_curr = t.detach().clone().requires_grad_(True)
            currents_curr = currents.detach().clone()
            seq_curr = seq.detach().clone()
            target_curr = target.detach().clone()
            beam_curr = None if beam_disp is None else beam_disp.detach().clone()

            output = self.model(seq_curr, t_curr, currents_curr, return_aux=True)
            pred = output["gap"]
            loss_data = torch.mean((pred - target_curr) ** 2)

            if use_physics:
                loss_phys, components = compute_physics_loss(
                    pred,
                    t_curr,
                    currents_curr,
                    self.params,
                    initial_gap=initial_gap,
                    beam_disp=beam_curr,
                    return_components=True,
                )
            else:
                zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                loss_phys = zero
                components = {"balance": zero, "initial": zero, "structural": zero, "total": zero}

            if use_reconstruction:
                loss_rec = torch.mean((output["reconstruction"] - seq_curr) ** 2)
            else:
                loss_rec = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

            loss = (
                loss_data
                + self.params.get("lambda_physics", 0.2) * loss_phys
                + self.params.get("lambda_reconstruction", 0.05) * loss_rec
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.get("grad_clip", 5.0))
            self.optimizer.step()

            history["data"].append(loss_data.item())
            history["physics"].append(loss_phys.item())
            history["reconstruction"].append(loss_rec.item())
            history["initial"].append(components["initial"].item())
            history["structural"].append(components["structural"].item())
            history["total"].append(loss.item())

        return self.model, history
