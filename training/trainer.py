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