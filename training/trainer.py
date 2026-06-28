import torch

from core.physics import compute_physics_loss


class Trainer:
    """Trainer for supervised, PINN, autoencoder PINN, and self-supervised modes."""

    def __init__(self, model, params, mode="ae_pinn"):
        self.model = model
        self.params = params
        self.mode = mode
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))

    def train(
        self,
        seq,
        t,
        currents,
        target,
        beam_disp=None,
        beam_field=None,
        x_grid=None,
        magnet_positions=None,
        field_potential=None,
        flux_density=None,
        boundary=None,
        train_indices=None,
        epochs=400,
    ):
        history = {
            "data": [],
            "physics": [],
            "reconstruction": [],
            "initial": [],
            "structural": [],
            "field": [],
            "total": [],
        }
        use_data = self.mode in {"data_only", "pinn", "ae_pinn"}
        use_physics = self.mode in {"pinn", "ae_pinn", "self_supervised"}
        use_reconstruction = self.mode in {"ae_pinn", "self_supervised"}
        initial_gap = target[0].detach()
        if train_indices is not None:
            train_indices = train_indices.long()
            seq = seq[train_indices]
            t = t[train_indices]
            currents = currents[train_indices]
            target = target[train_indices]
            if beam_disp is not None:
                beam_disp = beam_disp[train_indices]
            if beam_field is not None:
                beam_field = beam_field[train_indices]
            if field_potential is not None:
                field_potential = field_potential[train_indices]
            if flux_density is not None:
                flux_density = flux_density[train_indices]
            if boundary is not None:
                boundary = {
                    key: value[train_indices] if value.ndim > 1 and value.shape[0] >= len(train_indices) else value
                    for key, value in boundary.items()
                }

        for _ in range(epochs):
            self.optimizer.zero_grad()
            t_curr = t.detach().clone().requires_grad_(True)
            currents_curr = currents.detach().clone()
            seq_curr = seq.detach().clone()
            target_curr = target.detach().clone()
            beam_curr = None if beam_disp is None else beam_disp.detach().clone()
            beam_field_curr = None if beam_field is None else beam_field.detach().clone()
            flux_curr = None if flux_density is None else flux_density.detach().clone()

            output = self.model(seq_curr, t_curr, currents_curr, return_aux=True)
            pred = output["gap"]
            if use_data:
                loss_data = torch.mean((pred - target_curr) ** 2)
            else:
                loss_data = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

            if use_physics:
                loss_phys, components = compute_physics_loss(
                    pred,
                    t_curr,
                    currents_curr,
                    self.params,
                    initial_gap=initial_gap,
                    beam_disp=beam_curr,
                    beam_field=beam_field_curr,
                    x_grid=x_grid,
                    magnet_positions=magnet_positions,
                    field_potential=field_potential,
                    flux_density=flux_curr,
                    boundary=boundary,
                    return_components=True,
                )
            else:
                zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                loss_phys = zero
                components = {"balance": zero, "initial": zero, "structural": zero, "field": zero, "total": zero}

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
            history["field"].append(components["field"].item())
            history["total"].append(loss.item())

        return self.model, history
