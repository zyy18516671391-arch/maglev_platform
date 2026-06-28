import torch

from core.evaluator import evaluate
from core.model import MaglevModel
from training.trainer import Trainer


class MaglevService:
    """Application service wrapper for the multi-magnet model."""

    def __init__(self, params, num_magnets=4, feature_dim=6, window=8, latent_dim=32):
        self.params = params
        self.num_magnets = num_magnets
        self.feature_dim = feature_dim
        self.window = window
        self.model = MaglevModel(
            num_magnets=num_magnets,
            feature_dim=feature_dim,
            window=window,
            latent_dim=latent_dim,
        )

    def train(
        self,
        seq,
        t,
        currents,
        target,
        beam_disp=None,
        beam_field=None,
        x_grid=None,
        field_potential=None,
        flux_density=None,
        boundary=None,
        mode="ae_pinn",
        epochs=600,
    ):
        trainer = Trainer(self.model, self.params, mode=mode)
        self.model, history = trainer.train(
            seq,
            t,
            currents,
            target,
            beam_disp=beam_disp,
            beam_field=beam_field,
            x_grid=x_grid,
            field_potential=field_potential,
            flux_density=flux_density,
            boundary=boundary,
            epochs=epochs,
        )
        return history

    def predict(self, seq, t, currents):
        self.model.eval()
        with torch.no_grad():
            return self.model(seq, t, currents)

    def predict_full(self, seq, t, currents):
        self.model.eval()
        with torch.no_grad():
            return self.model(seq, t, currents, return_aux=True)

    def evaluate(self, seq, t, currents, target):
        return evaluate(self.model, seq, t, currents, target)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state)
