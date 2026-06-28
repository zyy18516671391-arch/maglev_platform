import torch
from core.model import MaglevModel
from training.trainer import Trainer
from core.evaluator import evaluate

class MaglevService:
    def __init__(self, params):
        self.params = params
        self.model = MaglevModel()

    def train(self, seq, t, I1, I2, target, mode="pinn", epochs=600):
        use_physics = (mode == "pinn")
        trainer = Trainer(self.model, self.params, use_physics=use_physics)
        self.model, history = trainer.train(seq, t, I1, I2, target, epochs=epochs)
        return history

    def predict(self, seq, t, I1, I2):
        with torch.no_grad():
            return self.model(seq, t, I1, I2)

    def evaluate(self, seq, t, I1, I2, target):
        return evaluate(self.model, seq, t, I1, I2, target)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class MaglevService:
    """Application service wrapper for the upgraded multi-magnet model."""

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

    def train(self, seq, t, currents, target, beam_disp=None, mode="ae_pinn", epochs=600):
        trainer = Trainer(self.model, self.params, mode=mode)
        self.model, history = trainer.train(seq, t, currents, target, beam_disp=beam_disp, epochs=epochs)
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
