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