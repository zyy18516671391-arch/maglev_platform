import torch


def evaluate(model, seq, t, I1, I2, target):

    model.eval()

    with torch.no_grad():

        pred = model(seq, t, I1, I2)

        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

        # R2
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-6)
        r2 = r2.item()

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }