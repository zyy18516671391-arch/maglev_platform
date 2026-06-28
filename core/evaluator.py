import torch


def regression_metrics(target, pred):
    """Evaluate all magnet gaps with aggregate regression metrics."""
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-6)).item()

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate(model, seq, t, currents, target):
    """Evaluate a model on one tensor slice."""
    model.eval()
    with torch.no_grad():
        pred = model(seq, t, currents)
    return regression_metrics(target, pred)


def evaluate_splits(model, dataset):
    """Evaluate train/validation/test time splits with the same model."""
    metrics = {}
    model.eval()
    with torch.no_grad():
        pred = model(dataset.seq, dataset.t, dataset.currents)
    for split_name, indices in dataset.split_indices.items():
        if len(indices) == 0:
            continue
        metrics[split_name] = regression_metrics(dataset.target_gap[indices], pred[indices])
    return metrics
