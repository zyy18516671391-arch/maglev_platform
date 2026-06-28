import math
from dataclasses import dataclass

import pandas as pd
import torch


FEATURE_COLUMNS = ["current", "gap", "flux", "accel", "beam_disp", "temperature"]


@dataclass(frozen=True)
class DatasetBundle:
    seq: torch.Tensor
    t: torch.Tensor
    currents: torch.Tensor
    target_gap: torch.Tensor
    beam_disp: torch.Tensor
    raw_features: torch.Tensor
    feature_columns: list[str]


def _window_tensor(features, t, currents, target_gap, beam_disp, window):
    seq = []
    for start in range(features.shape[0] - window + 1):
        seq.append(features[start:start + window])

    offset = window - 1
    return DatasetBundle(
        seq=torch.stack(seq).float(),
        t=t[offset:].float(),
        currents=currents[offset:].float(),
        target_gap=target_gap[offset:].float(),
        beam_disp=beam_disp[offset:].float(),
        raw_features=features[offset:].float(),
        feature_columns=FEATURE_COLUMNS.copy(),
    )


def generate_multimagnet_dataset(
    num_magnets=4,
    total_steps=240,
    window=8,
    noise_level=0.02,
    sample_ratio=1.0,
    seed=42,
):
    """Generate a compact multi-magnet dataset for prototype experiments."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 6.0, total_steps).view(-1, 1)
    magnet_axis = torch.arange(num_magnets).float().view(1, -1)
    phase = magnet_axis * math.pi / max(num_magnets, 1)

    currents = 2.0 + 0.45 * torch.sin(2.0 * t + phase) + 0.08 * torch.cos(4.0 * t - phase)
    clean_gap = 0.50 + 0.035 * torch.sin(1.4 * t + phase) + 0.012 * torch.cos(2.6 * t - phase)
    neighbor_effect = torch.zeros_like(clean_gap)
    if num_magnets > 1:
        neighbor_effect[:, 1:] += clean_gap[:, :-1] - clean_gap[:, 1:]
        neighbor_effect[:, :-1] += clean_gap[:, 1:] - clean_gap[:, :-1]
    clean_gap = clean_gap + 0.08 * neighbor_effect

    beam_disp = 0.015 * torch.sin(1.1 * t + 0.4 * phase)
    accel = -0.035 * (1.4 ** 2) * torch.sin(1.4 * t + phase)
    flux = currents / (clean_gap + 0.05) + 0.04 * neighbor_effect
    temperature = 25.0 + 1.2 * torch.sin(0.35 * t + phase)

    measured_gap = clean_gap + noise_level * torch.randn(clean_gap.shape, generator=generator)
    measured_flux = flux + noise_level * torch.randn(flux.shape, generator=generator)
    measured_accel = accel + noise_level * torch.randn(accel.shape, generator=generator)

    features = torch.stack(
        [currents, measured_gap, measured_flux, measured_accel, beam_disp, temperature],
        dim=-1,
    )

    if sample_ratio < 1.0:
        keep = max(window + 2, int(total_steps * sample_ratio))
        indices = torch.linspace(0, total_steps - 1, keep).round().long().unique()
        t = t[indices]
        currents = currents[indices]
        clean_gap = clean_gap[indices]
        beam_disp = beam_disp[indices]
        features = features[indices]

    return _window_tensor(features, t, currents, clean_gap, beam_disp, window)


def load_csv_dataset(file_like, num_magnets=4, window=8):
    """Load a CSV into the same tensor shape as the synthetic generator.

    Required columns are `t`, `I1..IN`, and `gap1..gapN`. Optional columns are
    `fluxN`, `accelN`, `beam_dispN`, and `temperatureN`.
    """
    df = pd.read_csv(file_like)
    if "t" not in df.columns:
        raise ValueError("CSV must include a t column.")

    t = torch.tensor(df["t"].to_numpy(), dtype=torch.float32).view(-1, 1)
    currents = []
    gaps = []
    flux = []
    accel = []
    beam = []
    temp = []

    for i in range(1, num_magnets + 1):
        current_col = f"I{i}"
        gap_col = f"gap{i}"
        if current_col not in df.columns or gap_col not in df.columns:
            raise ValueError(f"CSV must include {current_col} and {gap_col}.")
        current_i = torch.tensor(df[current_col].to_numpy(), dtype=torch.float32)
        gap_i = torch.tensor(df[gap_col].to_numpy(), dtype=torch.float32)
        currents.append(current_i)
        gaps.append(gap_i)
        if f"flux{i}" in df.columns:
            flux_i = torch.tensor(df[f"flux{i}"].to_numpy(), dtype=torch.float32)
        else:
            flux_i = current_i / (gap_i + 0.05)
        if f"accel{i}" in df.columns:
            accel_i = torch.tensor(df[f"accel{i}"].to_numpy(), dtype=torch.float32)
        else:
            accel_i = torch.zeros_like(gap_i)
        if f"beam_disp{i}" in df.columns:
            beam_i = torch.tensor(df[f"beam_disp{i}"].to_numpy(), dtype=torch.float32)
        else:
            beam_i = torch.zeros_like(gap_i)
        if f"temperature{i}" in df.columns:
            temp_i = torch.tensor(df[f"temperature{i}"].to_numpy(), dtype=torch.float32)
        else:
            temp_i = torch.full_like(gap_i, 25.0)

        flux.append(flux_i)
        accel.append(accel_i)
        beam.append(beam_i)
        temp.append(temp_i)

    currents_t = torch.stack(currents, dim=1)
    gaps_t = torch.stack(gaps, dim=1)
    beam_t = torch.stack(beam, dim=1)
    features = torch.stack(
        [currents_t, gaps_t, torch.stack(flux, dim=1), torch.stack(accel, dim=1), beam_t, torch.stack(temp, dim=1)],
        dim=-1,
    )

    return _window_tensor(features, t, currents_t, gaps_t, beam_t, window)
