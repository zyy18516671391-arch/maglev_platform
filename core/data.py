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
    beam_field: torch.Tensor
    x_grid: torch.Tensor
    field_potential: torch.Tensor
    flux_density: torch.Tensor
    boundary: dict[str, torch.Tensor]
    magnet_positions: torch.Tensor
    split_indices: dict[str, torch.Tensor]
    raw_features: torch.Tensor
    feature_columns: list[str]


def _make_split_indices(length, train_ratio=0.6, val_ratio=0.2):
    train_end = max(1, int(length * train_ratio))
    val_end = max(train_end + 1, int(length * (train_ratio + val_ratio)))
    val_end = min(val_end, length)
    indices = torch.arange(length)
    return {
        "train": indices[:train_end].long(),
        "val": indices[train_end:val_end].long(),
        "test": indices[val_end:].long(),
    }


def _project_magnet_values_to_grid(values, magnet_positions, x_grid):
    """Linearly project [time, magnets] values to a denser beam grid."""
    if values.shape[1] == len(x_grid):
        return values
    projected = []
    for x in x_grid:
        if x <= magnet_positions[0]:
            projected.append(values[:, 0])
        elif x >= magnet_positions[-1]:
            projected.append(values[:, -1])
        else:
            right = torch.searchsorted(magnet_positions, x).item()
            left = right - 1
            span = magnet_positions[right] - magnet_positions[left]
            weight = (x - magnet_positions[left]) / (span + 1e-8)
            projected.append(values[:, left] * (1 - weight) + values[:, right] * weight)
    return torch.stack(projected, dim=1)


def _window_tensor(
    features,
    t,
    currents,
    target_gap,
    beam_disp,
    window,
    beam_field=None,
    x_grid=None,
    field_potential=None,
    flux_density=None,
    boundary=None,
    magnet_positions=None,
):
    seq = []
    for start in range(features.shape[0] - window + 1):
        seq.append(features[start:start + window])

    offset = window - 1
    steps, num_magnets = target_gap.shape
    if x_grid is None:
        x_grid = torch.linspace(0.0, 1.0, 33)
    if magnet_positions is None:
        magnet_positions = torch.linspace(0.0, 1.0, num_magnets)
    if beam_field is None:
        beam_field = beam_disp
    if field_potential is None:
        field_potential = currents / (target_gap + 0.05)
    if flux_density is None:
        flux_density = field_potential
    if boundary is None:
        boundary = {
            "gap_initial": target_gap[0].clone(),
            "beam_left": beam_field[:, 0:1].clone(),
            "beam_right": beam_field[:, -1:].clone(),
            "field_left": field_potential[:, 0:1].clone(),
            "field_right": field_potential[:, -1:].clone(),
        }
    aligned_boundary = {}
    for key, value in boundary.items():
        if value.ndim > 1 and value.shape[0] == len(t):
            aligned_boundary[key] = value[offset:]
        else:
            aligned_boundary[key] = value

    return DatasetBundle(
        seq=torch.stack(seq).float(),
        t=t[offset:].float(),
        currents=currents[offset:].float(),
        target_gap=target_gap[offset:].float(),
        beam_disp=beam_disp[offset:].float(),
        beam_field=beam_field[offset:].float(),
        x_grid=x_grid.float(),
        field_potential=field_potential[offset:].float(),
        flux_density=flux_density[offset:].float(),
        boundary={key: value.float() for key, value in aligned_boundary.items()},
        magnet_positions=magnet_positions.float(),
        split_indices=_make_split_indices(len(t) - offset),
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
    beam_points=33,
):
    """Generate a compact multi-magnet dataset for prototype experiments."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 6.0, total_steps).view(-1, 1)
    magnet_positions = torch.linspace(0.0, 1.0, num_magnets)
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
    x_grid = torch.linspace(0.0, 1.0, beam_points)
    x_phase = x_grid.view(1, -1)
    beam_field = (
        0.012 * torch.sin(torch.pi * x_phase) * torch.sin(1.1 * t)
        + 0.004 * torch.sin(2 * torch.pi * x_phase) * torch.cos(1.8 * t)
    )
    accel = -0.035 * (1.4 ** 2) * torch.sin(1.4 * t + phase)
    magnet_shape = torch.sin(torch.pi * magnet_positions.view(1, -1))
    field_potential = currents / (clean_gap + 0.05) + 0.03 * magnet_shape
    flux = field_potential + 0.04 * neighbor_effect
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
        beam_field = beam_field[indices]
        field_potential = field_potential[indices]
        flux = flux[indices]
        features = features[indices]

    boundary = {
        "gap_initial": clean_gap[0].clone(),
        "beam_left": beam_field[:, 0:1].clone(),
        "beam_right": beam_field[:, -1:].clone(),
        "field_left": field_potential[:, 0:1].clone(),
        "field_right": field_potential[:, -1:].clone(),
    }

    return _window_tensor(
        features,
        t,
        currents,
        clean_gap,
        beam_disp,
        window,
        beam_field=beam_field,
        x_grid=x_grid,
        field_potential=field_potential,
        flux_density=flux,
        boundary=boundary,
        magnet_positions=magnet_positions,
    )


def load_csv_dataset(file_like, num_magnets=4, window=8, beam_points=33):
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
    beam_field_cols = []
    field_potential_cols = []
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
        if f"beam_field{i}" in df.columns:
            beam_field_i = torch.tensor(df[f"beam_field{i}"].to_numpy(), dtype=torch.float32)
        else:
            beam_field_i = beam_i
        if f"field_potential{i}" in df.columns:
            potential_i = torch.tensor(df[f"field_potential{i}"].to_numpy(), dtype=torch.float32)
        else:
            potential_i = flux_i
        beam_field_cols.append(beam_field_i)
        field_potential_cols.append(potential_i)

    currents_t = torch.stack(currents, dim=1)
    gaps_t = torch.stack(gaps, dim=1)
    beam_t = torch.stack(beam, dim=1)
    beam_field = torch.stack(beam_field_cols, dim=1)
    field_potential = torch.stack(field_potential_cols, dim=1)
    flux_density = torch.stack(flux, dim=1)
    features = torch.stack(
        [currents_t, gaps_t, flux_density, torch.stack(accel, dim=1), beam_t, torch.stack(temp, dim=1)],
        dim=-1,
    )

    magnet_positions = torch.linspace(0.0, 1.0, num_magnets)
    x_grid = torch.linspace(0.0, 1.0, beam_points)
    if beam_field.shape[1] != beam_points:
        beam_field = _project_magnet_values_to_grid(beam_field, magnet_positions, x_grid)
    boundary = {
        "gap_initial": gaps_t[0].clone(),
        "beam_left": beam_field[:, 0:1].clone(),
        "beam_right": beam_field[:, -1:].clone(),
        "field_left": field_potential[:, 0:1].clone(),
        "field_right": field_potential[:, -1:].clone(),
    }

    return _window_tensor(
        features,
        t,
        currents_t,
        gaps_t,
        beam_t,
        window,
        beam_field=beam_field,
        x_grid=x_grid,
        field_potential=field_potential,
        flux_density=flux_density,
        boundary=boundary,
        magnet_positions=magnet_positions,
    )
