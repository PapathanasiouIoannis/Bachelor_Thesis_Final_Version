"""Shared PyTorch MLP definition and loading helpers."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
import torch.nn as nn


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int] | tuple[int, ...], dropout_rate: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(size)))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(float(dropout_rate)))
            in_dim = int(size)

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mlp_params(params_path: str | Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)
    if isinstance(params.get("hidden_layer_sizes"), str):
        params["hidden_layer_sizes"] = ast.literal_eval(params["hidden_layer_sizes"])
    return params


def build_mlp(input_dim: int, params: dict, device: torch.device | str) -> DynamicMLP:
    return DynamicMLP(
        input_dim=input_dim,
        hidden_sizes=params["hidden_layer_sizes"],
        dropout_rate=params["dropout_rate"],
    ).to(device)


def load_mlp_model(
    params_path: str | Path,
    weights_path: str | Path,
    input_dim: int,
    device: torch.device | str,
) -> DynamicMLP:
    model = build_mlp(input_dim=input_dim, params=load_mlp_params(params_path), device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model
