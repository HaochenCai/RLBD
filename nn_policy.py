"""
Neural cut-selection policy: forward only, no training.

- CutPolicyNet: (cut_features Nx5, state_vec 24) -> logits (N,)
- For nn_sample_one: softmax(logits/temp) then Monte Carlo sample one cut.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

CUT_FEATURE_DIM = 5
STATE_VEC_DIM = 24
INPUT_DIM = CUT_FEATURE_DIM + STATE_VEC_DIM  # 5 + 24


class CutPolicyNet(nn.Module):
    """
    Per-cut input [cut_feature (5), state_vec (24)] = 29 dims;
    outputs logit for softmax sampling.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(
        self,
        cut_features: Union[torch.Tensor, np.ndarray],
        state_vec: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        cut_features: (N, 5)
        state_vec: (24,) or (1, 24)
        returns logits: (N,)
        """
        if isinstance(cut_features, np.ndarray):
            cut_features = torch.from_numpy(cut_features).float()
        if isinstance(state_vec, np.ndarray):
            state_vec = torch.from_numpy(state_vec).float()
        device = cut_features.device
        state_vec = state_vec.to(device)
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)  # (1, 24)
        N = cut_features.size(0)
        state_broadcast = state_vec.expand(N, -1)  # (N, 24)
        x = torch.cat([cut_features, state_broadcast], dim=1)  # (N, 29)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x).squeeze(-1)  # (N,)
        return logits


def forward_logits(
    net: CutPolicyNet,
    cut_features: np.ndarray,
    state_vec: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Numpy in -> numpy logits (N,); no grad, eval forward only.
    """
    if device is None:
        device = torch.device("cpu")
    net.eval()
    with torch.no_grad():
        logits = net(
            torch.from_numpy(cut_features).float().to(device),
            torch.from_numpy(state_vec).float().to(device),
        )
        return logits.cpu().numpy()
