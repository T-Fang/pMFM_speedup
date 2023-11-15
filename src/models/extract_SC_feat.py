from typing import List

import torch
import torch.nn.functional as F
import torch.nn as nn
from src.utils.SC_utils import batched_SC_to_triu_vector

NUM_OF_SC_FEAT = 200


class ExtractScFeatCnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 5 * 5, NUM_OF_SC_FEAT)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x


class ExtractScFeatMLP(nn.Module):

    def __init__(self, output_dims: List[int]):
        super().__init__()
        self.output_dims = output_dims
        layers: List[nn.Module] = []

        input_dim = 2278  # upper triangular matrix without diagonal entries: 68 * 67 / 2
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        self.layers: nn.Module = nn.Sequential(*layers)

    @property
    def SC_feat_dim(self):
        return self.output_dims[-1]

    def forward(self, batched_SC: torch.Tensor) -> torch.Tensor:
        SC_vectors = batched_SC_to_triu_vector(batched_SC)
        return self.layers(SC_vectors)
