"""Configurable feedforward classifier for synthetic research data."""

import torch.nn as nn


class HealthcareFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.25):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)