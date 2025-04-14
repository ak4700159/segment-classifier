
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)  # A~E 총 5개 클래스
        )

    def forward(self, x):
        return self.layers(x)



