
import torch.nn as nn

class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=5, dropout_rate=0.3, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(block)
            prev_dim = h_dim

        self.out_layer = nn.Linear(prev_dim, output_dim)
        self._initialize_weights()

    def forward(self, x):
        for block in self.layers:
            prev = x
            x = block(x)
            if self.use_residual and x.shape == prev.shape:
                x = x + prev  # 잔차 연결 (차원 일치 시)
        return self.out_layer(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)