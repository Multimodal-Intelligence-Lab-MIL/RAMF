# src/modules/MLP_projection_layers.py
import torch.nn as nn



# 🆕 Turn特征的MLP投影层（用于turn1, turn2, turn3）
class TurnMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            #nn.Linear(512, 512),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        # x: [B, input_dim] -> [B, hidden_dim]
        return self.projection(x)


class TextMLP(nn.Module):
    """文本特征投影MLP"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, x):
        return self.projection(x)


class AudioMLP(nn.Module):
    """音频特征投影MLP"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):
        return self.projection(x)


class VideoMLP(nn.Module):
    """视频特征投影MLP"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, x):
        return self.projection(x)