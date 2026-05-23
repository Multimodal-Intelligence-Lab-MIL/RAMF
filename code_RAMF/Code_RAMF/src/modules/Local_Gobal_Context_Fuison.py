import torch
import torch.nn as nn

class LGCF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape


        x_t = x.transpose(1, 2)  # (B, D, L)
        local_feat = self.local_conv(x_t)  # (B, D, L)


        global_feat = self.global_pool(x_t).squeeze(-1)  # (B, D)


        max_pool = torch.max(local_feat, dim=2)[0]  # (B, D)


        combined = torch.cat([max_pool, global_feat], dim=1)  # (B, 2D)
        gate_value = self.gate(combined)  # (B, D)


        output = gate_value * max_pool + (1 - gate_value) * global_feat

        return output