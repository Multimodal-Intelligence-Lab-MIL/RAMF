import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SCA(nn.Module):

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


        self.key_query_conv = nn.Conv2d(num_heads, num_heads, kernel_size=(3, 3), padding=(1, 1))
        self.head_conv = nn.Conv1d(num_heads//2, num_heads//2, kernel_size=2, stride=2, groups= num_heads//2)

        self.dropout = nn.Dropout(dropout)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape


        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)


        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]


        attn_weights = self.key_query_conv(attn_weights)


        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)


        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)


        if self.num_heads >= 2:

            even_heads = attn_weights[:, 0::2, :, :]
            odd_heads = attn_weights[:, 1::2, :, :]

            reshaped = torch.cat([even_heads, odd_heads], dim=2)
            mixed = self.head_conv(reshaped.view(batch_size, self.num_heads // 2, -1))
            mixed = mixed.view(batch_size, self.num_heads // 2, seq_len, seq_len)


            attn_weights = torch.zeros_like(attn_weights)
            attn_weights[:, 0::2, :, :] = mixed
            attn_weights[:, 1::2, :, :] = mixed


        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, dim]


        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)


        attn_output = self.group_norm(attn_output.permute(0, 2, 1)).permute(0, 2, 1)

        return attn_output
