# src/modules/DMTFusion_classifier.py
import torch
import torch.nn as nn
from .MLP_Projection import TextMLP, AudioMLP, VideoMLP, TurnMLP
from .Local_Gobal_Context_Fuison import LGCF
from .Semantic_Cross_Attention import SCA


class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()

        #The  three modal MLP projectors + turn1 MLP projector
        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)

        # MLP projectors for turn2 and turn3
        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)  # turn2
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)  # turn3


        self.seq_pool = LGCF(hidden_dim)

        # text + audio + video + turn1
        self.first_fusion = SCA(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # Merge the first layer output + turn2 + turn3
        self.second_fusion = SCA(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):

        text_projected = self.text_mlp(text_input)  # [B, seq_len, hidden_dim]
        audio_projected = self.audio_mlp(audio_input)  # [B, seq_len, hidden_dim]
        video_projected = self.video_mlp(video_input)  # [B, seq_len, hidden_dim]
        turn1_projected = self.turn1_mlp(turn1_input)  # [B, hidden_dim]


        text_pooled = self.seq_pool(text_projected)  # [B, hidden_dim]
        audio_pooled = self.seq_pool(audio_projected)  # [B, hidden_dim]
        video_pooled = self.seq_pool(video_projected)  # [B, hidden_dim]



        first_stage_tokens = torch.stack([text_pooled, audio_pooled, video_pooled, turn1_projected],dim=1)  # [B, 4, hidden_dim]
        first_fused = self.first_fusion(first_stage_tokens)  # [B, 4, hidden_dim]
        first_fused_pooled = first_fused.mean(dim=1)  # [B, hidden_dim]


        turn2_projected = self.turn2_mlp(turn2_input)  # [B, hidden_dim]
        turn3_projected = self.turn3_mlp(turn3_input)  # [B, hidden_dim]


        second_stage_tokens = torch.stack([first_fused_pooled, turn2_projected, turn3_projected],dim=1)  # [B, 3, hidden_dim]
        second_fused = self.second_fusion(second_stage_tokens)  # [B, 3, hidden_dim]
        final_fused = second_fused.mean(dim=1)  # [B, hidden_dim]


        logits = self.classifier(final_fused)

        return logits


