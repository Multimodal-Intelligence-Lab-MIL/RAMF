import os
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

print("CUDA available:", torch.cuda.is_available())

import random

random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Your data directory
DATA_PATH = os.path.join(BASE_DIR, "Data_HateMM")


# Load feature data

# Load Text Features
with open(os.path.join(DATA_PATH, 'text_feature_embedding_path'), 'rb') as fp:
    textData = pickle.load(fp)


# Load audio features
with open(os.path.join(DATA_PATH, 'audio_feature_embedding_path'), 'rb') as fp:
    audData = pickle.load(fp)

# Load video features
with open(os.path.join(DATA_PATH, 'video_feature_embedding_path'), 'rb') as fp:
    vidData = pickle.load(fp)


# Determine the feature dimensions for each modality
# Set the feature dimensions for each modality according to the new requirements
TEXT_SEQ_LEN = 100
TEXT_FEATURE_DIM = 768

AUDIO_SEQ_LEN = 100
AUDIO_FEATURE_DIM = 40

VIDEO_SEQ_LEN = 100
VIDEO_FEATURE_DIM = 768

print(f"Text shape: ({TEXT_SEQ_LEN}, {TEXT_FEATURE_DIM}), "
      f"Audio shape: ({AUDIO_SEQ_LEN}, {AUDIO_FEATURE_DIM}), "
      f"Video shape: ({VIDEO_SEQ_LEN}, {VIDEO_FEATURE_DIM})")



class MultiModalDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_id = self.data_list[idx]

        text_feat = textData.get(video_id, np.zeros((TEXT_SEQ_LEN, TEXT_FEATURE_DIM)))
        audio_feat = audData.get(video_id, np.zeros((AUDIO_SEQ_LEN, AUDIO_FEATURE_DIM)))
        video_feat = vidData.get(video_id, np.zeros((VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM)))

        return (
            torch.tensor(text_feat, dtype=torch.float32),
            torch.tensor(audio_feat, dtype=torch.float32),
            torch.tensor(video_feat, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class TextMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

    def forward(self, x):
        return self.projection(x)


class AudioMLP(nn.Module):

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


class VideoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        return self.projection(x)


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




class DynamicTemporalFusion(nn.Module):
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



class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()


        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)

        self.seq_pool = DynamicTemporalFusion(hidden_dim)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

        self.fusion2 = SCA(embed_dim=256, num_heads=4, dropout=0.1)

    def forward(self, text_input, audio_input, video_input):

        text_projected = self.text_mlp(text_input)
        audio_projected = self.audio_mlp(audio_input)
        video_projected = self.video_mlp(video_input)

        text_pooled = self.seq_pool(text_projected)
        audio_pooled = self.seq_pool(audio_projected)
        video_pooled = self.seq_pool(video_projected)

        fused = torch.stack([text_pooled, audio_pooled, video_pooled], dim=1)
        fused = self.fusion2(fused)

        fused_pooled = fused.mean(dim=1)  # [B, 256]

        logits = self.classifier(fused_pooled)

        return logits

# --------------------------
# Training and Evaluation Functions
# --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    for text, audio, video, labels in dataloader:
        text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(text, audio, video)
        '''
        You should choose different setting for different dataset. For hatemm set (0.41, 0.59) is follow the original paper.
        '''
        #loss = criterion(outputs, labels)# For multihateclip
        #loss = F.cross_entropy(outputs, labels, weight=torch.FloatTensor([0.41, 0.59]).to(device))#For hatemm
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_val_loss(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for text, audio, video, labels in dataloader:
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, audio, video, labels in dataloader:
            text, audio, video = text.to(device), audio.to(device), video.to(device)
            outputs = model(text, audio, video)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    return acc, macro_f1, weighted_f1


def evaluate_test(model, dataloader, device):

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for text, audio, video, labels in dataloader:
            text, audio, video = text.to(device), audio.to(device), video.to(device)
            outputs = model(text, audio, video)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    f1_h = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    auc = roc_auc_score(all_labels, all_probs)
    precision_h = precision_score(all_labels, all_preds, pos_label=1)
    recall_h = recall_score(all_labels, all_preds, pos_label=1)

    return acc, macro_f1, f1_h, auc, precision_h, recall_h



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_epochs = 60
    batch_size = 64
    learning_rate = 1e-4


##################################################################################
#Folded Dataset Partitioning File
###################################################################################
    five_fold_pickle = os.path.join(DATA_PATH, "five_folds_HateMM.pickle")
    with open(five_fold_pickle, "rb") as f:
        fold_data = pickle.load(f)


    test_accs, test_macro_f1s = [], []
    test_f1_hs, test_aucs = [], []
    test_precision_hs, test_recall_hs = [], []

    print("Start：")
    for fold_id in range(1, 6):
        print(f"\n======== Fold {fold_id} ========")


        train_list, train_label = fold_data[f'Fold_{fold_id}']['train']
        val_list, val_label = fold_data[f'Fold_{fold_id}']['val']
        test_list, test_label = fold_data[f'Fold_{fold_id}']['test']

        train_dataset = MultiModalDataset(train_list, train_label)
        val_dataset = MultiModalDataset(val_list, val_label)
        test_dataset = MultiModalDataset(test_list, test_label)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = MultiModalClassifier(
            text_dim=TEXT_FEATURE_DIM,
            audio_dim=AUDIO_FEATURE_DIM,
            video_dim=VIDEO_FEATURE_DIM
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # To record the optimal Macro F1 score for the validation set within the current fold, along with the corresponding model parameters.

        best_val_macro_f1 = float('-inf')
        best_model_weights = None

        for epoch in range(num_epochs):

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            val_loss = evaluate_val_loss(model, val_loader, criterion, device)
            val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, val_loader, device)
            test_acc, test_macro_f1, test_weighted_f1,_,_,_ = evaluate_test(model, test_loader, device)

            print(f"Epoch {epoch + 1:02d}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Accuracy={val_acc:.4f}, Macro F1={val_macro_f1:.4f}, Weighted F1={val_weighted_f1:.4f}"
                  #f"Val Accuracy={test_acc:.4f}, Macro F1={test_macro_f1:.4f}"
            )

            # Determine whether to update the optimal model based on the MacorF1 score of the validation set.
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                print("Best model updated based on val_acc!")
                best_model_weights = copy.deepcopy(model.state_dict())

        # ========== Training complete. Save the optimal model for this fold. ==========
        best_model_path = os.path.join(BASE_DIR, f"sep_transformer_best_model_fold_{fold_id}.pth")
        torch.save(best_model_weights, best_model_path)
        print(f"[*] Best model for fold {fold_id} saved to {best_model_path}")


        model.load_state_dict(best_model_weights)


        test_acc, test_macro_f1, test_f1_h, test_auc, test_ph, test_rh = evaluate_test(model, test_loader,
                                                                                       device)
        print(f"==> [Fold {fold_id} Test] Acc={test_acc:.4f}, MacroF1={test_macro_f1:.4f}, "
              f"F1(H)={test_f1_h:.4f}, AUC={test_auc:.4f}, Prec(H)={test_ph:.4f}, Recall(H)={test_rh:.4f}")

        test_accs.append(test_acc)
        test_macro_f1s.append(test_macro_f1)
        test_f1_hs.append(test_f1_h)
        test_aucs.append(test_auc)
        test_precision_hs.append(test_ph)
        test_recall_hs.append(test_rh)


    avg_acc = np.mean(test_accs)
    std_acc = np.std(test_accs, ddof=1)
    avg_macro_f1 = np.mean(test_macro_f1s)
    std_macro_f1 = np.std(test_macro_f1s, ddof=1)
    avg_f1_h = np.mean(test_f1_hs)
    std_f1_h = np.std(test_f1_hs, ddof=1)
    avg_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs, ddof=1)
    avg_precision_h = np.mean(test_precision_hs)
    std_precision_h = np.std(test_precision_hs, ddof=1)
    avg_recall_h = np.mean(test_recall_hs)
    std_recall_h = np.std(test_recall_hs, ddof=1)

    print("\n================ Average test set performance (Cross-Validation) ================")
    print(f"Accuracy       : {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro F1       : {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"F1 (H)         : {avg_f1_h:.4f} ± {std_f1_h:.4f}")
    print(f"AUC            : {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Precision (H)  : {avg_precision_h:.4f} ± {std_precision_h:.4f}")
    print(f"Recall (H)     : {avg_recall_h:.4f} ± {std_recall_h:.4f}")


if __name__ == "__main__":
    main()