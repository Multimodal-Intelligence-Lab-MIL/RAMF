# src/data/dataset_DMTFusion.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from configs.RAMF import (DATA_PATH, TEXT_SEQ_LEN, TEXT_FEATURE_DIM, AUDIO_SEQ_LEN, AUDIO_FEATURE_DIM,
                               VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM, TURN_FEATURE_DIM,
                               TEXT_FEATURE_PATH, AUDIO_FEATURE_PATH, VIDEO_FEATURE_PATH,
                               TURN1_FEATURE_PATH, TURN2_FEATURE_PATH, TURN3_FEATURE_PATH)

def load_features():

    with open(os.path.join(DATA_PATH, TEXT_FEATURE_PATH), 'rb') as fp:
        textData = pickle.load(fp)

    with open(os.path.join(DATA_PATH, AUDIO_FEATURE_PATH), 'rb') as fp:
        audData = pickle.load(fp)

    with open(os.path.join(DATA_PATH, VIDEO_FEATURE_PATH), 'rb') as fp:
        vidData = pickle.load(fp)


    with open(os.path.join(DATA_PATH, TURN1_FEATURE_PATH), 'rb') as fp:
        turn1_data = pickle.load(fp)

    with open(os.path.join(DATA_PATH, TURN2_FEATURE_PATH), 'rb') as fp:
        turn2_data = pickle.load(fp)

    with open(os.path.join(DATA_PATH, TURN3_FEATURE_PATH), 'rb') as fp:
        turn3_data = pickle.load(fp)

    print(f"Text shape: ({TEXT_SEQ_LEN}, {TEXT_FEATURE_DIM}), "
          f"Audio shape: ({AUDIO_SEQ_LEN}, {AUDIO_FEATURE_DIM}), "
          f"Video shape: ({VIDEO_SEQ_LEN}, {VIDEO_FEATURE_DIM}), "
          f"Turn1 shape: {TURN_FEATURE_DIM}, "
          f"Turn2 shape: {TURN_FEATURE_DIM}, "
          f"Turn3 shape: {TURN_FEATURE_DIM}")

    return textData, audData, vidData, turn1_data, turn2_data, turn3_data


class MultiModalDataset(Dataset):
    def __init__(self, data_list, labels, textData, audData, vidData, turn1_data, turn2_data, turn3_data):
        self.data_list = data_list
        self.labels = labels
        self.textData = textData
        self.audData = audData
        self.vidData = vidData
        self.turn1_data = turn1_data
        self.turn2_data = turn2_data
        self.turn3_data = turn3_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_id = self.data_list[idx]

        text_feat = self.textData.get(video_id, np.zeros((TEXT_SEQ_LEN, TEXT_FEATURE_DIM)))
        audio_feat = self.audData.get(video_id, np.zeros((AUDIO_SEQ_LEN, AUDIO_FEATURE_DIM)))
        video_feat = self.vidData.get(video_id, np.zeros((VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM)))

        turn1_feat = self.turn1_data.get(video_id, np.zeros(TURN_FEATURE_DIM))
        turn2_feat = self.turn2_data.get(video_id, np.zeros(TURN_FEATURE_DIM))
        turn3_feat = self.turn3_data.get(video_id, np.zeros(TURN_FEATURE_DIM))

        return (
            torch.tensor(text_feat, dtype=torch.float32),
            torch.tensor(audio_feat, dtype=torch.float32),
            torch.tensor(video_feat, dtype=torch.float32),
            torch.tensor(turn1_feat, dtype=torch.float32),
            torch.tensor(turn2_feat, dtype=torch.float32),
            torch.tensor(turn3_feat, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )