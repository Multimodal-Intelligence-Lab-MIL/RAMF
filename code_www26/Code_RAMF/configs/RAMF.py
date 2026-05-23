# configs/DMTFusion.py
import os
import random
import numpy as np
import torch


RANDOM_SEED = 2021


DATA_PATH = "C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_HateMM"

# Model save path
SAVE_DIR = "C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_HateMM/scripts/models/DMTFusion"


TEXT_FEATURE_PATH = 'Feature_embedding/DMTFusion_style/hatexplain_whispermedium_seq_100_feature_embedding.p'
AUDIO_FEATURE_PATH = 'Feature_embedding/DMTFusion_style/all_audio_mfcc_features_float32_CMFusion.p'
VIDEO_FEATURE_PATH = 'Feature_embedding/DMTFusion_style/all_vit_features_CMFusion.p'

TURN1_FEATURE_PATH = 'Feature_embedding/LLM_description/llama/objective_description_bert_llama_features.p'  # turn1 Objective description
TURN2_FEATURE_PATH = 'Feature_embedding/LLM_description/llama/hate_assumption_bert_llama_features.p'        # turn2 Hate Hypothesis
TURN3_FEATURE_PATH = 'Feature_embedding/LLM_description/llama/nonhate_assumption_bert_llama_features.p'     # turn3 Non-Hate Hypothesis

TEXT_SEQ_LEN = 100
TEXT_FEATURE_DIM = 768

AUDIO_SEQ_LEN = 100
AUDIO_FEATURE_DIM = 40

VIDEO_SEQ_LEN = 100
VIDEO_FEATURE_DIM = 768

TURN_FEATURE_DIM = 768


NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

