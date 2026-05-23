import json
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
'''
# Data Path
FOLDER_NAME = 'C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_HateMM/Feature_embedding/LLM_description'

# Path to the input text pickle file
TEXT_FILES = {
    'objective': 'turn1_objective_description_texts.p',
    'hate': 'turn2_hate_assumption_texts.p',
    'nonhate': 'turn3_nonhate_assumption_texts.p'
}

# Load the BERT model and tokeniser
MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


model.eval()

class BERT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

bert_model = BERT_Model()
bert_model.eval()


def tokenize_text(text, max_len=512):

    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def extract_text_features(text):

    if not text or not text.strip():
        print("Empty text provided for feature extraction")
        return None

    try:
        
        input_ids, attention_mask = tokenize_text(text)

        
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask)

            
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]

            
            feature = cls_embedding.squeeze(0).detach().numpy()  # [hidden_dim]

        return feature
    except Exception as e:
        print(f"Error extracting features from text: {e}")
        return None


def read_pickle_file(file_path):
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pickle file {file_path}: {e}")
        return None


def extract_features_from_texts(texts_dict, text_type):
    
    features = {}

    print(f"Extracting features for {text_type} texts...")
    print(f"Found {len(texts_dict)} texts to process")

    for video_id, text in tqdm(texts_dict.items(), desc=f"Processing {text_type}"):
        if text and text.strip():
            feature = extract_text_features(text)
            if feature is not None:
                features[video_id] = feature
            else:
                print(f"Failed to extract features for {video_id}")
        else:
            print(f"Empty text for {video_id}, skipping")

    print(f"Successfully extracted features for {len(features)} out of {len(texts_dict)} texts")
    return features


def main():
   
    print("Starting BERT feature extraction from cleaned text files...")

   
    text_data = {}
    for text_type, filename in TEXT_FILES.items():
        file_path = os.path.join(FOLDER_NAME, filename)
        print(f"\nReading {filename}...")

        data = read_pickle_file(file_path)
        if data is not None:
            text_data[text_type] = data
            print(f"Loaded {len(data)} {text_type} texts")

            
            if data:
                sample_id = list(data.keys())[0]
                sample_text = data[sample_id]
                print(f"Sample text preview: {sample_text[:100]}...")
        else:
            print(f"Failed to load {filename}")
            text_data[text_type] = {}

    
    all_features = {}

    for text_type, texts_dict in text_data.items():
        if texts_dict:
            print(f"\n{'=' * 50}")
            features = extract_features_from_texts(texts_dict, text_type)
            all_features[text_type] = features
        else:
            print(f"No texts found for {text_type}")
            all_features[text_type] = {}

    
    print(f"\n{'=' * 50}")
    print("Saving BERT features...")

    
    os.makedirs(FOLDER_NAME, exist_ok=True)

    
    feature_files = [
        (all_features['objective'], 'objective_description_bert_features.p', 'objective description'),
        (all_features['hate'], 'hate_assumption_bert_features.p', 'hate assumption'),
        (all_features['nonhate'], 'nonhate_assumption_bert_features.p', 'non-hate assumption')
    ]

    for features, filename, description in feature_files:
        if features:
            file_path = os.path.join(FOLDER_NAME, filename)
            with open(file_path, 'wb') as fp:
                pickle.dump(features, fp)
            print(f"✓ Saved {len(features)} {description} features to {filename}")

            
            if features:
                sample_id = list(features.keys())[0]
                sample_feature = features[sample_id]
                print(f"  Feature shape: {sample_feature.shape}")
                print(f"  Feature type: {type(sample_feature)}")
        else:
            print(f"✗ No {description} features to save")

    print(f"\nBERT feature extraction completed!")
    print(f"Files saved in: {FOLDER_NAME}")


def extract_features_for_specific_type(text_type):
    
    if text_type not in TEXT_FILES:
        print(f"Unknown text type: {text_type}")
        return

    filename = TEXT_FILES[text_type]
    file_path = os.path.join(FOLDER_NAME, filename)

    print(f"Processing {text_type} texts from {filename}...")

    
    texts_dict = read_pickle_file(file_path)
    if texts_dict is None:
        print(f"Failed to load {filename}")
        return

    
    features = extract_features_from_texts(texts_dict, text_type)

    
    output_filename = f"{text_type}_bert_features.p"
    output_path = os.path.join(FOLDER_NAME, output_filename)

    with open(output_path, 'wb') as fp:
        pickle.dump(features, fp)

    print(f"Saved {len(features)} features to {output_filename}")


if __name__ == "__main__":
    
    main()

    # If you only wish to handle specific types, you can use:
    # extract_features_for_specific_type('objective')  
    # extract_features_for_specific_type('hate')       
    # extract_features_for_specific_type('nonhate')    

'''

import json
import re
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import os

# Data Path
FOLDER_NAME = 'C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_HateMM/Feature_embedding/LLM_description'

# Path to the input text pickle file
TEXT_FILES = {
    'objective': 'turn1_objective_description_texts.p',
    'hate': 'turn2_hate_assumption_texts.p',
    'nonhate': 'turn3_nonhate_assumption_texts.p'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


MODEL_NAME = "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
hatexplain_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
hatexplain_model = BertModel.from_pretrained(MODEL_NAME).to(device)


hatexplain_model.eval()


class HatEXplain_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs


hatexplain_bert_model = HatEXplain_Model()
hatexplain_bert_model.eval()


def tokenize_text(text, max_len=512):

    encoded_dict = hatexplain_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device)


def extract_text_features(text):

    if not text or not text.strip():
        print("Empty text provided for feature extraction")
        return None

    try:

        input_ids, attention_mask = tokenize_text(text)


        with torch.no_grad():
            outputs = hatexplain_bert_model(input_ids, attention_mask)


            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]


            feature = cls_embedding.squeeze(0).detach().cpu().numpy()  # [hidden_dim]

        return feature
    except Exception as e:
        print(f"Error extracting features from text: {e}")
        return None


def read_pickle_file(file_path):

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pickle file {file_path}: {e}")
        return None


def extract_features_from_texts(texts_dict, text_type):

    features = {}

    print(f"Extracting HatEXplain features for {text_type} texts...")
    print(f"Found {len(texts_dict)} texts to process")

    for video_id, text in tqdm(texts_dict.items(), desc=f"Processing {text_type}"):
        if text and text.strip():
            feature = extract_text_features(text)
            if feature is not None:
                features[video_id] = feature
            else:
                print(f"Failed to extract features for {video_id}")
        else:
            print(f"Empty text for {video_id}, skipping")

    print(f"Successfully extracted features for {len(features)} out of {len(texts_dict)} texts")
    return features


def main():

    print("Starting HatEXplain BERT feature extraction from cleaned text files...")

    text_data = {}
    for text_type, filename in TEXT_FILES.items():
        file_path = os.path.join(FOLDER_NAME, filename)
        print(f"\nReading {filename}...")

        data = read_pickle_file(file_path)
        if data is not None:
            text_data[text_type] = data
            print(f"Loaded {len(data)} {text_type} texts")


            if data:
                sample_id = list(data.keys())[0]
                sample_text = data[sample_id]
                print(f"Sample text preview: {sample_text[:100]}...")
        else:
            print(f"Failed to load {filename}")
            text_data[text_type] = {}


    all_features = {}

    for text_type, texts_dict in text_data.items():
        if texts_dict:
            print(f"\n{'=' * 50}")
            features = extract_features_from_texts(texts_dict, text_type)
            all_features[text_type] = features
        else:
            print(f"No texts found for {text_type}")
            all_features[text_type] = {}


    print(f"\n{'=' * 50}")
    print("Saving HatEXplain BERT features...")


    os.makedirs(FOLDER_NAME, exist_ok=True)


    feature_files = [
        (all_features['objective'], 'hatexplain_objective_description_bert_features.p', 'objective description'),
        (all_features['hate'], 'hatexplain_hate_assumption_bert_features.p', 'hate assumption'),
        (all_features['nonhate'], 'hatexplain_nonhate_assumption_bert_features.p', 'non-hate assumption')
    ]

    for features, filename, description in feature_files:
        if features:
            file_path = os.path.join(FOLDER_NAME, filename)
            with open(file_path, 'wb') as fp:
                pickle.dump(features, fp)
            print(f"✓ Saved {len(features)} {description} features to {filename}")


            if features:
                sample_id = list(features.keys())[0]
                sample_feature = features[sample_id]
                print(f"  Feature shape: {sample_feature.shape}")
                print(f"  Feature type: {type(sample_feature)}")
        else:
            print(f"✗ No {description} features to save")

    print(f"\nHatEXplain BERT feature extraction completed!")
    print(f"Files saved in: {FOLDER_NAME}")


def extract_features_for_specific_type(text_type):

    if text_type not in TEXT_FILES:
        print(f"Unknown text type: {text_type}")
        return

    filename = TEXT_FILES[text_type]
    file_path = os.path.join(FOLDER_NAME, filename)

    print(f"Processing {text_type} texts from {filename} with HatEXplain model...")


    texts_dict = read_pickle_file(file_path)
    if texts_dict is None:
        print(f"Failed to load {filename}")
        return


    features = extract_features_from_texts(texts_dict, text_type)


    output_filename = f"hatexplain_{text_type}_bert_features.p"
    output_path = os.path.join(FOLDER_NAME, output_filename)

    with open(output_path, 'wb') as fp:
        pickle.dump(features, fp)

    print(f"Saved {len(features)} HatEXplain features to {output_filename}")


if __name__ == "__main__":
    # Handling all types of text
    main()

    # If you only wish to handle specific types, you can use:
    # extract_features_for_specific_type('objective')
    # extract_features_for_specific_type('hate')
    # extract_features_for_specific_type('nonhate')