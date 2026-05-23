import json
import re
import pickle
import os
from tqdm import tqdm

# Data Path
FOLDER_NAME = 'C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_MHC_E/'
JSON_FILE_PATH = 'C:/Users/31589/PycharmProjects/Hateful_Video_Localisation/Data_MHC_E/multi_turn_hate_MHC_E_detection_results.json'




def clean_raw_response(raw_response):
   """Remove JSON structure characters from raw_response, retaining standard punctuation marks."""
   if not raw_response:
       return ""

   try:

       text = re.sub(r'```json\s*\n?', '', raw_response)
       text = re.sub(r'\n?```', '', text)

       text = re.sub(r'"strength":\s*[\d.]+,?\s*', ' ', text)


       text = text.replace('\\n', ' ')
       text = text.replace('\\"', '')
       text = text.replace('\\\\', '\\')
       text = text.replace('\\t', ' ')
       text = text.replace('\\r', ' ')
       text = text.replace('\\/', '/')


       text = text.replace('{', ' ')
       text = text.replace('}', ' ')
       text = text.replace('[', ' ')
       text = text.replace(']', ' ')


       text = re.sub(r':\s*"', ': ', text)
       text = re.sub(r'"[a-zA-Z_][a-zA-Z0-9_]*":', ' ', text)


       text = re.sub(r'^\s*"', '', text)
       text = re.sub(r'"\s*$', '', text)
       text = re.sub(r'"\s*,\s*"', ' ', text)


       text = re.sub(r',\s*$', '', text)
       text = re.sub(r'^\s*,', '', text)


       text = re.sub(r'\\(?![a-zA-Z])', ' ', text)


       text = text.replace('"', '')


       text = re.sub(r'\s+', ' ', text)
       text = text.strip()

       return text

   except Exception as e:
       print(f"Error cleaning raw_response: {e}")
       return raw_response


def read_json_file(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


def extract_text_data_from_single_video(video_data):

    turn1_text = ""
    turn2_text = ""
    turn3_text = ""


    if not video_data or not isinstance(video_data, dict):
        print(f"Skipping invalid video_data: {type(video_data)}")
        return turn1_text, turn2_text, turn3_text

    video_id = video_data.get('video_id', 'unknown')
    multi_turn_results = video_data.get('multi_turn_results')


    if not multi_turn_results or not isinstance(multi_turn_results, dict):
        print(f"Skipping video {video_id}: no valid multi_turn_results")
        return turn1_text, turn2_text, turn3_text

    print(f"Processing video: {video_id}")


    if 'turn_1' in multi_turn_results:
        turn_1 = multi_turn_results['turn_1']
        if turn_1 and isinstance(turn_1, dict):
            raw_response = turn_1.get('raw_response', '')
            if raw_response:
                turn1_text = clean_raw_response(raw_response)
                #print(f"  - Turn 1 text length: {len(turn1_text)}")
                #print(f"    Preview: {turn1_text[:100]}...")
            else:
                print(f"  - Turn 1: No raw_response")
        else:
            print(f"  - Turn 1: Invalid data")


    if 'turn_2' in multi_turn_results:
        turn_2 = multi_turn_results['turn_2']
        if turn_2 and isinstance(turn_2, dict):
            raw_response = turn_2.get('raw_response', '')
            if raw_response:
                turn2_text = clean_raw_response(raw_response)
                #print(f"  - Turn 2 text length: {len(turn2_text)}")
                #print(f"    Preview: {turn2_text[:100]}...")
            else:
                print(f"  - Turn 2: No raw_response")
        else:
            print(f"  - Turn 2: Invalid data")


    if 'turn_3' in multi_turn_results:
        turn_3 = multi_turn_results['turn_3']
        if turn_3 and isinstance(turn_3, dict):
            raw_response = turn_3.get('raw_response', '')
            if raw_response:
                turn3_text = clean_raw_response(raw_response)
                #print(f"  - Turn 3 text length: {len(turn3_text)}")
                #print(f"    Preview: {turn3_text[:100]}...")
            else:
                print(f"  - Turn 3: No raw_response")
        else:
            print(f"  - Turn 3: Invalid data")

    return turn1_text, turn2_text, turn3_text


def extract_all_text_data(data):



    turn1_texts = {}  # objective description
    turn2_texts = {}  # hate assumption
    turn3_texts = {}  # non-hate assumption


    if isinstance(data, list):
        print(f"Processing {len(data)} videos from list...")

        for video_data in tqdm(data, desc="Processing videos"):
            if isinstance(video_data, dict):
                video_id = video_data.get('video_id', 'unknown')
                turn1_text, turn2_text, turn3_text = extract_text_data_from_single_video(video_data)


                if turn1_text:
                    turn1_texts[video_id] = turn1_text
                if turn2_text:
                    turn2_texts[video_id] = turn2_text
                if turn3_text:
                    turn3_texts[video_id] = turn3_text

    elif isinstance(data, dict):
        print("Processing single video from dict...")

        video_id = data.get('video_id', 'unknown')
        turn1_text, turn2_text, turn3_text = extract_text_data_from_single_video(data)

        if turn1_text:
            turn1_texts[video_id] = turn1_text
        if turn2_text:
            turn2_texts[video_id] = turn2_text
        if turn3_text:
            turn3_texts[video_id] = turn3_text

    else:
        print(f"Unsupported data type: {type(data)}")
        return {}, {}, {}

    return turn1_texts, turn2_texts, turn3_texts


def main():

    print("Starting text data extraction...")


    data = read_json_file(JSON_FILE_PATH)
    if data is None:
        print("Failed to read JSON file. Exiting.")
        return


    turn1_texts, turn2_texts, turn3_texts = extract_all_text_data(data)


    os.makedirs(FOLDER_NAME, exist_ok=True)


    text_files = [
        (turn1_texts, 'turn1_objective_description_texts', 'objective description'),
        (turn2_texts, 'turn2_hate_assumption_texts', 'hate assumption'),
        (turn3_texts, 'turn3_nonhate_assumption_texts', 'non-hate assumption')
    ]

    for texts, filename_base, description in text_files:
        if texts:

            pickle_path = os.path.join(FOLDER_NAME, f"{filename_base}.p")
            with open(pickle_path, 'wb') as fp:
                pickle.dump(texts, fp)


            json_path = os.path.join(FOLDER_NAME, f"{filename_base}.json")
            with open(json_path, 'w', encoding='utf-8') as fp:
                json.dump(texts, fp, ensure_ascii=False, indent=2)

            print(f"\n✓ Saved {len(texts)} {description} texts:")
            print(f"  - Pickle: {filename_base}.p")
            print(f"  - JSON: {filename_base}.json")


            text_lengths = [len(text) for text in texts.values()]
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            print(f"  - Average text length: {avg_length:.1f} characters")
            print(
                f"  - Min/Max length: {min(text_lengths) if text_lengths else 0}/{max(text_lengths) if text_lengths else 0}")


            if texts:
                sample_id = list(texts.keys())[0]
                sample_text = texts[sample_id]
                print(f"  - Sample ({sample_id}): {sample_text[:150]}...")
        else:
            print(f"\n✗ No {description} texts found")

    print(f"\nText extraction completed!")
    print(f"Files saved in: {FOLDER_NAME}")


def process_multiple_videos(json_files_list):

    all_turn1_texts = {}
    all_turn2_texts = {}
    all_turn3_texts = {}

    for json_file in tqdm(json_files_list, desc="Processing JSON files"):
        print(f"\nProcessing file: {json_file}")
        data = read_json_file(json_file)
        if data is None:
            continue

        turn1_texts, turn2_texts, turn3_texts = extract_all_text_data(data)


        all_turn1_texts.update(turn1_texts)
        all_turn2_texts.update(turn2_texts)
        all_turn3_texts.update(turn3_texts)


    os.makedirs(FOLDER_NAME, exist_ok=True)

    text_files = [
        (all_turn1_texts, 'turn1_objective_description_texts', 'objective description'),
        (all_turn2_texts, 'turn2_hate_assumption_texts', 'hate assumption'),
        (all_turn3_texts, 'turn3_nonhate_assumption_texts', 'non-hate assumption')
    ]

    for texts, filename_base, description in text_files:
        if texts:

            pickle_path = os.path.join(FOLDER_NAME, f"{filename_base}.p")
            with open(pickle_path, 'wb') as fp:
                pickle.dump(texts, fp)


            json_path = os.path.join(FOLDER_NAME, f"{filename_base}.json")
            with open(json_path, 'w', encoding='utf-8') as fp:
                json.dump(texts, fp, ensure_ascii=False, indent=2)

            print(f"\n✓ Saved {len(texts)} total {description} texts:")
            print(f"  - Pickle: {filename_base}.p")
            print(f"  - JSON: {filename_base}.json")


if __name__ == "__main__":

    main()

    # If you have multiple JSON files, please use the following code:
    # json_files = ['file1.json', 'file2.json', 'file3.json']
    # process_multiple_videos(json_files)