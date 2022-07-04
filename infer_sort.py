import os
import librosa
from typing import List, Union
from inference import load_model, infer


####### INPUT VARIABLES #######

SOURCE_MODEL = 'speechbrain/lang-id-voxlingua107-ecapa'
SOURCE_MAIN_DIR = '/workspace/data'
TARGET_MAIN_DIR = '/workspace/cleaned/'

###############################

def get_audiopaths(dirname: str) -> List[str]:
    # dirname: directory name to start looking for audio paths

    audiopaths = []

    for root, _, filenames in os.walk(dirname):
        for fn in filenames:
            fp = os.path.join(root, fn)
            if not fp.endswith('.wav'):
                continue
            audiopaths.append(fp)

    return audiopaths

def calculate_duration(audiopaths: List[str], output_dir: str, languages: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    duration_dict = {'en': 0, 'others': 0}

    with open(os.path.join(output_dir, 'prediction_labels.csv'), 'w', encoding='utf-8') as fw:
        fw.write('filename,label\n')
    
        for filepath, label in zip(audiopaths, languages):
            if label == 'en: English':
                duration_dict['en'] += librosa.get_duration(filename=filepath)
            else:
                duration_dict['others'] += librosa.get_duration(filename=filepath)
            fw.write(f'{filepath},{label}\n')
            
    for key in duration_dict.keys():
        duration = duration_dict[key] / 3600
        print(f'Total for {key} (hrs): {round(duration, 3)}')

def sort_audiopaths(audiopaths: List[str], dirname: str, languages: List[str]) -> None:
    # audiopaths input -> ['/workspace/data/xx/yy.wav', ...]
    # languages input -> ['id: Indonesian', 'id: Indonesian, 'ms: Malay', ...]
    pass

if __name__ == '__main__':

    model = load_model(SOURCE_MODEL)
    audiopaths = get_audiopaths(SOURCE_MAIN_DIR)
    lang_predictions = infer(model, audiopaths)
    # sort_audiopaths(audiopaths, TARGET_MAIN_DIR, lang_predictions)
    calculate_duration(audiopaths, output_dir='./outputs', languages=lang_predictions)