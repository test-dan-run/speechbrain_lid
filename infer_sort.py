import os
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

def sort_audiopaths(audiopaths: List[str], dirname: str, languages: List[str]) -> None:
    # audiopaths input -> ['/workspace/data/xx/yy.wav', ...]
    # languages input -> ['id: Indonesian', 'id: Indonesian, 'ms: Malay', ...]
    pass

if __name__ == '__main__':

    model = load_model(SOURCE_MODEL)
    audiopaths = get_audiopaths(SOURCE_MAIN_DIR)
    lang_predictions = infer(model, audiopaths)
    # sort_audiopaths(audiopaths, TARGET_MAIN_DIR, lang_predictions)