import os
from speechbrain.pretrained import EncoderClassifier

def load_model(source: str, savedir: str = 'tmp') -> EncoderClassifier:
    if os.path.exists(source):
        pass
    else:
        return EncoderClassifier.from_hparams(source, savedir=savedir)

def infer(model: EncoderClassifier, audio_path: str) -> str:
    signal = model.load_audio(audio_path)
    scores, best_llh, best_pos, best_langs =  model.classify_batch(signal)
    
    return best_langs[0]

if __name__ == '__main__':

    SOURCE_MODEL = 'speechbrain/lang-id-voxlingua107-ecapa'
    SIGNAL_PATH = 'https://omniglot.com/soundfiles/udhr/udhr_th.mp3'

    model = load_model(SOURCE_MODEL)
    lang_prediction = infer(model, SIGNAL_PATH)

    print(lang_prediction)