import torchaudio
from tools.helpers import get_device

def get_wavfile_for_eval(filepath, target_sr=16000, clip=5):
    wavfile, sr = torchaudio.load(filepath)
    if sr != target_sr:
        wavfile = torchaudio.transforms.Resample(sr,target_sr)(wavfile)

    # clip wavfile to 5s
    wavfile = wavfile[:,:target_sr*clip]
    wavfile = wavfile.to(get_device())
    return wavfile


def pad_num_to_len(num, length=10):
    num_str = f"{num:.{length-2}f}"
    if len(num_str) > length:
        if '.' in num_str:
            num_str = num_str[:length]
    return num_str