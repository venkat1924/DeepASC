import torch
import numpy as np
from scipy.signal import resample
import os
import soundfile as sf
import numpy as np
import sys
import random
from compare.DeepANC.cloned.data_utils_online import get_eval_dataloader

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = F"{SCRIPT_DIR}/../my_data/models/"

def save_model(model, optimizer, scheduler, epoch, model_name):
    data = {
        'epoch': epoch,
        'model_state_dict': model.net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(data, f"{BASE_PATH}{model_name}.pt")
    return

def remove_model(model_name):
    if os.path.exists(f"{BASE_PATH}{model_name}.pt"):
        os.remove(f"{BASE_PATH}{model_name}.pt")
    return

def load_model(model, optimizer, scheduler, model_name):
    data = torch.load(f"{BASE_PATH}{model_name}.pt")

    model.net.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(data['scheduler_state_dict'])
    epoch = data['epoch']
    return model, optimizer, scheduler, epoch

def resample_audio(audio_batch, sr, target_sr):
    resampled_batch = []
    for audio in audio_batch:
        num_samples = int(len(audio) * target_sr / sr)
        resampled_audio = resample(audio, num_samples)
        resampled_batch.append(np.float32(resampled_audio))
    return torch.stack([torch.from_numpy(audio) for audio in resampled_batch])