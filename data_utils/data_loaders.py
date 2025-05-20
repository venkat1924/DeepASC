from itertools import cycle
import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from torch.utils.data import random_split
import random

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
META_PATH = os.path.join(BASE_DIR, "data/Audioset/Meta")

reverbation_times = [0.15, 0.175, 0.2, 0.225, 0.25]

class WavClip(object):
    def __init__(self, filepath, clip_length, samples, random):
        self.filepath = filepath
        self.filename = os.path.basename(filepath).split('.')[0]
        self.clip_length = clip_length
        self.samples = samples
        self.num_clips = int(samples // clip_length)
        self.random = random
        self.init()

    def init(self):
        self.num_seen = 0

        if self.random:
            self.start = np.random.randint(0, self.samples - self.clip_length + 1)
        else:
            self.start = 0

        self.end = (self.start + self.clip_length) % self.samples


    def move_next(self):
        self.num_seen += 1
        if self.num_seen == self.num_clips:
            self.init()
            return
        self.start = (self.start + self.clip_length) % self.samples
        self.end = (self.end + self.clip_length) % self.samples

    def to_dict(self):
        return {
            "filepath": self.filepath,
            "clip_length": self.clip_length,
            "samples": self.samples,
        }


def create_metadata_json(wav_files, key, max_size=None, target_sr=16000, seconds=3, random_cutoff=False):
    print(f"CREATING METADATA JSON WITH: {key} max_size={max_size} seconds={seconds}")
    last_report = 0
    zeros = 0
    clips_data = dict()
    clip_length = seconds * target_sr
    clips_num = 0

    for wav_file in wav_files:
        # check if file exists
        if not os.path.exists(wav_file):
            continue
        
        wavfile, sr = torchaudio.load(wav_file)

        if sr != target_sr:
            print(f"{wav_file} has sr {sr} and not {target_sr}", flush=True)
            wavfile = torchaudio.transforms.Resample(sr, target_sr)(wavfile)
            torchaudio.save(wav_file, wavfile, target_sr)
        
        if torch.all(wavfile == 0):
            os.remove(wav_file)
            zeros += 1
            continue

        num_samples = wavfile.shape[1]
        num_clips = int(num_samples // clip_length)
        if num_clips == 0:
            continue
        wav_data = WavClip(wav_file, clip_length, num_samples, random=random_cutoff)
        for i in range(num_clips):
            clip_idx = clips_num + i
            clips_data[clip_idx] = wav_data

        clips_num += num_clips
        if clips_num - last_report > 1000:
            print(f"Processed {clips_num} clips. Zeros: {zeros}", flush=True)
            last_report = clips_num

        if max_size is not None and clips_num > max_size:
            break

    print(f"Storing {key}.json with {clips_num} clips", flush=True)     
    with open(f"{META_PATH}/{key}.json", "w") as f:
        json_clips_data = {k: v.to_dict() for k, v in clips_data.items()}
        json.dump(json_clips_data, f, indent=4, sort_keys=True)
    return clips_data, clips_num


class WavDataset(Dataset):
    def __init__(self, key, wav_files,max_size=None, target_sr=16000, seconds=5, random_cutoff=False, filenames=False, v=1):
        self.target_sr = target_sr
        self.seconds = seconds
        self.clip_length = target_sr * seconds
        self.clips_num = 0
        self.random_cutoff = random_cutoff
        self.clips_data = self.get_clips_data(key, wav_files=wav_files, max_size=max_size)
        self.filenames = filenames
        self.v = v
        self.base_path = os.path.dirname(self.clips_data[0].filepath)


    def get_clips_data(self, key ,wav_files, max_size):
        # load json file if exists
        if os.path.exists(f"{META_PATH}/{key}.json"):
            clips_data = dict()
            with open(f"{META_PATH}/{key}.json", "r") as f:
                rare_data = json.load(f)
            print(f"Loaded {key}.json with {len(rare_data)} clips", flush=True)

            for k, v in rare_data.items():
                key = int(k)
                num_clips = int( v["samples"] // v["clip_length"])
                found = False
                for i in range(num_clips, 0, -1):
                    if key - i in clips_data and clips_data[key - i].filepath == v["filepath"]:
                        clips_data[key] = wav_data
                        found = True
                        break
                if not found:
                    wav_data = WavClip(v["filepath"], v["clip_length"], v["samples"], random=self.random_cutoff)
                    clips_data[key] = wav_data
            self.clips_num = len(clips_data)
            return clips_data
        else:
            clips_data, clips_num = create_metadata_json(wav_files, key, max_size=max_size, target_sr=self.target_sr, seconds=self.seconds, random_cutoff=self.random_cutoff)
            self.clips_num = clips_num
            return clips_data
        
    def __len__(self):
        return self.clips_num

    def __getitem__(self, idx):
        # find clip index
        wav_data = self.clips_data[idx]

        # Resample if necess

        if wav_data.end < wav_data.start:
            wavfile_start, sr = torchaudio.load(wav_data.filepath, frame_offset=wav_data.start)
            wavfile_end, sr = torchaudio.load(wav_data.filepath,frame_offset=0, num_frames=wav_data.end)
            wavfile = torch.cat((wavfile_start,wavfile_end), dim=1)
        else:
            wavfile, sr = torchaudio.load(wav_data.filepath, frame_offset=wav_data.start, num_frames=wav_data.clip_length)
        
        wavfile = wavfile.squeeze()

        if wavfile.shape != (self.clip_length,):
            print(f"Error: {wav_data.filepath} shape is {wavfile.shape} with sr {sr} start: {wav_data.start} end: {wav_data.end} total_samples: {wav_data.samples} clips_num: {wav_data.num_clips}")

        wav_data.move_next()
        # print(wav_data.filepath, wav_data.num_seen, flush=True)
        if self.filenames:
            if self.v != 3 and self.v != 4:
                return wavfile, wav_data.filepath, wav_data.num_seen
            if self.v == 3:
                targets = {
                    t60: torchaudio.load(f"{self.base_path}/optimal_3s_v3/{wav_data.filename}_{wav_data.num_seen}_{str(t60).replace('.','-')}.wav")[0].squeeze() for t60 in reverbation_times
                }
            elif self.v == 4:
                targets = torchaudio.load(f"{self.base_path}/optimal_3s_avg/{wav_data.filename}_{wav_data.num_seen}_avg.wav")[0].squeeze() 
            return wavfile, wav_data.filepath, wav_data.num_seen, targets

        elif self.v == 1:
            return wavfile
        elif self.v == 2:
            t60 = random.choice(reverbation_times)
            target = torchaudio.load(f"{self.base_path}/optimal_3s_v3/{wav_data.filename}_{wav_data.num_seen}_{str(t60).replace('.','-')}.wav")[0].squeeze()
            return wavfile, target, t60
        elif self.v == 3:
            targets = {
                t60: torchaudio.load(f"{self.base_path}/optimal_3s_v3/{wav_data.filename}_{wav_data.num_seen}_{str(t60).replace('.','-')}.wav")[0].squeeze() for t60 in reverbation_times
            }
        elif self.v == 4:
            targets = torchaudio.load(f"{self.base_path}/optimal_3s_avg/{wav_data.filename}_{wav_data.num_seen}_avg.wav")[0].squeeze()
        # print("wav shape", wavfile.shape,"targets shape",targets.shape, flush=True)
        return wavfile, targets

class EvalDataSet(WavDataset):
    def __init__(self, key, wav_files, noise_types, target_sr=16000, seconds=5,random_cutoff=False,filenames=False,v=1):
        super(EvalDataSet, self).__init__(f"{key}_eval", wav_files, max_size=None, target_sr=target_sr, seconds=seconds, random_cutoff=random_cutoff,filenames=filenames, v=v)
        self.clips_data = {i: v for i, v in enumerate(self.clips_data.values()) if self.apply_filter(v.filepath, noise_types)}
        self.clips_data = {i: v for i, v in enumerate(self.clips_data.values())}
        self.filenames = filenames
        self.v = v

    def __len__(self):
        return len(self.clips_data)

    def extract_noise_type(self, filepath):
        noise_type = filepath.split('/')[-1].split('.')[0]
        return noise_type

    def apply_filter(self, filename, noise_types):
        noise_type = self.extract_noise_type(filename)
        return noise_type in noise_types
    
    def __getitem__(self, idx):
        if self.filenames:
            wavfile, filepath, num_seen = super(EvalDataSet, self).__getitem__(idx)
            return wavfile, filepath, num_seen, self.extract_noise_type(filepath)
        else:
            filepath = self.clips_data[idx].filepath
            if self.v == 1:
                wavfile = super(EvalDataSet, self).__getitem__(idx)
                return wavfile, self.extract_noise_type(filepath)
            else:
                wavfile, targets = super(EvalDataSet, self).__getitem__(idx)
                return wavfile, targets, self.extract_noise_type(filepath)

def get_dataloaders(key, data_files, batch_size, max_size=None, train_ratio=0.9,seconds=5, random_cutoff=True, filenames=False, v=1, shuffle=False):
    if v == 1:
        dataset = WavDataset(key,wav_files=data_files, max_size=max_size, seconds=seconds, random_cutoff=random_cutoff, filenames=filenames, v=v)
    else:
        dataset = WavDataset(key,wav_files=data_files, max_size=max_size, seconds=seconds, random_cutoff=False, filenames=filenames, v=v)
        

    if train_ratio == 1:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader
    
    # Determine the sizes of the train/test splits
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size


    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def get_eval_dataloader(key,data_files, batch_size, noise_types, seconds=5,random_cutoff=False,filenames=False, v=1):
    eval_dataset = EvalDataSet(key, wav_files=data_files, noise_types=noise_types, seconds=seconds,random_cutoff=random_cutoff,filenames=filenames,v=v)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return eval_dataloader


def get_speech_test_dataloader(key,data_files, batch_size, seconds=1):
    speech_test_loader = get_dataloaders(key=f"{key}_timit_test",data_files=data_files, batch_size=batch_size,max_size=66667,train_ratio=1, seconds=seconds,v=1, shuffle=False, random_cutoff=False)
    return speech_test_loader