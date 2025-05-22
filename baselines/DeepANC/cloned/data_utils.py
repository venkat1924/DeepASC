import random
import torchaudio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample



class WavReader(object):
    def __init__(self, in_file, sr, mode):
        # if mode is 'train', in_file is a list of filenames;
        # if mode is 'eval', in_file is a filename
        self.mode = mode
        self.in_file = in_file
        self.sr = sr
        assert self.mode in {'train', 'eval'}
        self.wav_dict = {i: wavfile for i, wavfile in enumerate(in_file)}
        self.wav_indices = sorted(list(self.wav_dict.keys()))
        self.split_sec = 5

    def load(self, idx):
        filename = self.wav_dict[idx]
        # wavfile = torchaudio.load(filename)[0].squeeze()
        wavfile, sr = torchaudio.load(filename)
        # split wavfile to self.split_sec seconds
        wavfile = wavfile.squeeze()

        if sr != self.sr:
            num_samples = int(len(wavfile) * self.sr / sr)
            wavfile = torch.from_numpy(resample(wavfile, num_samples))
        return wavfile

    def __iter__(self):
        for idx in self.wav_indices:
            yield idx, self.load(idx)


class PerUttLoader(object):
    def __init__(self, in_file, sr, shuffle=True, mode='train'):
        self.shuffle = shuffle
        self.mode = mode
        self.wav_reader = WavReader(in_file,sr, mode)
        self.eps = torch.finfo(torch.float32).eps

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.wav_reader.wav_indices)

        for idx, utt in self.wav_reader:
            yield utt


class SegSplitter(object):
    def __init__(self, segment_size, sample_rate, hop_size):
        self.seg_len = int(sample_rate * segment_size)
        self.hop_len = int(sample_rate * hop_size) 
        
    def __call__(self, utt_eg):
        n_samples = utt_eg['n_samples']
        segs = []
        if n_samples < self.seg_len:
            pad_size = self.seg_len - n_samples
            seg = np.pad(utt_eg, [(0, pad_size)])
            segs.append(seg)
        else:
            s_point = 0
            while True:
                if s_point + self.seg_len > n_samples:
                    break
                seg = dict()
                seg = utt_eg[s_point:s_point+self.seg_len]
                s_point += self.hop_len
                segs.append(seg)
        return segs


class AudioLoader(object):
    def __init__(self, 
                 in_file, 
                 sample_rate,
                 unit='seg',
                 segment_size=4.0,
                 segment_shift=1.0, 
                 batch_size=4, 
                 buffer_size=16,
                 mode='train'):
        self.mode = mode
        assert self.mode in {'train', 'eval'}
        self.unit = unit
        assert self.unit in {'seg', 'utt'}
        if self.mode == 'train':
            self.utt_loader = PerUttLoader(in_file,sample_rate, shuffle=True, mode='train')
        else:
            self.utt_loader = PerUttLoader(in_file,sample_rate, shuffle=False, mode='eval')
        if unit == 'seg':
            self.seg_splitter = SegSplitter(segment_size, sample_rate, hop_size=segment_shift)
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def make_batch(self, load_list):
        n_batches = len(load_list) // self.batch_size
        if n_batches == 0:
            return []
        else:
            batch_queue = [[] for _ in range(n_batches)]
            idx = 0
            for seg in load_list[0:n_batches*self.batch_size]:
                batch_queue[idx].append(seg)
                idx = (idx + 1) % n_batches
            if self.unit == 'utt':
                for batch in batch_queue:
                    sig_len = max([eg.shape[0] for eg in batch])
                    for i in range(len(batch)):
                        pad_size = sig_len - batch[i].shape[0]
                        batch[i] = F.pad(batch[i], (0, pad_size))
            return batch_queue

    # def to_tensor(self, x) :
    #     return x.float()

    def batch_buffer(self):
        while True:
            try:
                utt_eg = next(self.load_iter)
                if self.unit == 'seg':
                    segs = self.seg_splitter(utt_eg)
                    self.load_list.extend(segs)
                else:
                    self.load_list.append(utt_eg)
            except StopIteration:
                self.stop_iter = True
                break
            if len(self.load_list) >= self.buffer_size:
                break
        
        batch_queue = self.make_batch(self.load_list)
        batch_list = []
        for eg_list in batch_queue:
            batch = torch.stack(eg_list, dim=0)
            batch_list.append(batch)
        # drop used segments and keep remaining segments
        rn = len(self.load_list) % self.batch_size
        self.load_list = self.load_list[-rn:] if rn else []
        return batch_list

    def __iter__(self):
        self.load_iter = iter(self.utt_loader)
        self.stop_iter = False
        self.load_list = []
        while True:
            if self.stop_iter:
                break
            egs_buffer = self.batch_buffer()
            for egs in egs_buffer:
                yield egs

