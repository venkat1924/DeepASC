import numpy as np
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

PARENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_PARENT_FOLDER = os.path.dirname(PARENT_FOLDER)
sys.path.append(PARENT_PARENT_FOLDER)
sys.path.append(f"{PARENT_PARENT_FOLDER}/DeepASC")

from data_utils.data_tools import sef
from tools.simulator import RIRGenSimulator, ANSTISIGNAL_ERROR, NOISE_ERROR
from tools.utils import get_wavfile_for_eval
from tools.helpers import get_device, nmse

from data_utils.data_loaders import get_dataloaders, get_eval_dataloader, get_speech_test_dataloader

device = 'cuda'
rir_samples=512
sr = 16000
simulator = RIRGenSimulator(sr=sr, reverbation_times=[0.2], device=device, rir_samples=rir_samples, hp_filter=True, v=3)
fftconvolve_valid = torchaudio.transforms.FFTConvolve(mode="valid")

class FxNLMS():
    
    def __init__(self, w_len, mu):
        self.grads = 0
        self.w = torch.zeros(1, w_len, dtype=torch.float).to(device)
        self.x_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.st_buf = torch.zeros(1, w_len, dtype= torch.float).to(device)
        self.mu = mu
        # self.w.requires_grad = True
        # self.optimizer= torch.optim.SGD([self.w], lr=mu)
    
    def predict(self,x, st):
        self.x_buf = torch.roll(self.x_buf,1,1)
        self.x_buf[0,0] = x
        yt = self.w @ self.x_buf.t()
        
        self.st_buf = torch.roll(self.st_buf,1,1)
        self.st_buf[0,0] = st
        power = self.st_buf @ self.st_buf.t() # FxNLMS different from FxLMS
        return yt, power
    
    def step(self, loss):
        loss = torch.clamp(loss, -1e-03, 1e-03)
        grad = self.mu * loss * self.st_buf.flip(1)
        # grad = torch.clamp(grad, -1e-06, 1e-06)
        # grad = torch.nn.functional.normalize(grad, dim=1) * 5e-05
        # self.grads += grad
        self.w += grad
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        
def get_score(noisy_signal, anti_signal, clean_signal):
    noisy_pt = simulator.simulate(noisy_signal, 0.2, NOISE_ERROR)  
    anti_st = simulator.simulate(anti_signal, 0.2, ANSTISIGNAL_ERROR)
    
    clean_pt = simulator.simulate(clean_signal, 0.2, NOISE_ERROR)
    return nmse(clean_pt, noisy_pt + anti_st).item()  


def execute_her(wavfile, mu,simulator=simulator,gama="inf", reverbation_time=0.2,thf=False):
    fxlms = FxNLMS(w_len=rir_samples, mu=mu)
    st = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].squeeze(0).to(device)
    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)

    padded_signal = torch.nn.functional.pad(wavfile, (rir_samples//2,0), mode='constant', value=0)
    st = simulator.rirs[(reverbation_time, ANSTISIGNAL_ERROR)].squeeze(0).to(device)
    y_buf = torch.zeros(1, rir_samples, dtype=torch.float).to(device)
    pt_signal = simulator.simulate(padded_signal, reverbation_time, NOISE_ERROR)[0]

    padded_signal = torch.tanh(padded_signal) if thf else padded_signal
    st_signal = simulator.simulate(padded_signal, reverbation_time, ANSTISIGNAL_ERROR)[0]
    
    ys = []
    len_data = pt_signal.shape[0]
    for i in range(len_data - rir_samples//2):
        # Feedfoward
        xin = st_signal[i]
        dis = pt_signal[i]
        
        y, power = fxlms.predict(wavfile[0,i], xin)
        y_buf = torch.roll(y_buf, -1, 0)
        y_buf[0, -1] = y
        y_buf_sef = sef(y_buf, gama=gama)
        sy = st @ y_buf_sef.t().flip(0)
        
        loss = sy - dis

        fxlms.step(loss)
        ys.append(y.item())
        
    return ys      

def test():
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    clean_signal = resampler(torchaudio.load("compare/clean_p232_001.wav")[0]).to(device)
    noisy_signal = resampler(torchaudio.load("compare/noisy_p232_001.wav")[0]).to(device)
        
    anti_signal = execute_her(clean_signal, noisy_signal, mu=0.1, simulator=simulator, gama="inf", reverbation_time=0.2, thf=False)
        # print(f"NMSE LOSS is: {get_score(wavfiles, anti_signal)}", flush=True)
        
if __name__ == "__main__":
    test()
