import os 
import numpy as np
import torch
import torchaudio
import gpuRIR
import pyroomacoustics as pra
import rir_generator
import scipy.io
import random

file_dir = os.path.dirname(os.path.abspath(__file__))
fftconvolve = torchaudio.transforms.FFTConvolve(mode="same")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = f"{SCRIPT_PATH}/../data_utils"

NOISE_REF = 0
NOISE_ERROR = 1
ANSTISIGNAL_ERROR = 2

def inverse_filter_v1(h):
    # Compute the FFT of the signal and the filter
    filter_fft = torch.fft.rfft(h)

    # Handle the case where the filter might be zero
    eps = 1e-6
    filter_fft = torch.where(filter_fft.abs() > eps, filter_fft, eps)
    return torch.fft.irfft(1 / filter_fft)

def wiener_filter(signal, rir, K=0.001):
    rir /= torch.sum(rir)
    n = signal.size(-1) - rir.size(-1) + 1
    signal_fft = torch.fft.fft(signal, n=n, dim=-1)
    rir_fft = torch.fft.fft(rir, n=n, dim=-1)
    _filter = torch.conj(rir_fft) / (torch.abs(rir_fft) ** 2 + K)
    inv_signal_fft = signal_fft * _filter
    inv_signal = torch.fft.ifft(inv_signal_fft, dim=-1).real
    return inv_signal

def weiner_wiki(signal, rir, SNR=0.00005):
    n = signal.size(-1) - rir.size(-1) + 1
    H = torch.fft.fft(rir, n=n, dim=-1)
    
    # S is the mean power spectral density of the original signal 
    signal_fft = torch.fft.fft(signal, n=n, dim=-1)
    G = (1 / H) * (1 / (1 + (1/H.abs() ** 2) * SNR))
    # G = (1 / H) * (1 / (1 + (1/(((H.abs() ** 2).T * SNR)).T)))

    inv_signal_fft = signal_fft * G
    inv_signal = torch.fft.ifft(inv_signal_fft, dim=-1).real
    return inv_signal

def _simulate(signal_batch, rir, device, padding="same"):
    signal_batch = signal_batch.to(device).unsqueeze(1)
    processed_signals = torch.nn.functional.conv1d(signal_batch, rir, padding=padding)
    processed_signals = processed_signals.squeeze(1)
    return processed_signals

def _simulate_v2(signal_batch, rir, device, padding="same"):
    signal_batch = signal_batch.to(device).unsqueeze(1)
    
    # Apply the filter in the forward direction
    processed_signals = torch.nn.functional.conv1d(signal_batch, rir, padding=padding)
    
    # Reverse the filtered signal
    processed_signals = torch.flip(processed_signals, [2])
    
    # Apply the filter again in the forward direction
    processed_signals = torch.nn.functional.conv1d(processed_signals, rir, padding=padding)
    
    # Reverse the signal back to its original order
    processed_signals = torch.flip(processed_signals, [2])
    
    processed_signals = processed_signals.squeeze(1)
    return processed_signals

class BoseSimulator:
    def __init__(self, sr, reverbation_times, device, rir_samples=512, hp_filter=False, c=343,v=1,trun=None):
        self.sr = sr
        self.device = device
        self.rir_length = rir_samples
        self.reverbation_times = reverbation_times
        self.base_path = f"{DATA_PATH}/BOSEdb/pandar_db/BoseQC20/acoustic_booth"
        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        self.trun=trun
        print(f"TRUNCATATION OF RIR FILTERS IS: {self.trun}. reverbation_times length is {len(self.reverbation_times)}")
        self.rirs = self.get_rirs()

    def get_id_str(self,_id):
        if _id < 10:
            return f"0{_id}"
        return str(_id)


    def get_rirs(self):
        rirs = dict()

        primary_paths = [f"{self.base_path}/persons/PANDAR_TF_0{self. get_id_str(t60)}_person_BoseQC20.ita" for t60 in self.reverbation_times]
        for i, p_id in enumerate(self.reverbation_times):
            p = primary_paths[i]
            mat_data = scipy.io.loadmat(p)
            ita_object = mat_data['ITA_TOOLBOX_AUDIO_OBJECT']
            
            rirs[(p_id, NOISE_ERROR)] = torch.tensor(ita_object['data'][0][0][:,4]).to(torch.float32) # PrimaryL
            rirs[(p_id, NOISE_ERROR)] = self.resample(rirs[(p_id, NOISE_ERROR)]).view(1,1,-1).to(self.device)

            rirs[(p_id, ANSTISIGNAL_ERROR)] = torch.tensor(ita_object['data'][0][0][:,0]).to(torch.float32) # SecondaryL
            rirs[(p_id, ANSTISIGNAL_ERROR)] = self.resample(rirs[(p_id, ANSTISIGNAL_ERROR)]).view(1,1,-1).to(self.device)

            if self.trun is not None:
                rirs[(p_id, NOISE_ERROR)] = rirs[(p_id, NOISE_ERROR)][:,:,:self.trun]
                rirs[(p_id, ANSTISIGNAL_ERROR)] = rirs[(p_id, ANSTISIGNAL_ERROR)][:,:,:self.trun]
            
        return rirs
    
    def simulate(self, signal_batch,  p_id=None, signal_type=ANSTISIGNAL_ERROR):
        p_id = random.randint(1, 23) if p_id is None else p_id
        rir = self.rirs[(p_id, signal_type)]
        return fftconvolve(signal_batch, rir.squeeze(0))


class RIRGenSimulator:
    def __init__(self, sr, reverbation_times, device, rir_samples=512, hp_filter=False, c=343,v=3):
        self.sr = sr
        self.device = device
        self.room_dim = [3, 4, 2]
        self.ref_mic = [1.5, 1, 1]
        self.ls_source = [1.5, 2.5, 1]
        self.error_mic = [1.5, 3, 1]
        self.reverbation_times = reverbation_times
        self.rir_length = rir_samples
        self.hp_filter = hp_filter
        self.c = c
        self.rirs = self.get_rirs()
        self.v = v

    def get_rirs(self):
        rirs = dict()
        for t60 in self.reverbation_times:
            for rir_type in [NOISE_ERROR, ANSTISIGNAL_ERROR]:
                if rir_type == NOISE_ERROR:
                    pos_src = self.ref_mic
                    pos_rcv = self.error_mic
                elif rir_type == ANSTISIGNAL_ERROR:
                    pos_src = self.ls_source
                    pos_rcv = self.error_mic
                rir = rir_generator.generate( # consider hp_filter = False
                    c=self.c,
                    fs=self.sr,
                    s=pos_src,
                    r=[pos_rcv],
                    L=self.room_dim,
                    reverberation_time=t60,
                    nsample=self.rir_length,
                    hp_filter=self.hp_filter
                )
                rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir)).to(self.device).view(1, 1, -1).float()
        return rirs
        

    def simulate(self, signal_batch, t60, signal_type, padding="same"):
        rir = self.rirs[(t60, signal_type)]
        if self.v == 1:
            return _simulate(signal_batch, rir, self.device, padding)
        elif self.v == 2:
            return _simulate_v2(signal_batch, rir, self.device, padding)
        else:
            return fftconvolve(signal_batch, rir.squeeze(0))


class PyRoomSimulator:
    def __init__(self, sr, reverbation_times, device, rir_samples=512):
        self.sr = sr
        self.device = device
        self.room_dim = np.array([3, 4, 2])
        self.ref_mic = np.array([1.5, 1, 1])
        self.ls_source = np.array([1.5, 2.5, 1])
        self.error_mic = np.array([1.5, 3, 1])
        self.reverbation_times = reverbation_times
        self.rir_length = rir_samples
        self.rirs = self.get_rirs()

    def get_rirs(self):
        rirs = dict()

        for t60 in self.reverbation_times:
            e_absorption, max_order = pra.inverse_sabine(t60, self.room_dim)
            for rir_type in [NOISE_ERROR, ANSTISIGNAL_ERROR]:
                room = pra.ShoeBox(self.room_dim, fs=self.sr,  materials=pra.Material(e_absorption), max_order=max_order)

                if rir_type == NOISE_ERROR:
                    pos_src = self.ref_mic
                    pos_rcv = self.error_mic
                elif rir_type == ANSTISIGNAL_ERROR:
                    pos_src = self.ls_source
                    pos_rcv = self.error_mic

                wav,sr = torchaudio.load("source3.wav")
                room.add_source(pos_src,signal=wav.flatten().numpy())

                mic = pra.MicrophoneArray(pos_rcv.reshape((-1, 1)), self.sr)
                room.add_microphone_array(mic)

                room.simulate()
                room.mic_array.to_wav(
                    f"primary_source3_v2.wav",
                    # norm=False,
                    # bitdepth=np.int16,
                )
                room.compute_rir()
                rir = room.rir[0][0]
                # make RIR adjustments to ISM model (by pyroomacooustics maintainer https://github.com/DavidDiazGuerra/gpuRIR/issues/61)
                rir_ism = rir[40:40+self.rir_length] * (1/(torch.pi * 4))
                rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir)).to(self.device).view(1, -1).float()
        return rirs
        

    def simulate(self, signal_batch, t60, signal_type):
        rir = self.rirs[(t60, signal_type)]
        return fftconvolve(signal_batch, rir)

class GPUSimulator:
    def __init__(self, sr, reverbation_times, device, rir_samples=512):
        self.sr = sr
        self.reverbation_times = reverbation_times
        self.device = device
        self.room_dim = np.array([3, 4, 2])
        self.ref_mic = np.array([[1.5, 1, 1]])
        self.ls_source = np.array([[1.5, 2.5, 1]])
        self.error_mic = np.array([[1.5, 3, 1]])
        self.rir_length = rir_samples / self.sr # length of the RIR in seconds
        self.nb_img = gpuRIR.t2n(T=self.rir_length, rooms_sz=self.room_dim)

        self.rirs = self.get_rirs()

    def get_rirs(self):
        rirs = dict()

        for t60 in self.reverbation_times:
            beta = gpuRIR.beta_SabineEstimation(self.room_dim, t60)

            for rir_type in [NOISE_ERROR, ANSTISIGNAL_ERROR]:
                if rir_type == NOISE_ERROR:
                    pos_src = self.ref_mic
                    pos_rcv = self.error_mic
                elif rir_type == ANSTISIGNAL_ERROR:
                    pos_src = self.ls_source
                    pos_rcv = self.error_mic
                rir = gpuRIR.simulateRIR(self.room_dim, beta, pos_src, pos_rcv, self.nb_img, Tmax=self.rir_length, fs=self.sr)
                rirs[(t60, rir_type)] = torch.from_numpy(np.squeeze(rir)).to(self.device).view(1, 1, -1)
                rirs[(t60, rir_type)].requires_grad_(False)
        return rirs
    

    def simulate(self, signal_batch, t60, signal_type):
        rir = self.rirs[(t60, signal_type)]
        return _simulate(signal_batch, rir, self.device)
    

if __name__ == "__main__":
    device = "cuda"
    wav,sr = torchaudio.load("source4.wav")
    wav = wav.to(device)

    simulator = PyRoomSimulator(sr, [0.2], device)

    primary_wav = simulator.simulate(wav, 0.2, NOISE_ERROR)

    torchaudio.save("primary_source4.wav", primary_wav.cpu(), sr)