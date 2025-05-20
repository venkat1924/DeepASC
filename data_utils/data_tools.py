from itertools import cycle
import random
import torch
from data_utils.processor import get_tr_list
from data_utils.data_loaders import get_eval_dataloader, get_dataloaders, get_speech_test_dataloader


def get_noise_types():
    return ['babble', 'factory1', 'factory2', 'destroyerengine']

class DataProvider():
    def __init__(self, batch_size, max_size, data_key, seconds=5, v=1, filenames=False):
        self.data_key = data_key

        data_files = get_tr_list()
        self.train_loader, self.valid_loader = get_dataloaders(key=data_key,data_files=data_files, batch_size=batch_size,max_size=max_size,train_ratio=0.9, seconds=seconds,v=v, filenames=filenames)


sqrt_2 = 2 ** 0.5
sqrt_pi_2 = (torch.pi/2) ** 0.5
factors = [0.1**0.5, 1**0.5, 10**0.5, "inf"]
def get_random_factor():
    return factors[random.randint(0, len(factors) - 1)]

def sef(tensor, factor="inf"):
    # u = x/(sqrt(2)*factor), du = 1/(sqrt(2)*factor) dx => dx = sqrt(2)*factor * du
    # integral(e^(-(x^2)/(2*factor^2)), 0, tensor) dx =
    # integral(e^(-(x/(sqrt(2)*factor))^2), 0, tensor) dx = 
    # integral(e^(-u^2) * sqrt(2)*factor, 0, tensor-u) du = 
    # sqrt(2)*factor * integral(e^(-u^2), 0, tensor-u) du = 
    # sqrt(2)*factor * ((sqrt(pi)/2) * erf(tensor-u) - (sqrt(pi)/2) * erf(0))  = 
    # sqrt(2)*factor * sqrt(pi)/2 * (erf(tensor-u) - erf(0))  = 
    # factor * sqrt(pi/2) * (erf(tensor/(sqrt(2)*factor)) - erf(0))) [because u = x/(sqrt(2)*factor)]
    if factor == "random":
        factor =  get_random_factor()
    
    if factor == "inf":
        return tensor

    return factor * sqrt_pi_2 * (torch.erf(tensor/(sqrt_2*factor)))

def delay_signal(hop_len, frames_delay, signals):
    if frames_delay == 0:
        return signals
    
    pad_len = frames_delay * hop_len
    pad = torch.zeros((signals.shape[0],pad_len)).to(signals.device)
    signals = torch.cat([pad, signals[:,:-pad_len]],dim=-1)
    return signals

def get_scaled_speech(signal, noise, snr):
    """Combines signal and noise at a desired SNR level."""
    snr_linear = 10 ** (snr / 10.0)
    
    # Calculate the power of the signal and noise
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate the required noise power for the given SNR
    required_noise_power = signal_power / snr_linear
    
    # Scale the noise to the required power
    signal = signal / torch.sqrt(required_noise_power / noise_power)
    
    # Add the scaled noise to the signal
    # noisy_signal = signal + noise
    
    return signal

def get_scaled_noise(signal, noise, snr):
    """Combines signal and noise at a desired SNR level."""
    snr_linear = 10 ** (snr / 10.0)
    
    # Calculate the power of the signal and noise
    signal_power = torch.mean(signal ** 2, dim=-1)
    noise_power = torch.mean(noise ** 2, dim=-1)
    
    # Calculate the required noise power for the given SNR
    required_noise_power = signal_power / snr_linear
    
    # Scale the noise to the required power
    noise = noise * torch.sqrt(required_noise_power / noise_power).unsqueeze(-1)
    
    # Add the scaled noise to the signal
    # noisy_signal = signal + noise
    noise = torch.where(torch.isnan(noise), 0, noise)
    
    return noise

def combine_signal_noise(signal, noise, snr):
    """Combines signal and noise at a desired SNR level."""
    scaled_speech = get_scaled_speech(signal, noise, snr)
    noisy_signal = scaled_speech + noise
    
    return noisy_signal