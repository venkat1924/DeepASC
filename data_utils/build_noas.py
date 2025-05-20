import os
import sys
import time
import torch
import torchaudio

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_FOLDER = f"{SCRIPT_FOLDER}/.."
sys.path.insert(0, PARENT_FOLDER)

from tools.simulator import RIRGenSimulator, ANSTISIGNAL_ERROR, NOISE_ERROR
from tools.helpers import get_device, nmse
from data_utils.data_loaders import get_dataloaders, get_eval_dataloader
from data_utils.processor import get_tr_list
from data_utils.data_tools import sef, factors


device = get_device()
reverberation_times = [0.15, 0.175, 0.2, 0.225, 0.25]
sr = 16000

simulator = RIRGenSimulator(sr, reverberation_times, device, hp_filter=True,v=3)
data_loader = get_dataloaders(key="20k_3s_", data_files=[], batch_size=8, max_size=22223,train_ratio=1, seconds=3,random_cutoff=False, filenames=True)

dir_name = f"{SCRIPT_FOLDER}/data/Audioset/optimal_3s_v3/"
os.makedirs(dir_name, exist_ok=True)
def save_signals(signals, file_path,sample_idx, t60):

    for i, signal in enumerate(signals):
        file_name = f"{os.path.basename(file_path[i]).split('.')[0]}_{sample_idx[i]}_{str(t60).replace('.','-')}.wav"
        torchaudio.save(f"{dir_name}{file_name}", signal.unsqueeze(0).detach().cpu(), sr)

def compute_nmse(target, signal, t60):
    st = simulator.simulate(signal, t60, signal_type=ANSTISIGNAL_ERROR)
    return nmse(target, st)

def compute_nmse_avg(pt, signal, use_sef=False):
    if use_sef:
        st = torch.stack([torch.stack([simulator.simulate(sef(signal,factor), t60,signal_type=ANSTISIGNAL_ERROR) for t60 in reverberation_times]) for factor in factors])
        return nmse(pt, st).mean(dim=(0,1))
    else:
        st = torch.stack([simulator.simulate(signal, t60,signal_type=ANSTISIGNAL_ERROR) for t60 in reverberation_times])
        return nmse(pt, st).mean(dim=0)

def find_noas(signal):
    pt = torch.stack([simulator.simulate(signal, t60,signal_type=NOISE_ERROR) for t60 in reverberation_times])

    current_try = torch.tensor(signal).clone().requires_grad_(True)
    current_optimal = current_try.clone()
    current_try_nmse = compute_nmse_avg(pt, current_try, use_sef=True)
    print(f"Starting NMSE: {current_try_nmse.mean()}")
    current_optimal_nmse = current_try_nmse.clone()

    optimizer = torch.optim.Adam([current_try], lr=0.01)
    steps = 0
    target_nmse = -11
    nmse_factor = 2
    drops = 0
    while steps < 2000:
        optimizer.zero_grad()
        current_try_nmse.sum().backward()
        optimizer.step()
        # current_optimal.grad.zero_()
        current_try_nmse = compute_nmse_avg(pt, current_try, use_sef=True)
    
        current_optimal = torch.where((current_optimal_nmse > current_try_nmse).unsqueeze(1), current_try, current_optimal).clone()
        current_optimal_nmse = compute_nmse_avg(pt, current_optimal, use_sef=True)
        steps += 1
        
        if drops < 2 and (current_optimal_nmse.mean() < target_nmse or steps % 800 == 0):
            print(f"Dropped lr NMSE: {current_optimal_nmse.mean()}, max:{current_optimal_nmse.max()} after {steps} steps", flush=True)
            optimizer.param_groups[0]['lr'] *= 0.3
            target_nmse -= nmse_factor
            drops += 1


    return current_optimal, current_optimal_nmse, steps



def main():
    # rng = [150,300]
    total_nmse = 0
    total_samples = 0 
    # for i, (signal, file_paths, sample_idx) in enumerate(data_loader):
    for i, (signal, file_paths, sample_idx) in enumerate(data_loader):
        # if i < rng[0] or i > rng[1]:
        #     continue
        signal = signal.to(device)
        optimals, _nmse, steps = find_noas(signal)
        if torch.isnan(_nmse.mean()):
            print(f"ERROR")
        else:
            total_nmse += _nmse.mean()*signal.shape[0]
            total_samples += signal.shape[0]
        # print(f"[{i + 1}/{len(data_loader)}] File: {file_path[0]}_{sample_idx.item()} - T60: {t60} - Steps: {steps} - NMSE: {total_nmse/total_samples}")
        print(f"[{i + 1}/{len(data_loader)}] {[os.path.basename(file_path) for file_path in file_paths]} - Steps: {steps} - NMSE: {total_nmse/max(1,total_samples)} [{_nmse.mean()}]", flush=True)
        save_signals(optimals, file_paths,sample_idx, "avg")



if __name__ == "__main__":
    print("PROCESS ID:" + str(os.getpid()))

    print("time: " + time.strftime('%d-%H-%M-%S'))

    main()