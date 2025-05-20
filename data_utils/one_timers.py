import os
import glob
import soundfile as sf
import wave
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils.data_tools import DataProvider
from tools.helpers import get_device, nmse
from tools.simulator import RIRGenSimulator, NOISE_ERROR, ANSTISIGNAL_ERROR
import random

device = get_device()
t60s =  [0.15, 0.175, 0.2, 0.225, 0.25]
simulator = RIRGenSimulator(16000,t60s, device, hp_filter=True, v=3)

def convert_flac_to_wav(flac_file_path, wav_file_path):
    """Convert a single FLAC file to WAV format."""
    data, samplerate = sf.read(flac_file_path)
    sf.write(wav_file_path, data, samplerate)
    os.remove(flac_file_path)

def convert_all_flacs_to_wavs(root_dir):
    """Convert all FLAC files in the root_dir and its subdirectories to WAV."""
    flac_files = glob.glob(os.path.join(root_dir, '**/*.flac'), recursive=True)
    for i, flac_file in enumerate(flac_files):
        wav_file = os.path.splitext(flac_file)[0] + '.wav'
        convert_flac_to_wav(flac_file, wav_file)

        print("Converted file {}/{}: {} -> {}".format(i + 1, len(flac_files), flac_file, wav_file))
        # Optionally, remove the original FLAC file
        # os.remove(flac_file)

def print_fxlms_mean():
    # Step 1: Open the file in read mode
    with open('data/code/speechbrain/_separation/data_utils/logs/fxlms_wsj_5s.log', 'r') as file:
        # Step 2: Initialize variables to store the sum of scores and the count of scores
        total_score = 0
        count = 0
        
        # Step 3: Iterate through each line in the file
        for line in file:
            # Step 4: Check if the line contains a score
            if "score" in line:
                # Step 5: Extract the score value from the line
                score = float(line.split(':')[1].strip())
                # Step 6: Add the score to the total_score and increment the count
                if score < 1:
                    total_score += score
                    count += 1
                
        
        # Step 7: Calculate the mean score
        mean_score = total_score / count if count else 0
        
        # Step 8: Print the mean score
        print(f'Mean Score: {mean_score}, count: {count}')

def calc_noas_dataset_score():
    data_provider = DataProvider(batch_size=32,max_size=22223, data_key="20k_3s", seconds=3, v=4)
    train_loader = data_provider.train_loader
    
    scores = []
    for i, (signals, noass) in enumerate(train_loader):
        signals, noass = signals.to(device), noass.to(device)
        
        t60 = random.choice(t60s)
        
        pts = simulator.simulate(signals, t60=t60, signal_type=NOISE_ERROR)
        sts = simulator.simulate(noass, t60=t60, signal_type=ANSTISIGNAL_ERROR)
        
        scores.append(nmse(pts, sts).item())
        
        print(f"[{i}/{len(train_loader)}] {(sum(scores)/len(scores)):.5f} [Score: {scores[-1]}]")


if __name__ == '__main__':
    # librispeech_dir = './data/LibriSpeech/test-clean'  # Update this path to your LibriSpeech folder
    # convert_all_flacs_to_wavs(librispeech_dir)
    calc_noas_dataset_score()