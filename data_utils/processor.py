import os
import glob
import sys
import torchaudio
import torch
# from pydub import AudioSegment
import os
import csv

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_FOLDER = f"{SCRIPT_FOLDER}/.."
sys.path.insert(0, PARENT_FOLDER)


data_sets = {
    'Audioset',
    # 'FSDnoisy18k.audio_train',
    # 'arca23k-dataset',
    # 'demand'
}


class WavSample:
    def __init__(self, data, sr, name):
        self.data = data
        self.sr = sr
        self.name = name

def txtfile_name(ds):
    return f'{SCRIPT_FOLDER}/meta/{ds.replace("/", "_")}.txt'


def export_filenames_to_txt(ds,recursive=True):
    data_dir = f'data_utils/data/{ds}'
    # Get a list of all .wav files in the directory
    wav_files = [file for file  in glob.glob(os.path.join(data_dir, '**' if recursive  else '','*.wav'), recursive=recursive) if 'downloads' not in file]
    # write all the filenames to a txt file
    with open(txtfile_name(ds), 'w') as f:
        for wav_file in wav_files:
            f.write(wav_file + '\n')


def get_tr_list(data_sets=data_sets):
    tr_list = []
    for ds in data_sets:
        with open(txtfile_name(ds=ds), 'r') as f:
            files = [line.strip() for line in f.readlines()]
            print(f"Loaded from {ds}: {len(files)} files")
            tr_list += files

    return tr_list


if __name__ == '__main__':
    # files = get_tr_list(data_sets={"timit_TIMIT_TRAIN"})
    # export_filenames_to_txt(ds="Audioset")
    export_filenames_to_txt(ds="NoiseX-92")
    


