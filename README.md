# Deep Active Speech Cancellation with Mamba-Masking Network

This repository contains the implementation of our method for Active Speech Cancellation (ASC) using a masking mechanism that directly interacts with the encoded reference signal. This README provides instructions on how to train and fine-tune the model, as well as how to build the necessary datasets.

You can listen to our demo audio samples [here](https://mishalydev.github.io/DeepASC-Demo/).

## Repository Structure

- `DeepASC/`: Contains the main method implementations.
- `DeepASC/hparams/exp/`: Contains the YAML configuration files for experiments.
- `data_utils/`: Contains utilities for data processing and dataset creation.
- `baselines`: Contains our implementations of ANC baseline methods.

## Getting Started

### Prerequisites
- Required Python packages (can be installed via `requirements.txt`)
- Note: Depending on your `mamba_ssm` version, the `layernorm` module may be `mamba_ssm.ops.triton.layer_norm` or `mamba_ssm.ops.triton.layernorm` (without underscore). Update import statements as needed.

### Dataset Preparation
We assume that your dataset consists of WAV files from Google Audioset is already under `data_utils/data/Audioset`. For reproducibility, we also provide the list of Google Audioset noise categories used in our paper in `data_utils/data/Audioset/category_names.txt`.

Before using the dataset for the first time, you need to create a txt file listing all the WAV file names and save it as `data_utils/meta/Audioset.txt`. You can use the `export_filenames_to_txt` function in `data_utils/processor.py` for this purpose.

As described in the paper, we use 3-second noise samples at 16 kHz. 
Each noise signal is passed through the room simulator during train/test time.

Note: Our code assumes that the audio files are in mono channel format. Please ensure that your audio files are not in stereo format.


#### Building the NOAS Signals Dataset

To build the NOAS signals dataset, run the following command:

```bash
python data_utils/build_noas.py
```

### Training the Model

#### ANC Training

To train the model using for ASC/ANC tasks, run the following command:

```bash
python DeepASC/train.py {yaml_file}.yaml
```

#### Fine-Tuning with NOAS Optimization

To fine-tune the model using NOAS optimization, run the following command:

```bash
python DeepASC/train_noas.py
```

## Room Acoustics Simulation

This repository provides a unified interface for simulating room acoustics using several Room Impulse Response (RIR) simulators, all accessible via a common interface in `tools/simulator.py`. This design allows you to easily switch between different simulation packages with minimal code changes.

**Available Simulators:**
- **RIRGenSimulator**: Generates synthetic RIRs using the `rir_generator` package.
- **PyRoomSimulator**: Generates synthetic RIRs using the `Pyroomacoustics` package.
- **GPUSimulator**: Generates synthetic RIRs using the `gpuRIR` package.
- **BoseSimulator**: Uses real-world measured RIRs from Bose in-ear headphones [[Liebich et al., 2019]](#references).

**Unified Interface:**
All simulators implement a `simulate` method:
```python
simulate(signal_batch, t60_or_pid, signal_type, padding="same")
```
- `signal_batch`: Tensor of shape `[batch, samples]` (mono audio).
- `t60_or_pid`: Reverberation time (T60) or person ID, depending on simulator.
- `signal_type`: Type of signal path (e.g., `NOISE_ERROR`, `ANSTISIGNAL_ERROR`).
- `padding`: (Optional) Convolution padding mode (`"same"` by default).

**Shared Configuration & Assumptions:**
- **RIR Length**: All simulators use an RIR length of 512 samples by default.
- **Room Geometry**: `[3, 4, 2]` meters.
- **Microphone & Source Positions**:
    - Reference mic: `[1.5, 1, 1]`
    - Error mic: `[1.5, 3, 1]`
    - Loudspeaker source: `[1.5, 2.5, 1]`

**Convolution Variants (`v` argument):**
`RIRGenSimulator` simulator supports different convolution strategies via the `v` argument:
- `v=1`: Standard 1D convolution.
- `v=2`: Forward-backward convolution (applies the filter twice, with signal reversal).
- Default (shared for all simulators): Uses FFT-based convolution for efficiency.


## Usage

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Prepare your dataset as described in the Dataset Preparation section.

3. Train the model using one of the training methods described above.


> **Note:** If you encounter any issues or something doesn't work as expected, please let us know by opening an issue. We will do our best to address and fix it as soon as possible!

## References

- Liebich, S., Fabry, J., Jax, P., & Vary, P. (2019). Acoustic path database for ANC in-ear headphone development. [Link](https://api.semanticscholar.org/CorpusID:204793245)