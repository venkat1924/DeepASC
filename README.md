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
We assume that your training set is located under `data_utils/data/Audioset`. Before using the dataset for the first time, you need to create a txt file with all the WAV file names in `data_utils/meta/Audioset.txt`. You can use the `export_filenames_to_txt` function in `data_utils/processor.py` for this purpose.

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
- **BoseSimulator**: Uses measured RIRs from the \ dataset.
- **RIRGenSimulator**: Generates synthetic RIRs using the `rir_generator` package.
- **PyRoomSimulator**: Generates synthetic RIRs using the `Pyroomacoustics` package.
- **GPUSimulator**: Generates synthetic RIRs using the `gpuRIR` package.

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

3. Prepare your dataset as described in the Dataset Preparation section.

4. Train the model using one of the training methods described above.
