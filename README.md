# Deep Active Speech Cancellation with Multi-Band Mamba Network

This repository contains the implementation of our method for Active Speech Cancellation (ASC) using a multi-band mamba architecture. This README provides instructions on how to train and fine-tune the model, as well as how to build the necessary datasets.

## Repository Structure

- `DeepASC/`: Contains the main method implementations.
- `DeepASC/hparams/exp/`: Contains the YAML configuration files for experiments.
- `data_utils/`: Contains utilities for data processing and dataset creation.
- `compare`: Contains our implementations of deep-learning baselines ANC.

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

## Usage

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your dataset as described in the Dataset Preparation section.

4. Train the model using one of the training methods described above.
