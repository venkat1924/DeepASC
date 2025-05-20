#!/usr/bin/env/python3

'''
Copied and modified from
https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/train.py
'''

"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import copy
import logging
import os
import sys
import random
from types import SimpleNamespace

import numpy as np
import torch
print(f"devices #{torch.cuda.device_count()}")
# torch.cuda.set_device(3)
torch.cuda.empty_cache()

from hyperpyyaml import load_hyperpyyaml
# from tqdm import tqdm


import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.core import AMPConfig
from speechbrain.nnet.loss.si_snr_loss import si_snr_loss

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_FOLDER = f"{SCRIPT_FOLDER}/.."
sys.path.append(PARENT_FOLDER)

from data_utils.data_tools import DataProvider, get_noise_types, sef
from tools.simulator import ANSTISIGNAL_ERROR, NOISE_ERROR, RIRGenSimulator
from tools.helpers import load_model, save_model, get_device
from tools.utils import pad_num_to_len

from models import get_model

from torch.utils.tensorboard import SummaryWriter
import time
import sys

from tools.helpers import nmse

os.environ['WANDB__SERVICE_WAIT'] = '999999'
device = get_device()
# print(f"device: {device}")
logger = logging.getLogger(__name__)

# play with - data, batch, simulator, loss
yaml_file = f"DeepASC/hparams/exp/{sys.argv[1]}.yaml"

def get_hparams(file=yaml_file):
    yaml_name = f'{PARENT_FOLDER}/{file}'
    hparams_file, run_opts, overrides = sb.parse_arguments([yaml_name, "--device", device])
    with open(hparams_file) as fin:
        hparams = SimpleNamespace(**load_hyperpyyaml(fin, overrides))
    print("opened yaml file " + yaml_name)
    return hparams

hparams = get_hparams()
exp_name = hparams.exp_name
print( f"{hparams.sef}-sef|{hparams.data_key}|rir_generator|{hparams.lr}|bch-{hparams.batch_size}|")
name = exp_name
print(f"exp_name: {name}")
#  + "_".join(str(value) for value in hparams.prefix.values())

writer = SummaryWriter(log_dir=f"{SCRIPT_FOLDER}/runs/{name}")


class Trainer:
    def __init__(self, hparams, sr, model, optimizer, scheduler, first_epoch=0):
        self.reverberation_times = [0.15, 0.175, 0.2, 0.225, 0.25]
        self.sr = sr
        self.device = device
        self.simulator = RIRGenSimulator(self.sr, self.reverberation_times, self.device, hp_filter=True, v=3)
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.hparams = hparams
        self.step = 0
        self.avg_train_loss = 0.0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.first_epoch = first_epoch
        self.nonfinite_count = 0
        gradscaler_enabled = self.hparams.precision == "fp16" and "cuda" in self.device
        self.scaler = torch.cuda.amp.GradScaler(enabled=gradscaler_enabled)

    def compute_objectives(self, signals, anti_signals, t60):
        """Computes the sinr loss"""
        pt = self.simulator.simulate(signals.float(), t60, signal_type=NOISE_ERROR).type(signals.type())

        anti_signals = sef(anti_signals, hparams.sef)
        st = self.simulator.simulate(anti_signals.float(), t60 ,signal_type=ANSTISIGNAL_ERROR).type(signals.type()) 
        # raise Exception("Not implemented")
        loss = nmse(pt, st).to(signals.device)
        return loss

    def fit_batch(self, batch):
        """Trains one batch"""
        amp = AMPConfig.from_name(self.hparams.precision)
        t60 = self.reverberation_times[random.randint(0, len(self.reverberation_times) - 1)]
        signals = batch.to(self.device)

        with torch.autocast(
            dtype=amp.dtype, device_type=torch.device(self.device).type
        ):
            anti_signals = self.model(signals)
            loss = self.compute_objectives(signals, anti_signals, t60)

            # hard threshold the easy dataitems
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss = loss[loss > th]
                if loss.nelement() > 0:
                    loss = loss.mean()
            else:
                loss = loss.mean()

        if (
            loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
        ):  # the fix for computational problems
            self.scaler.scale(loss).backward()

            # print_grad(self.model, paths)
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.hparams.clip_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0.0).to(self.device)

        self.optimizer.zero_grad()
        return loss.detach().cpu()

    def __update_average(self, avg, val):
        if torch.isfinite(val):
            avg -= avg / self.step
            avg += float(val) / self.step
        return avg

    def _valid(self, valid_set, epoch):
        # Validation stage
        self.model.eval()

        avg_valid_loss = 0.0
        batches = len(valid_set)

        for i, batch in enumerate(valid_set):
            if torch.all(batch == 0):
                continue
            self.step += 1
            t60 = self.reverberation_times[random.randint(0, len(self.reverberation_times) - 1)]
            signals = batch.to(self.device)
            with torch.no_grad():
                anti_signals = self.model(signals)
                loss = self.compute_objectives(signals, anti_signals, t60).mean()

            avg_valid_loss = self.__update_average(avg_valid_loss, loss)
            
            print(f"[{i + 1}/{batches}] loss:{avg_valid_loss:.5f} [{pad_num_to_len(loss)}]", flush=True)

        # if epoch != 0:
        save_model(epoch=epoch, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, model_name=name,score=avg_valid_loss)
        
        self.step = 0
        self.model.train()

        # ReduceLROnPlateau
        if isinstance(
            self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
        ):
            current_lr, next_lr = self.hparams.lr_scheduler(
                [self.optimizer], epoch, avg_valid_loss
            )
            schedulers.update_learning_rate(self.optimizer, next_lr)
            print("LR: ",self.optimizer.param_groups[0]["lr"])
        # IntervalScheduler
        elif isinstance(
            self.hparams.lr_scheduler, schedulers.IntervalScheduler
        ):
            current_lr, next_lr = self.hparams.lr_scheduler(self.optimizer)
            print(self.optimizer.param_groups[0]["lr"])

        else:
            # no change
            current_lr = self.optimizer.param_groups[0]["lr"]
        
        writer.add_scalars("Loss/epoch", {
            f"test": avg_valid_loss,
            f"lr": current_lr,
        }, epoch)

    def _train(self, train_set,epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=False)
        batches = len(train_set)

        self.avg_train_loss = 0.0

        for i, batch in enumerate(train_set):
            if torch.all(batch == 0):
                print("skipping batch. All zeros")
                continue

            loss = self.fit_batch(batch)
            self.step += 1
            self.avg_train_loss = self.__update_average(self.avg_train_loss, loss)
            
            print(f"[{i + 1}/{batches}] loss:{self.avg_train_loss:.5f} [{pad_num_to_len(loss)}]", flush=True)
            

        self.optimizer.zero_grad(set_to_none=True)
        self.step = 0

        writer.add_scalars("Loss/epoch", {
            "train": self.avg_train_loss,
        }, epoch)
    
    def fit(
        self,
        train_set,
        valid_set,
    ):
        # Only show progressbar if requested and main_process
        epoch_counter = hparams.epoch_counter
        epoch_counter.current = self.first_epoch
        # self._valid(valid_set=valid_set, epoch=epoch_counter.current)

        # Iterate epochs
        for epoch in epoch_counter:
            print("EPOCH: ", epoch_counter.current, "lr: ", self.optimizer.param_groups[0]["lr"], flush=True)
            self._train(train_set=train_set, epoch=epoch_counter.current)
            self._valid(valid_set=valid_set, epoch=epoch_counter.current)


def get_trainer(model):
    optimizer = hparams.optimizer(model.parameters())
    scheduler = hparams.lr_scheduler
    epoch = 0
    if hparams.load:
        if hparams.load_tools:
            model, optimizer, scheduler, epoch = load_model(model,hparams.checkpoint, optimizer, scheduler, strict=hparams.strict, old_version=hparams.old_version)
        else:
            model, _, _, _ = load_model(model,hparams.checkpoint, strict=hparams.strict, old_version=hparams.old_version)
            
    print("lr: ", optimizer.param_groups[0]["lr"])
    print("halve after: ", scheduler.dont_halve_until_epoch)
        

    print(f"STARTING FROM EPOCH: { epoch }")
    return Trainer(
        hparams,
        sr=16000,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        first_epoch=epoch
    )
     
if __name__ == "__main__":
    # print the process id
    # cuda 11.7

    print("Process ID: ", os.getpid())
    print(f"{name}| RUNNING ON {device} {time.strftime('%d-%H-%M-%S')}")
    print(f"PRECISION: {hparams.precision}")

    model = get_model(hparams, getattr(hparams, "single_band", False))

    print(f"NUM OF PARAMS: ", sum(p.numel() for p in model.parameters()))

    trainer = get_trainer(model)

    # Load hyperparameters file with command-line overrides
    data_provider = DataProvider(batch_size=hparams.batch_size,max_size=hparams.data_samples, data_key=hparams.data_key, seconds=hparams.sample_length, filenames=False)
    train_data, valid_set = data_provider.train_loader, data_provider.valid_loader

    trainer.fit(
        train_set=train_data,
        valid_set=valid_set,
    )


