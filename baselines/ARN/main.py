import random
import sys
import os
SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PARENT_FOLDER = f"{SCRIPT_FOLDER}/../.."
sys.path.append(PARENT_FOLDER)
from data_utils.data_loaders import get_eval_dataloader
from data_utils.data_tools import DataProvider, get_noise_types, sef
from tools.simulator import ANSTISIGNAL_ERROR, NOISE_ERROR, RIRGenSimulator, PyRoomSimulator
from tools.utils import pad_num_to_len
from tools.helpers import load_model, nmse, save_model, get_device, RMSNormalizer, delay_signals

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import model_c as model_py
import ola as ola


from torch.utils.tensorboard import SummaryWriter
from utils import predict

from speechbrain.nnet.loss.si_snr_loss import si_snr_loss


device = get_device()
exp_name = "<CHOOSE_EXPERIMENT_NAME>"

writer = SummaryWriter(log_dir=f"{SCRIPT_FOLDER}/runs/{exp_name}")
print(f"PROCESS ID: {os.getpid()}",flush=True)
print(f"{exp_name}| RUNNING ON {device} {time.strftime('%d-%H-%M-%S')}")

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001, #0.0001
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--warmup', type=int, default=4000,
                    help='warmup for learning rate')
parser.add_argument('--cooldown', type=int, default=None,
                    help='cooldown for learning rate')
parser.add_argument('--accumulate', type=int, default=1,
                    help='number of batches to accumulate before gradient update')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.05,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0, #1.2e-6 TEST
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

data_provider = DataProvider(batch_size=args.batch_size,max_size=22223, data_key="20k_3s", seconds=3, v=1)
train_data, test_data = data_provider.train_loader, data_provider.test_loader
noisx_data = get_eval_dataloader("20k_3s" ,[], batch_size=1, noise_types=get_noise_types(), seconds=3)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda", flush=True)
    else:
        torch.cuda.manual_seed(args.seed)


import os

###############################################################################
# Build the model
###############################################################################

window_size = 256
N = 512
sr = 16000
sef_factor = "random"
# gama_valid = 0.5**0.5
# sef_factor = "inf"
gama_valid = "inf"
reverberation_times = [0.15, 0.175, 0.2, 0.225, 0.25]
model = model_py.SHARNN(window_size, N, 4*N, args.nlayers, args.dropouth)
simulator = RIRGenSimulator(sr, reverberation_times, device, hp_filter=True, v=3)
# simulator = PyRoomSimulator(device=device, sr=sr, reverbation_times=reverberation_times)

criterion = nn.MSELoss().to(device)
###
rms_normalizer = RMSNormalizer(skip=True)
params = list(model.parameters()) 
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args, flush=True)
print('Model total parameters:', total_params, flush=True)
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Example: StepLR
# scheduler = None
scaler = torch.cuda.amp.GradScaler()
# model = load_model(model, optimizer=optimizer, scheduler=scheduler, model_name=exp_name)
model = model.to(device)

###############################################################################
# Training code
###############################################################################

def compute_objective(signals, anti_signals, t60=None, loss_fn=None):
    if t60 is None:
        t60 = reverberation_times[random.randint(0, len(reverberation_times) - 1)]

    pt = simulator.simulate(signals.float(), t60, signal_type=NOISE_ERROR).type(signals.type())
    st = simulator.simulate(anti_signals.float(), t60 ,signal_type=ANSTISIGNAL_ERROR).type(signals.type())

    loss = criterion(pt,st) if loss_fn is None else loss_fn(pt,st)

    return loss

def eval_model_callback(model, signals, gama="inf"):
    norm_signals = rms_normalizer.normalize(signals)
    norm_anti_signals = predict(norm_signals, model)
    anti_signals = rms_normalizer.denormalize(norm_anti_signals)
    anti_signals = sef(anti_signals, gama)
    scores = compute_objective(signals=signals, anti_signals=anti_signals, t60=0.2, loss_fn=nmse)
    return scores

def valid(epoch):
    model.eval()
    scores_dict = {noise_type: [] for noise_type in get_noise_types()}
    with torch.no_grad():
        for batch, noise_types in noisx_data:
            signals = batch.to(device)

            norm_signals = rms_normalizer.normalize(signals)
            delayied_signals = delay_signals(norm_signals, delay_samples=160)
            norm_anti_signals = predict(delayied_signals, model)
            anti_signals = rms_normalizer.denormalize(norm_anti_signals)
            anti_signals = sef(anti_signals, factor=gama_valid)
            scores = compute_objective(signals=signals, anti_signals=anti_signals, t60=0.2, loss_fn=nmse)

            scores_dict[noise_types[0]].append(scores.cpu().item())

    results = {k: np.mean(v) for k, v in scores_dict.items()}
    print(f"NMSE: {results}", flush=True)
    for k, v in results.items():
        writer.add_scalar(f"NMSE/{k}", v, epoch)
    model.train()

    results.pop('factory2')
    score = np.mean(list(results.values()))
    if epoch == 0:
        return

    print(f"epoch {epoch} valid loss: {pad_num_to_len(score)}, {results}", flush=True)

def test(epoch):
    model.eval()
    samples = 0
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            signals = batch.to(device)
            if torch.all(signals == 0):
                print("all signals are 0, skipping batch", flush=True)
                continue
            
            norm_signals = rms_normalizer.normalize(signals)
            delayied_signals = delay_signals(norm_signals, delay_samples=160)
            norm_anti_signals = predict(delayied_signals, model)
            norm_anti_signals = sef(norm_anti_signals, factor=sef_factor)
            raw_loss = compute_objective(norm_signals, norm_anti_signals, loss_fn=nmse)

            test_loss += raw_loss.data * len(batch)
            samples += len(batch)

            if i % 10 == 0:
                print(f"epoch: {epoch}, batch: [{i + 1}/{len(test_data)}], epoch_loss: {pad_num_to_len(test_loss/samples)} [{pad_num_to_len(raw_loss.data)}]", flush=True)
    
    score = test_loss.item() / samples    
    print(f"epoch {epoch} test loss: {pad_num_to_len(score)}", flush=True)
    writer.add_scalars("Loss/epoch", {
        "test": score,
    }, epoch)
    model.train()
    save_model(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, model_name=exp_name,score=score)




def train():
    model.train()

    test(0)
    
    for epoch in range(40):
        epoch_train_loss = 0
        samples = 0

        for i, batch in enumerate(train_data):
            signals = batch.to(device)
            if torch.all(signals == 0):
                print("all signals are 0, skipping batch", flush=True)
                continue
            
            optimizer.zero_grad()
            
            norm_signals = rms_normalizer.normalize(signals)
            delayied_signals = delay_signals(norm_signals,delay_samples=160)
            norm_anti_signals = predict(delayied_signals, model)
            norm_anti_signals = sef(norm_anti_signals, factor=sef_factor)
            
            raw_loss = compute_objective(norm_signals, norm_anti_signals)

            scaler.scale(raw_loss).backward()
            scaler.unscale_(optimizer) # for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += raw_loss.data * len(batch)
            samples += len(batch)

            print(f"epoch: {epoch + 1}, batch: [{i + 1}/{len(train_data)}], epoch_loss: {pad_num_to_len(epoch_train_loss/samples)} [{pad_num_to_len(raw_loss.data)}]", flush=True)


        scheduler.step()

        print(f"epoch {epoch + 1} train loss: {epoch_train_loss/samples}", flush=True)
        writer.add_scalars("Loss/epoch", {
            "train": epoch_train_loss/samples,
        }, epoch + 1)
    
        test(epoch + 1)


if __name__ == "__main__":
    train()