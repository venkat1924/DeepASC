import torch
import random
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
import os
# from matplotlib import pyplot as plt
import numpy as np

from datetime import datetime
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)

from tools.simulator import RIRGenSimulator, GPUSimulator, PyRoomSimulator, NOISE_ERROR, ANSTISIGNAL_ERROR
from tools.helpers import nmse

from data_utils.data_loaders import get_dataloaders, get_eval_dataloader
from data_utils.data_tools import delay_signal, get_noise_types, sef

SCTIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCTIPT_DIR)
from compare.DeepANC.cloned.pipeline_modules import NetFeeder, Resynthesizer, normalize, extract_norm_params, denormalize, get_device
from compare.DeepANC.cloned.networks import Net
from compare.DeepANC.my_utils.helpers import save_model, load_model



from torch.utils.tensorboard import SummaryWriter

# Convert to string
now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
print("NOW STR: ", now_str)

simulators = {
    "gpu": GPUSimulator,
    "rirgen": RIRGenSimulator,
    "pra": PyRoomSimulator
}

norm = False
batch_size = 32
lr = 0.0005
decay_period = 10
decay_factor = 0.7
# decay_factor = 1
epochs = 40
clip = -1
win_len = 0.020
hop_len = 0.010
sr=16000
frames_delay = 1
simulator_type = "rirgen"
data_key = "20k_3s"
sef_factor = "random"
gama_valid = "inf"
trun = 512
model_sufix = f"{sef_factor}-sef|{frames_delay}-delay|{data_key}|{simulator_type}|ep-{epochs}|{lr}|bch-{batch_size}|dec-{decay_factor}-{decay_period}|{'norm|' if  norm else ''}"
name = f"data-{data_key}|{simulator_type}|bc-{batch_size}|lr-{lr}|"
writer = SummaryWriter(log_dir=f"logs/runs/deep_anc/{name}")


def evaluate_model_callback(model, simulator, wavfiles):
    wavfiles = wavfiles.to(get_device())
    pt = simulator.simulate(wavfiles, 0.2, signal_type=NOISE_ERROR)

    anti_signals = model.predict(wavfiles, denorm=True)[1]

    st = simulator.simulate(anti_signals, 0.2 ,signal_type=ANSTISIGNAL_ERROR)
    return nmse(pt, st)

class Model(object):
    def __init__(self, win_len=win_len, hop_len=hop_len, sr=sr):
        #=======#
        self.device = get_device()
        
        self.sr = sr
        self.win_size = int(win_len * sr)
        self.hop_size = int(hop_len * sr)

        self.feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        self.resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)
        self.net = Net().to(self.device)

        self.reverberation_times = [0.15,0.175,0.2, 0.225, 0.25]
        self.simulator = RIRGenSimulator(self.sr, self.reverberation_times, self.device, hp_filter=True, v=3)
        # self.net = DataParallel(self.net, device_ids=[gpu_id]).to(self.device)

    def forward(self, x, frames_delay=frames_delay):
        delayed_noises = delay_signal(self.hop_size, frames_delay, x)
        feat = self.feeder(delayed_noises)
        est = self.net(feat)
        anti_signals = self.resynthesizer(est, delayed_noises.shape).to(self.device) # TODO shape should be of pt.shape 
        return anti_signals

    def predict(self, noises, denorm=False, gama=sef_factor):
        delayed_noises = delay_signal(self.hop_size, frames_delay, noises)

        if norm:
            norm_params = extract_norm_params(noises)
            noises = normalize(noises, norm_params)

            d_norm_params = extract_norm_params(delayed_noises)
            delayed_noises = normalize(delayed_noises, d_norm_params)

        feat = self.feeder(delayed_noises)
        est = self.net(feat)
        anti_signals = self.resynthesizer(est, delayed_noises.shape).to(self.device) # TODO shape should be of pt.shape 

        if norm and denorm:
            anti_signals = denormalize(anti_signals, d_norm_params)
            noises = denormalize(noises, norm_params)

        anti_signals = sef(anti_signals, gama)
        return noises, anti_signals
        
    def train(self, tr_loader, ts_loader, eval_loader, optimizer, scheduler, start_epoch, max_epochs=epochs, norm=norm, clip=clip):
        criterion = torch.nn.MSELoss(reduction="mean")
        clip_norm = clip
        total_loss = 0
        total_items = 0
        best = 1

        # results = self.eval(eval_loader, nmse)
        # print(f"NMSE: {results}", flush=True)
        
        # test = self.test(ts_loader, criterion)
        writer.add_scalars("Loss/epoch", {
            # "test": test,
            "lr": optimizer.param_groups[0]['lr']
        }, start_epoch)
        # print("TEST LOSS: ", test, flush=True)
        # train model
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_loss = 0
            epoch_items = 0
            for n_iter, egs in enumerate(tr_loader): # tr_loader loads after resampling
                t60 = self.reverberation_times[random.randint(0, len(self.reverberation_times) - 1)]
                noises = egs.to(self.device)
                optimizer.zero_grad()
                
                noises, anti_signals = self.predict(noises)
                pt = self.simulator.simulate(noises, t60, signal_type=NOISE_ERROR)
                st = self.simulator.simulate(anti_signals, t60 ,signal_type=ANSTISIGNAL_ERROR)                
                loss = criterion(pt, st)

                total_items += noises.shape[0]
                epoch_items += noises.shape[0]

                loss.backward()
                if clip_norm >= 0.0:
                    clip_grad_norm_(self.net.parameters(), clip_norm)
                optimizer.step()
                # calculate loss
                running_loss = loss.item()
                epoch_loss += running_loss * noises.shape[0]
                total_loss += running_loss * noises.shape[0]

                if n_iter % 300 == 0:
                    print('Epoch [{}/{}], Iter [{}], epoch_loss = {:.8f}, total_loss = {:.8f}'.format(epoch,
                            max_epochs, n_iter, epoch_loss / epoch_items, total_loss / total_items),flush=True)

            test = self.test(ts_loader, criterion)
            writer.add_scalars("Loss/epoch", {
                "train": epoch_loss / epoch_items,
                "test": test,
                "lr": optimizer.param_groups[0]['lr']
            }, epoch)
            print("TEST LOSS: ", test, flush=True)


            if test < best:
                best = test
                save_model(self, optimizer, scheduler, epoch, f"{model_sufix}_best")
                print(f"Best model saved at epoch {epoch} with score {test}")
            
            results = self.eval(eval_loader, nmse)
            print(f"NMSE: {results}")
            for k, v in results.items():
                writer.add_scalar(f"NMSE/{k}", v, epoch)
                
            results.pop('factory2')
            score = np.mean(list(results.values()))
            print(f"NMSE: {results} with score {score}")    
            # scheduler.step() # learning rate decay

    
    def test(self, ts_loader, criterion, norm=norm):
        self.net.eval()

        total_loss = 0
        total_items = 0
        with torch.no_grad():
            for egs in ts_loader: # tr_loader loads after resampling
                t60 = self.reverberation_times[random.randint(0, len(self.reverberation_times) - 1)]
                noises = egs.to(self.device)

                noises, anti_signals = self.predict(noises)
                
                pt = self.simulator.simulate(noises, t60, signal_type=NOISE_ERROR)
                st = self.simulator.simulate(anti_signals, t60 ,signal_type=ANSTISIGNAL_ERROR)
                loss = criterion(pt, st)
                
                total_loss += loss.mean().item() * noises.shape[0]
                total_items += noises.shape[0]

        self.net.train()
        return total_loss / total_items
    
    def eval(self, eval_loader, criteria, t60=None, norm=norm):
        self.net.eval()
        scores_dict = {noise_type: [] for noise_type in get_noise_types()}
        with torch.no_grad():
            for wavfiles, noise_types in eval_loader:
                wavfiles = wavfiles.to(self.device)
                t60 = 0.2
                # t60 = self.reverberation_times[random.randint(0, len(self.reverberation_times) - 1)]
                pt = self.simulator.simulate(wavfiles, t60, signal_type=NOISE_ERROR).squeeze(0)

                anti_signals = self.predict(wavfiles, denorm=True, gama=gama_valid)[1]

                st = self.simulator.simulate(anti_signals, t60 ,signal_type=ANSTISIGNAL_ERROR).squeeze(0)
                scores = criteria(pt, st)
                for score, noise_type in zip(scores, noise_types):
                    scores_dict[noise_type].append(score.cpu())
        results = {k: np.mean(v) for k, v in scores_dict.items()}
        self.net.train()
        return results
        

def get_model(name=None):
    model = Model()
    optimizer = Adam(model.net.parameters(), lr=lr, amsgrad=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_period, gamma=decay_factor)
    scheduler = None
    if name is not None:
        return load_model(model, optimizer, scheduler,name)
    else:
        return model, optimizer, scheduler, 1
    

if __name__ == "__main__":
    # Define the model
    print(f"Process ID: ", os.getpid())
    print(f"script is starting [{model_sufix}] with cuda: {torch.cuda.is_available()}")
    print(name)
    
    tr_loader, ts_loader = get_dataloaders(key=data_key, data_files=[], batch_size=batch_size, train_ratio=0.9, seconds=3, v=1)
    
    noise_types = get_noise_types()
    eval_loader = get_eval_dataloader(key=data_key,data_files=[], batch_size=batch_size, noise_types=noise_types, seconds=3)
    model, optimizer, scheduler, start_epoch  = get_model()

    model.train(tr_loader, ts_loader, eval_loader, optimizer, scheduler, start_epoch=start_epoch)
    writer.flush()
    writer.close()

