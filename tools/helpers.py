from types import SimpleNamespace
import torch
import os

from hyperpyyaml import load_hyperpyyaml

import sys
import speechbrain as sb


def nmse(pt, st):
    error = pt - st

    numerator = torch.sum(error**2,dim=-1)
    denominator = torch.sum(pt**2,dim=-1)
    
    nmse_value = 10 * torch.log10(torch.where(numerator > 0 ,numerator, torch.tensor(torch.finfo(torch.float32).eps*0.001)) / torch.where(denominator > 0,denominator, torch.tensor(torch.finfo(torch.float32).eps)))
    # nmse_value = 10 * torch.log10(torch.sum(error**2,dim=1) / torch.sum(pt**2,dim=1))
    return nmse_value

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = f"{os.path.dirname(SCRIPT_PATH)}/data_utils/models/"

os.makedirs(f"{BASE_PATH}", exist_ok=True)
os.makedirs(f"{BASE_PATH}scores", exist_ok=True)

def should_save(model_name, score):
    if os.path.exists(f"{BASE_PATH}scores/{model_name}.txt"):
        with open(f"{BASE_PATH}scores/{model_name}.txt", "r+") as f:
            best_score = float(f.read())
            if score > best_score:
                return False
            else:
                f.seek(0)
                f.write(str(score))
    else:
        with open(f"{BASE_PATH}scores/{model_name}.txt", "w") as f:
            f.write(str(score))
    print(f"should save {model_name} with score {score}")
    return True

def save_model(model, optimizer, scheduler, epoch, model_name, score):
    if not should_save(model_name, score):
        return
    
    data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    path = f"{BASE_PATH}{model_name}"

    if scheduler is not None:
        try:
            scheduler.save(f"{path}_scheduler.pt")
        except:
            data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(data, f"{path}.pt")
    return

def remove_model(model_name):
    if os.path.exists(f"{BASE_PATH}{model_name}.pt"):
        os.remove(f"{BASE_PATH}{model_name}.pt")
    
    if os.path.exists(f"{BASE_PATH}{model_name}_scheduler.pt"):
        os.remove(f"{BASE_PATH}{model_name}_scheduler.pt")
    return


def load_model(model, model_name, optimizer=None, scheduler=None, strict=True):
    data = torch.load(f"{BASE_PATH}{model_name}.pt")

    unexpted_keys, missing_keys = model.load_state_dict(data['model_state_dict'],strict=strict)
    if not strict:
        print("Unexpected Keys:",unexpted_keys, "Missing Keys", missing_keys)
        
    epoch = data['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state_dict'])

    if scheduler is not None:
        if os.path.exists(f"{BASE_PATH}{model_name}_scheduler.pt"):
            scheduler_path = f"{BASE_PATH}{model_name}_scheduler.pt"
            scheduler.load(scheduler_path)
        else:
            scheduler.load_state_dict(data['scheduler_state_dict'])
    return model, optimizer, scheduler, epoch

def load_combined_model(s_inv_model,p_model, optimizer, scheduler, model_name):
    data = torch.load(f"{BASE_PATH}{model_name}.pt")

    s_inv_model.load_state_dict(data['s_inv_model_state_dict'])
    p_model.load_state_dict(data['p_model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    epoch = data['epoch']

    scheduler_path = f"{BASE_PATH}{model_name}_scheduler.pt"
    scheduler.load(scheduler_path)
    return s_inv_model, p_model, optimizer, scheduler, epoch 
    
    

def save_combined_model(s_inv_model, p_model, optimizer, scheduler, epoch, model_name, score):
    if not should_save(model_name, score):
        return

    data = {
        'epoch': epoch,
        's_inv_model_state_dict': s_inv_model.state_dict(),
        'p_model_state_dict': p_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    path = f"{BASE_PATH}{model_name}"
    scheduler.save(f"{path}_scheduler.pt")
    torch.save(data, f"{path}.pt")
    return


def save_combined_a_model(s_inv_model, p_model, s_inv_optimizer, p_optimizer, s_inv_scheduler, p_scheduler, epoch, model_name, score):
    if not should_save(model_name, score):
        return
    
    path = f"{BASE_PATH}{model_name}"
    
    data = {
        'epoch': epoch,
        's_inv_model_state_dict': s_inv_model.state_dict(),
        'p_model_state_dict': p_model.state_dict(),
        's_inv_optimizer_state_dict': s_inv_optimizer.state_dict(),
        'p_optimizer_state_dict': p_optimizer.state_dict(),
    }
    torch.save(data, f"{path}.pt")


    s_inv_scheduler.save(f"{path}_s_inv_scheduler.pt")
    p_scheduler.save(f"{path}_p_scheduler.pt")
    return

def remove_combined_a_model(model_name):
    if os.path.exists(f"{BASE_PATH}{model_name}.pt"):
        os.remove(f"{BASE_PATH}{model_name}.pt")
    
    if os.path.exists(f"{BASE_PATH}{model_name}_s_inv_scheduler.pt"):
        os.remove(f"{BASE_PATH}{model_name}_s_inv_scheduler.pt")
    
    if os.path.exists(f"{BASE_PATH}{model_name}_p_scheduler.pt"):
        os.remove(f"{BASE_PATH}{model_name}_p_scheduler.pt")
    return

def load_combined_a_model(s_inv_model, p_model, s_inv_optimizer, p_optimizer, s_inv_scheduler, p_scheduler, model_name):
    data = torch.load(f"{BASE_PATH}{model_name}.pt")

    s_inv_model.load_state_dict(data['s_inv_model_state_dict'])
    p_model.load_state_dict(data['p_model_state_dict'])
    s_inv_optimizer.load_state_dict(data['s_inv_optimizer_state_dict'])
    p_optimizer.load_state_dict(data['p_optimizer_state_dict'])
    epoch = data['epoch']


    s_inv_scheduler_path = f"{BASE_PATH}{model_name}_s_inv_scheduler.pt"
    s_inv_scheduler.load(s_inv_scheduler_path)
    p_inv_scheduler_path = f"{BASE_PATH}{model_name}_p_scheduler.pt"
    p_scheduler.load(p_inv_scheduler_path)
    return s_inv_model, p_model, s_inv_optimizer, p_optimizer, s_inv_scheduler, p_scheduler, epoch

def get_device(id=None):
    gpu_id = id
    device = "cuda" if id is None else f"cuda:{gpu_id}"
    return device




small_val = torch.finfo(torch.float32).eps  # To avoid divide by zero

def si_snr_denorm(pt, st):
    dot = torch.sum(st * pt, dim=1, keepdim=True) + small_val
    st_target_energy = torch.sum(st**2, dim=1, keepdim=True) 
    st_rescaled = st * st_target_energy / dot
    return st_rescaled


class RMSNormalizer:
    last_rms = None   
    def __init__(self, skip=False):
        self.skip = skip

    def extract_rms(self, tensor):
        rms = torch.sqrt(torch.mean(tensor**2)).to(get_device())
        self.last_rms = rms

    def normalize(self, tensor):
        if self.skip:
            return tensor
        
        self.extract_rms(tensor)
        # return tensor / (rms + 1e-7), lambda x: x * (rms + 1e-7)
        return tensor / (self.last_rms + 1e-7)

    def denormalize(self, tensor):
        if self.skip:
            return tensor
        
        return tensor * (self.last_rms + 1e-7)


def get_hparams(yaml_path):
    hparams_file, _, overrides = sb.parse_arguments([yaml_path])
    with open(hparams_file) as fin:
        hparams = SimpleNamespace(**load_hyperpyyaml(fin,overrides))
    return hparams


def get_fxlms_mu(noise):
    return  {
        "Engine": 0.05,
        "Babble": 0.15,
        "Speech": 0.2,
        "Factory": 0.2
    }[noise]

def get_fxlms_mu_(noise):
    return { 
        "destroyerengine": 0.05,
        "babble": 0.3,
        "speech": 0.2,
        "factory1": 0.2,
        "factory2": 0.2,
    }[noise]


def align_batchs(batch1, batch2):
    batch1_len = batch1.shape[0]
    batch2_len = batch2.shape[0]
    if batch1_len > batch2_len:
        return batch1[:batch2_len,:], batch2
    elif batch2_len > batch1_len:
        return batch1, batch2[:batch1_len,:]
    return batch1, batch2

def delay_signals(signals, delay_samples=160, delay_right=False):
    if delay_samples == 0:
        return signals
    
    sliced_signal = signals[:,:-delay_samples]
    if delay_right:
        padding_shape = (0, delay_samples)
    else:
        padding_shape = (delay_samples, 0)
    padded_tensor = torch.nn.functional.pad(sliced_signal, padding_shape, mode='constant', value=0)
    return padded_tensor
