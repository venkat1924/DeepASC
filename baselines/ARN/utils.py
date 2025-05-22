import torch
from tools.helpers import get_device
import compare.ARN.ola as ola

device = get_device()

def predict(signals, model, window_size=256):
    with torch.autocast(
        dtype=torch.float32, device_type=torch.device(device).type
    ):
        # inputs = unfold_frames(signals.unsqueeze(1))
        inputs, rest = ola.create_chuncks(signals.unsqueeze(1), window_size)
        outputs = model(inputs.squeeze(1))            
        # anti_signals = fold_frames(outputs).squeeze(1)
        anti_signals = ola.merge_chuncks(outputs.unsqueeze(1), rest).squeeze(1)
        return anti_signals