import copy
import torch

from scipy.signal import firwin
from tools.helpers import get_device
from tools.simulator import _simulate_v2
# try:
# from DeepASC.quant_modules_.mamba_blocks import MambaBlocksSequential
# except ImportError:
from DeepASC.modules.mamba_blocks import MambaBlocksSequential
    
#     from DeepASC.modules.mamba_blocks import MambaBlocksSequential
    
# from DeepASC._quant_modules.mamba_blocks import MambaBlocksSequential as QuantMambaBlocksSequential
from speechbrain.lobes.models.dual_path import Encoder, Dual_Path_Model
import speechbrain as sb
device = get_device()
sr = 16000

class MambaEncoder(torch.nn.Module):
    def __init__(self, encoder, mamba_block, is_mask=True):
        super().__init__()
        self.encoder = encoder
        self.masknet = mamba_block
        self.is_mask = is_mask
        
    
    def forward(self, x):
        embed = self.encoder(x)
        x = self.masknet(embed)
        
        # if self.squeeze:
        #     return (embed * x).squeeze(0)
        if self.is_mask:
            res = embed * x
        else:
            res = x
        return res

class DeepASC(torch.nn.Module):
    def __init__(self, hparams):
        super(DeepASC, self).__init__()
        self.hparams = hparams
        self.cutoffs = hparams.cutoffs
        self.nbands = len(self.cutoffs)
        self.device = device
        self.filters = self.get_filters()

        self.dense_conv = torch.nn.Conv2d(in_channels=self.nbands, out_channels=1, kernel_size=1, stride=1, padding=0)
        if hparams.smart_init:
            torch.nn.init.constant_(self.dense_conv.weight, 1)
            torch.nn.init.constant_(self.dense_conv.bias, 0)

        self.encoder_mambas = self.get_mamba_blocks()
        self.decoder = hparams.decoder
        self.reinitialize_layers()

    def __reset_module_params(self, module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        for child_layer in module.modules():
            if module != child_layer:
                self.__reset_module_params(child_layer)


    def reinitialize_layers(self):
        self.__reset_module_params(self.dense_conv)
        self.__reset_module_params(self.encoder_mambas)
        self.__reset_module_params(self.decoder)

    def get_filters(self):
        order = 256

        filters = []
        for i in range(self.nbands):
            start, end = self.cutoffs[i]

            f = firwin(order, [start, end], fs=sr, pass_zero='bandpass')
            
            filters.append(torch.tensor(f, requires_grad=False).reshape(1,1,-1).to(torch.float32).to(self.device))
        
        return filters
    

    def get_mamba_blocks(self):
        encoder_mambas = []
        
        # mamba_sequntial = QuantMambaBlocksSequential if getattr(self.hparams, "quantize", False) else MambaBlocksSequential
        mamba_sequntial = MambaBlocksSequential
        for i in range(self.nbands):
            mamba_intra = mamba_sequntial(
                n_mamba=self.hparams.n_mamba_dp // 2,
                bidirectional=self.hparams.bidirectional,
                d_model=self.hparams.out_channels,
                d_state=self.hparams.ssm_dim,
                expand=self.hparams.mamba_expand,
                d_conv=self.hparams.mamba_conv,
                fused_add_norm=self.hparams.fused_add_norm,
                rms_norm=self.hparams.rms_norm,
                residual_in_fp32=self.hparams.residual_in_fp32
            )

            mamba_inter = mamba_sequntial(
                n_mamba=self.hparams.n_mamba_dp // 2,
                bidirectional=self.hparams.bidirectional,
                d_model=self.hparams.out_channels,
                d_state=self.hparams.ssm_dim,
                expand=self.hparams.mamba_expand,
                d_conv=self.hparams.mamba_conv,
                fused_add_norm=self.hparams.fused_add_norm,
                rms_norm=self.hparams.rms_norm,
                residual_in_fp32=self.hparams.residual_in_fp32
            )
            is_medium = i == self.nbands - 1
            mamba_encoder = MambaEncoder(
                    encoder=Encoder(
                        kernel_size=self.hparams.kernel_size,
                        out_channels=self.hparams.N_encoder_out
                    ),
                    mamba_block=Dual_Path_Model(
                        num_spks=self.hparams.num_spks,
                        in_channels=self.hparams.N_encoder_out,
                        out_channels=self.hparams.out_channels,
                        num_layers=self.hparams.n_dp_encoder * (2 if is_medium else 1),
                        K=self.hparams.chunk_size,
                        intra_model=mamba_intra,
                        inter_model=mamba_inter,
                        norm='ln',
                        linear_layer_after_inter_intra=False,
                        skip_around_intra=is_medium,
                        is_dual=getattr(self.hparams, "is_dual", True),
                    ),
                    is_mask=getattr(self.hparams, "is_mask", True)
                ).to(self.device)
            encoder_mambas.append(mamba_encoder)

        encoder_mambas = torch.nn.ModuleList(encoder_mambas)
        return encoder_mambas


    def seperate_to_bands(self, signals):
        signal_bands = []

        for f in self.filters:
            signal_band =  _simulate_v2(signals, f, self.device, padding="same")
            signal_bands.append(signal_band)

        return signal_bands


    def dense_layer(self, encoded_bands):
        x = torch.cat(encoded_bands, dim=1)        
        x = self.dense_conv(x).squeeze(1)
        return x


    def forward(self, signals):
        bands = self.seperate_to_bands(signals)
        encoded_bands = []

        for i, encoder in enumerate(self.encoder_mambas):
            encoded_bands.append(encoder(bands[i]).permute(1,0,2,3))

        dense_encode = self.dense_layer(encoded_bands)
        x = dense_encode
        return self.decoder(x)


class SBDeepASC(torch.nn.Module):
    def __init__(self, hparams):
        super(SBDeepASC, self).__init__()
        self.hparams = hparams
        self.encoder = hparams.encoder
        self.masknet = hparams.masknet
        self.decoder = hparams.decoder
        self.reinitialize_layers()

    def __reset_module_params(self, module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        for child_layer in module.modules():
            if module != child_layer:
                self.__reset_module_params(child_layer)


    def reinitialize_layers(self):
        self.__reset_module_params(self.encoder)
        self.__reset_module_params(self.masknet)
        self.__reset_module_params(self.decoder)


    def forward(self, inputs):
        mix_w = self.encoder(inputs)

        est_mask = self.masknet(mix_w)

        sep_h = mix_w * est_mask

        est_source = self.decoder(sep_h.squeeze(0))
        return est_source



def get_model(hparams, single_band=False):
    if single_band:
        return SBDeepASC(hparams).to(device)
    else:
        return DeepASC(hparams).to(device)
    
def connvet_masknet(hparams, masknet, net_block, is_medium, num_layers=None):
    masknet = Dual_Path_Model(
        num_spks=hparams.num_spks,
        in_channels=hparams.N_encoder_out,
        out_channels=hparams.out_channels,
        num_layers=num_layers if num_layers is not None else (6 * (2 if is_medium else 1)),
        K=hparams.chunk_size,
        intra_model=copy.deepcopy(net_block),
        inter_model=copy.deepcopy(net_block),
        norm='ln',
        linear_layer_after_inter_intra=False,
        skip_around_intra=is_medium
    ).to(device)
        
    # masknet.dual_mdl = torch.nn.ModuleList(layers)
    return masknet

def get_ablation_net_block(block_type, size):
    if block_type == "transformer":
        return torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=1024,
                dropout=0,
                norm_first= True,
                # bias: bool = True,
                # device=None,
                # dtype=None,
            ),
            num_layers=size,
        )                
        return sb.lobes.models.dual_path.SBTransformerBlock(
            num_layers=size,
            d_model=256,
            nhead=4,
            d_ffn=1024,
            dropout=0,
            norm_before=True,
            use_positional_encoding=True
        ).to(device)

    elif block_type == "lstm":
        class LSTMWrapper(torch.nn.Module):
            def __init__(self, size):
                super(LSTMWrapper, self).__init__()
                self.model = torch.nn.LSTM(
                    input_size=256,
                    hidden_size=128,
                    num_layers=size,
                    batch_first=True,
                    bidirectional=True
                ).to(device)

            def forward(self, x):
                x, _ = self.model(x)
                return x
            
        return LSTMWrapper(size)

    return None
    
def get_ablation_model(hparams, block_type, block_size, single_band=False, single_band_medium=False, num_layers=None):
    net_block = get_ablation_net_block(block_type, block_size)
    model = get_model(hparams, single_band)

    if single_band:
        model.masknet = connvet_masknet(hparams, model.masknet, net_block, is_medium=single_band_medium)
    else:
        for i, encoder_mamba in enumerate(model.encoder_mambas):
            is_medium = i == len(model.encoder_mambas) - 1
            encoder_mamba.masknet = connvet_masknet(hparams, encoder_mamba.masknet, net_block, is_medium=is_medium, num_layers=num_layers)
    
    return model