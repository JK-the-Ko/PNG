# Prompt-feature classifiers built on top of prompt autoencoder activations.
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from png.archs.arch import LayerNorm

def _get_prompt_ae_channels(prompt_ae_config_path):
    config = OmegaConf.load(prompt_ae_config_path)
    return config.params.encoder_config.params.channels_p


def _get_unshuffle_multiplier(downscale_factor):
    return downscale_factor ** 2


# Ordinary CNN-based Residual Module
class ResidualBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Create Normalization Layer Instance
        self.norm_1 = LayerNorm(in_channels, "With-Bias")
        self.norm_2 = LayerNorm(out_channels, "With-Bias")
        
        # Create Activation Layer Instance
        self.act = nn.GELU()

    def forward(self, input) :
        output = self.conv_1(self.act(self.norm_1(input)))
        output = self.conv_2(self.act(self.norm_2(output)))    

        return output + self.conv_out(input)
    

class ResidualBlockModule(nn.Module) :
    def __init__(self, in_channels, out_channels, is_down) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if is_down else 1, padding=1, bias=False)
        self.res_block_1 = ResidualBlock(out_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)

    def forward(self, input) :
        output = self.conv_in(input)
        output = self.res_block_1(output)
        output = self.res_block_2(output)
        
        return output

 
# CNN-based Prompt Autoencoder
class PromptClassifier(nn.Module) :
    def __init__(self, prompt_type, channels_g, num_class, prompt_ae_config_path=None) :
        # Inheritance
        super().__init__()
        
        # Initialize Variables
        self.prompt_type = prompt_type
        self.scale0_unshuffle = 8
        self.scale1_unshuffle = 4
        self.scale2_unshuffle = 2
        if prompt_ae_config_path is None:
            prompt_ae_config_path = "configs/models/prompt_ae/prompt_ae.yaml"
        prompt_channels = _get_prompt_ae_channels(prompt_ae_config_path)
        
        # Compute Input Prompt Feature Map Channels
        c_p0 = prompt_channels * _get_unshuffle_multiplier(self.scale0_unshuffle) * 2
        c_p1 = prompt_channels * _get_unshuffle_multiplier(self.scale1_unshuffle)
        c_p2 = prompt_channels * _get_unshuffle_multiplier(self.scale2_unshuffle) * 2
        c_p3 = prompt_channels * _get_unshuffle_multiplier(self.scale2_unshuffle)
        input_dim = c_p0 + c_p1 + c_p2 + c_p3
        
        # Create Residual Block Layer Instance
        self.res_block_0 = nn.Conv2d(input_dim, channels_g, kernel_size=1, stride=2, padding=1)
        self.res_block_1 = ResidualBlockModule(channels_g, channels_g, True)
        self.res_block_2 = ResidualBlockModule(channels_g, channels_g, True)
        self.res_block_3 = ResidualBlockModule(channels_g, channels_g, True)
        self.res_block_4 = ResidualBlockModule(channels_g, channels_g, False)
        
        # Create Linear Layer Instance
        self.linear = nn.Linear(channels_g, num_class)
        
    def forward(self, scale0, scale1, scale2, scale3) :
        # Apply Pixel-Unshuffling Process
        scale0 = F.pixel_unshuffle(scale0, self.scale0_unshuffle)
        scale1 = F.pixel_unshuffle(scale1, self.scale1_unshuffle)
        scale2 = F.pixel_unshuffle(scale2, self.scale2_unshuffle)
        
        # Concatenate Prompt Features
        input = torch.cat([scale0, scale1, scale2, scale3], dim=1)
        
        # Feed-Forward CNN
        output = self.res_block_0(input)
        output = self.res_block_1(output)
        output = self.res_block_2(output)
        output = self.res_block_3(output)
        output = self.res_block_4(output)

        # Feed-Forward MLP
        output = torch.flatten(F.adaptive_avg_pool2d(output, (1,1)), start_dim=1)
        output = self.linear(output)
        
        return output
