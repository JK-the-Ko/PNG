# Prompt autoencoder encoder and decoder definitions.
from torch import nn

from png.archs.arch import *

# CNN-based Prompt-Encoder
class PromptEncoder(nn.Module) :
    def __init__(self, input_dim, channels_g, kernel_local, channels_p, prompt_size) :
        # Inheritance
        super().__init__()
        
        # Create Encoder Layer Instances
        self.joint_pm = JointPromptModule(input_dim, channels_g, kernel_local, channels_p, prompt_size, prompt_size)
        self.global_pm_1 = GlobalPromptModule(channels_g, channels_g, channels_p, prompt_size//2, prompt_size//2, True)
        self.global_pm_2 = GlobalPromptModule(channels_g, channels_g*2, channels_p*2, prompt_size//4, prompt_size//4, True)
        self.global_pm_3 = GlobalPromptModule(channels_g*2, channels_g*4, channels_p*4, prompt_size//8, prompt_size//8, True)

    def forward(self, noise) :
        out_0, p_set0 = self.joint_pm(noise, noise)
        out_1, p_set1 = self.global_pm_1(out_0)
        out_2, p_set2 = self.global_pm_2(out_1)
        out_3, p_set3 = self.global_pm_3(out_2)
        
        return [out_0, p_set0], [out_1, p_set1], [out_2, p_set2], [out_3, p_set3]


class GlobalPromptEncoder(nn.Module) :
    def __init__(self, input_dim, channels_g, channels_p, prompt_size) :
        # Inheritance
        super().__init__()
        
        # Create Encoder Layer Instances
        self.global_pm_0 = GlobalPromptModule(input_dim, channels_g, channels_p, prompt_size, prompt_size, False)
        self.global_pm_1 = GlobalPromptModule(channels_g, channels_g, channels_p, prompt_size//2, prompt_size//2, True)
        self.global_pm_2 = GlobalPromptModule(channels_g, channels_g*2, channels_p*2, prompt_size//4, prompt_size//4, True)
        self.global_pm_3 = GlobalPromptModule(channels_g*2, channels_g*4, channels_p*4, prompt_size//8, prompt_size//8, True)

    def forward(self, noise) :
        out_0, p_set0 = self.global_pm_0(noise)
        out_1, p_set1 = self.global_pm_1(out_0)
        out_2, p_set2 = self.global_pm_2(out_1)
        out_3, p_set3 = self.global_pm_3(out_2)
        
        return [out_0, p_set0], [out_1, p_set1], [out_2, p_set2], [out_3, p_set3]


class LocalPromptEncoder(nn.Module) :
    def __init__(self, input_dim, channels_g, kernel_local, channels_p, prompt_size) :
        # Inheritance
        super().__init__()
        
        # Create Encoder Layer Instances
        self.local_pm = LocalPromptModule(input_dim, channels_g, kernel_local, channels_p, prompt_size, prompt_size)
        self.res_module_1 = ResidualBlockModule(channels_g, channels_g, True)
        self.res_module_2 = ResidualBlockModule(channels_g, channels_g*2, True)
        self.res_module_3 = ResidualBlockModule(channels_g*2, channels_g*4, True)
        
    def forward(self, noise) :
        out_0, p_set0 = self.local_pm(noise, noise)
        out_1 = self.res_module_1(out_0)
        out_2 = self.res_module_2(out_1)
        out_3 = self.res_module_3(out_2)
        
        return [out_0, p_set0], [out_1], [out_2], [out_3]


class VanillaEncoder(nn.Module) :
    def __init__(self, input_dim, channels_g) :
        # Inheritance
        super().__init__()
        
        # Create Encoder Layer Instances
        self.res_module_0 = ResidualBlockModule(input_dim, channels_g, False)
        self.res_module_1 = ResidualBlockModule(channels_g, channels_g, True)
        self.res_module_2 = ResidualBlockModule(channels_g, channels_g*2, True)
        self.res_module_3 = ResidualBlockModule(channels_g*2, channels_g*4, True)
        
    def forward(self, noise) :
        out_0 = self.res_module_0(noise)
        out_1 = self.res_module_1(out_0)
        out_2 = self.res_module_2(out_1)
        out_3 = self.res_module_3(out_2)
        
        return [out_0], [out_1], [out_2], [out_3]


 
# CNN-based Decoder
class Decoder(nn.Module) :
    def __init__(self, input_dim, channels_g) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instances
        self.upsample_1 = Upsample(channels_g*4, channels_g*2, is_up=True, scale=8)
        self.upsample_2 = Upsample(channels_g*2, channels_g, is_up=True, scale=4)
        self.upsample_3 = Upsample(channels_g, channels_g, is_up=True, scale=2)
        self.upsample_4 = Upsample(channels_g, input_dim, is_up=False, scale=1)

    def forward(self, input, clean) :
        out_0 = self.upsample_1(input, clean)
        out_1 = self.upsample_2(out_0, clean)
        out_2 = self.upsample_3(out_1, clean)
        out_3 = self.upsample_4(out_2, clean)
        
        return out_3 + clean
