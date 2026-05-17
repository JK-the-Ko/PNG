# Shared CNN prompt blocks used by the prompt autoencoder and related models.
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from png.archs.norm import LayerNorm
from png.utils.noise import extract_patches, corrcoef

# CNN-based Noise Prompt Block
class LocalCorrelationBlock(nn.Module) :
    def __init__(self, kernel_local, prompt_height, prompt_width, out_channels) :
        # Inheritance
        super().__init__()
        
        # Initialize Variables
        self.kernel = kernel_local
        self.height = prompt_height
        self.width = prompt_width
        
        # Compute Feature Channels
        in_channels = prompt_height+prompt_width
        emb_channels = 2*in_channels
        
        # Create Convolutional Layer Instance
        self.conv_1x1 = nn.Sequential(*[nn.Conv2d(in_channels, emb_channels, kernel_size=1, bias=False),
                                       nn.GELU(),
                                       nn.Conv2d(emb_channels, emb_channels, kernel_size=1, bias=False)])
        self.conv_3x3 = nn.Conv2d(self.kernel*self.kernel, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        
        assert self.height == self.width
        
    def forward(self, noise) :
        B, _, H, W = noise.size()
        
        patches = extract_patches(noise, ksizes=(self.kernel, self.kernel), strides=(1,1), rates=(1,1), padding="same") # [N, C, k*k, L]
        input = patches.permute(0,3,2,1) # [B, L, HW, C]

        coef = corrcoef(input, clip=False)[:,:,(self.kernel*self.kernel)//2,:]
        coef = coef.reshape(B,H,W,self.kernel**2).permute(0, 3, 1, 2) # [B, P, H ,W]
        coef_x, coef_y = coef.mean(dim=(2,)), coef.mean(dim=(3,)) # [B, P, W] & [B, P, H]
        coef_xy = torch.concat([coef_x, coef_y], dim=2) # [B, P, H+W]
        coef_xy = rearrange(coef_xy, 'B (Ph Pw) HpW -> B HpW Ph Pw', Ph=self.kernel, Pw=self.kernel) # [B, H+W, Ph, Pw]
        
        coef_emb = self.conv_1x1(coef_xy) # [B, H*W, Ph, Pw]
        out_shape = int(self.height**0.5) * 2
        coef_emb = rearrange(coef_emb, 'B (H W) Ph Pw -> B (Ph Pw) H W', H=out_shape, W=out_shape) # [B PhPw H W]
        coef_emb = F.interpolate(coef_emb, size=noise.size()[2:], mode="bilinear")
        coef_emb = self.conv_3x3(coef_emb)
        
        return coef_emb


class LocalPromptBlock(nn.Module) :
    def __init__(self, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
        
        # Create Prompt Component Instance
        self.prompt_local = nn.Parameter(torch.zeros(1, prompt_channels, prompt_height, prompt_width))
        
        # Local Block Instance
        self.local_corr = LocalCorrelationBlock(kernel_local, prompt_height, prompt_width, prompt_channels)
        
        # Create Convolutional Layer Instance
        self.conv_local = nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        
    def forward(self, noise) :
        # Extract Local Prompt Feature
        feat_local = self.local_corr(noise)
        prompt_local = F.softmax(feat_local, dim=1)*self.prompt_local    
        prompt_local = self.conv_local(prompt_local)
        
        return prompt_local


class JointPromptBlock(nn.Module) :
    def __init__(self, in_channels, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
        
        # Create Prompt Component Instance
        self.prompt_local = nn.Parameter(torch.zeros(1, prompt_channels, prompt_height, prompt_width))
        self.prompt_global = nn.Parameter(torch.zeros(1, prompt_channels, prompt_height, prompt_width))
        
        # Local Block Instance
        self.local_corr = LocalCorrelationBlock(kernel_local, prompt_height, prompt_width, prompt_channels)
        
        # Create Convolutional Layer Instance
        self.conv_local = nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_global_1 = nn.Conv2d(in_channels*2, prompt_channels, kernel_size=1, bias=False)
        self.conv_global_2 = nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        
    def forward(self, input, noise) :
        # Extract Local Prompt Feature
        feat_local = self.local_corr(noise)
        prompt_local = F.softmax(feat_local, dim=1)*self.prompt_local    
        prompt_local = self.conv_local(prompt_local)
        
        # Extract Global Prompt Feature
        avg = torch.mean(input, dim=[2,3]).unsqueeze(-1).unsqueeze(-1)
        std = torch.std(input, dim=[2,3]).unsqueeze(-1).unsqueeze(-1)
        
        feat_global = self.conv_global_1(torch.cat([avg, std], dim=1))
        prompt_global = F.softmax(feat_global, dim=1)*self.prompt_global
        prompt_global = self.conv_global_2(prompt_global)
        
        return prompt_local, prompt_global


class GlobalPromptBlock(nn.Module) :
    def __init__(self, in_channels, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
        
        # Create Prompt Component Instance
        self.prompt_global = nn.Parameter(torch.zeros(1, prompt_channels, prompt_height, prompt_width))
        
        # Create Convolutional Layer Instance
        self.conv_global_1 = nn.Conv2d(in_channels*2, prompt_channels, kernel_size=1, bias=False)
        self.conv_global_2 = nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        
    def forward(self, input) :
        # Extract Global Prompt Feature
        avg = torch.mean(input, dim=[2,3]).unsqueeze(-1).unsqueeze(-1)
        std = torch.std(input, dim=[2,3]).unsqueeze(-1).unsqueeze(-1)
        
        feat_global = self.conv_global_1(torch.cat([avg, std], dim=1))
        prompt_global = F.softmax(feat_global, dim=1)*self.prompt_global        
        prompt_global = self.conv_global_2(prompt_global)

        return prompt_global


 
# Ordinary CNN-based Residual Block
class ResidualBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Create Normalization Layer Instance
        self.norm_1 = LayerNorm(in_channels, "WithBias")
        self.norm_2 = LayerNorm(out_channels, "WithBias")
        
        # Create Activation Layer Instance
        self.act = nn.GELU()

    def forward(self, input) :
        output = self.conv_1(self.act(self.norm_1(input)))
        output = self.conv_2(self.act(self.norm_2(output)))    

        return output + self.conv_out(input)


class ResidualLocalPromptBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
    
        # Create Prompt Block Instance
        self.local_pb = LocalPromptBlock(kernel_local, prompt_channels, prompt_height, prompt_width)
        
        # Create Convolutional Layer Instance
        self.conv_1 = nn.Conv2d(in_channels+prompt_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_out = nn.Conv2d(in_channels+prompt_channels, out_channels, kernel_size=1, bias=False)
        
        # Create Normalization Layer Instance
        self.norm_1 = LayerNorm(in_channels+prompt_channels, "WithBias")
        self.norm_2 = LayerNorm(out_channels, "WithBias")
        
        # Create Activation Layer Instance
        self.act = nn.GELU()

    def forward(self, input, noise) :
        prompt_local = self.local_pb(noise)

        input = torch.cat([input, prompt_local], dim=1)
        output = self.conv_1(self.act(self.norm_1(input)))
        output = self.conv_2(self.act(self.norm_2(output)))

        return output + self.conv_out(input), prompt_local


class ResidualJointPromptBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
    
        # Create Prompt Block Instance
        self.joint_pb = JointPromptBlock(in_channels, kernel_local, prompt_channels, prompt_height, prompt_width)
        
        # Create Convolutional Layer Instance
        self.conv_1 = nn.Conv2d(in_channels+prompt_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_out = nn.Conv2d(in_channels+prompt_channels*2, out_channels, kernel_size=1, bias=False)
        
        # Create Normalization Layer Instance
        self.norm_1 = LayerNorm(in_channels+prompt_channels*2, "WithBias")
        self.norm_2 = LayerNorm(out_channels, "WithBias")
        
        # Create Activation Layer Instance
        self.act = nn.GELU()

    def forward(self, input, noise) :
        prompt_local, prompt_global = self.joint_pb(input, noise)
        
        input = torch.cat([input, prompt_local, prompt_global], dim=1)
        output = self.conv_1(self.act(self.norm_1(input)))
        output = self.conv_2(self.act(self.norm_2(output)))

        return output + self.conv_out(input), prompt_local, prompt_global


class ResidualGlobalPromptBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
    
        # Create Prompt Block Instance
        self.global_pb = GlobalPromptBlock(in_channels, prompt_channels, prompt_height, prompt_width)
        
        # Create Convolutional Layer Instance
        self.conv_1 = nn.Conv2d(in_channels+prompt_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.conv_out = nn.Conv2d(in_channels+prompt_channels, out_channels, kernel_size=1, bias=False)
        
        # Create Normalization Layer Instance
        self.norm_1 = LayerNorm(in_channels+prompt_channels, "WithBias")
        self.norm_2 = LayerNorm(out_channels, "WithBias")
        
        # Create Activation Layer Instance
        self.act = nn.GELU()

    def forward(self, input) :
        prompt_global = self.global_pb(input)
        
        input = torch.cat([input, prompt_global], dim=1)
        output = self.conv_1(self.act(self.norm_1(input)))
        output = self.conv_2(self.act(self.norm_2(output)))

        return output + self.conv_out(input), prompt_global


 
# Ordinary CNN-based Residual Module
class LocalPromptModule(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.res_local_pb = ResidualLocalPromptBlock(out_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, input, noise) :
        output = self.conv_in(input)
        output, prompt_local = self.res_local_pb(output, noise)
        output = self.res_block(output)
        
        return output, [prompt_local]


class JointPromptModule(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.res_joint_pb = ResidualJointPromptBlock(out_channels, out_channels, kernel_local, prompt_channels, prompt_height, prompt_width)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, input, noise) :
        output = self.conv_in(input)
        output, prompt_local, prompt_global = self.res_joint_pb(output, noise)
        output = self.res_block(output)
        
        return output, [prompt_local, prompt_global]


class GlobalPromptModule(nn.Module) :
    def __init__(self, in_channels, out_channels, prompt_channels, prompt_height, prompt_width, is_down) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if is_down else 1, padding=1, padding_mode="reflect")
        self.res_global_pb = ResidualGlobalPromptBlock(out_channels, out_channels, prompt_channels, prompt_height, prompt_width)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, input) :
        output = self.conv_in(input)
        output, prompt_global = self.res_global_pb(output)
        output = self.res_block(output)
        
        return output, [prompt_global]
    

class ResidualBlockModule(nn.Module) :
    def __init__(self, in_channels, out_channels, is_down) :
        # Inheritance
        super().__init__()
        
        # Create Convolutional Layer Instance
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if is_down else 1, padding=1, padding_mode="reflect")
        self.res_block_1 = ResidualBlock(out_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)

    def forward(self, input) :
        output = self.conv_in(input)
        output = self.res_block_1(output)
        output = self.res_block_2(output)
        
        return output


 
# Ordinary CNN-based Upsampling Module
class Upsample(nn.Module) :
    def __init__(self, in_channels, out_channels, is_up, scale) :
        # Inheritance
        super().__init__()
        
        # Initialize Variable
        self.is_up = is_up
        self.scale = scale
        
        # Create Convolutional Layer Instance
        self.res_block_1 = ResidualBlock(in_channels, in_channels)
        self.res_block_2 = ResidualBlock(in_channels, in_channels)
        self.conv_clean = nn.Conv2d(3*scale**2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, input, clean) :
        if self.is_up :
            clean = F.pixel_unshuffle(clean, self.scale)
        clean = self.conv_clean(clean)
        
        output = self.res_block_1(input+clean)
        output = self.res_block_2(output+clean)
        
        if self.is_up :
            output = F.interpolate(output, scale_factor=2, mode="nearest")
            
        output = self.conv_out(output)

        return output
