# Lightning module for prompt-conditioned consistency modeling.
from copy import deepcopy
from typing import Mapping, Any, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from piq import LPIPS
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from png.models.loggers import TensorBoardLogger, WandbLogger, LocalImageLogger
from png.utils.common import instantiate_from_config, instantiate_from_config_with_arg,  get_obj_from_str
from png.utils.metrics import calculate_kld, calculate_akld
from png.utils.misc import const_like

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`."""
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

    
class LitPromptCM(pl.LightningModule):
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        **kwargs,
        ):
        if "ae_config" not in kwargs and hparams_file is None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", {})
            has_embedded_ae = (
                any(key.startswith("encoder.") for key in state_dict)
                and any(key.startswith("decoder.") for key in state_dict)
            )
            ae_config = checkpoint.get("hyper_parameters", {}).get("ae_config")
            if has_embedded_ae and ae_config is not None and ae_config.get("ckpt_path", None):
                ae_config = deepcopy(ae_config)
                ae_config.pop("ckpt_path", None)
                kwargs["ae_config"] = ae_config

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )

    def __init__(
        self,
        cm_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] = None,
        misc_config: Mapping[str, Any] = None,
        ):
        super().__init__()

        self.misc_config = misc_config
        self.model = instantiate_from_config(model_config)
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.requires_grad = False

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        self.sigma_data = cm_config.sigma_data
        self.sigma_max = cm_config.sigma_max
        self.sigma_min = cm_config.sigma_min
        self.disc_steps_max = cm_config.disc_steps_max
        self.disc_steps_min = cm_config.disc_steps_min
        self.lognormal_mean = cm_config.lognormal_mean 
        self.lognormal_std = cm_config.lognormal_std 
        self.weight_schedule = cm_config.weight_schedule
        self.loss_norm = cm_config.loss_norm 
        self.rho = cm_config.rho
        self.ema_decay_rate = cm_config.ema_decay_rate
        self.ae_scale = np.float32(cm_config.sigma_data) / np.float32(cm_config.ae_raw_std) if cm_config.ae_raw_std else None
        self.ae_bias = np.float32(cm_config.mu_data) - np.float32(cm_config.ae_raw_mean) * self.ae_scale if cm_config.ae_raw_mean else None
        
        ae_config = deepcopy(ae_config)
        ae_path = ae_config.get('ckpt_path', None)
        if ae_path:
            print(f'load ae weight from: {ae_path}')
            ae_config.pop('ckpt_path')
            ae = get_obj_from_str(ae_config.target).load_from_checkpoint(ae_path, strict=True, map_location="cpu")
            self.encoder = ae.encoder
            self.decoder = ae.decoder
        else:
            self.encoder = instantiate_from_config(ae_config.encoder_config)
            self.decoder = instantiate_from_config(ae_config.decoder_config)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

        self.sample_seed = self.misc_config.sample_seed

        if self.misc_config.compile:
            self.model = torch.compile(self.model)
            self.model_ema = torch.compile(self.model_ema)
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        self.val_dataset_names = [k for k in self.data_config.validate.keys()]

        self.remove_key = []
        if self.loss_norm == "lpips":
            self.remove_key.append('lpips')
            self.lpips = LPIPS(replace_pooling=True, reduction="none").to(self.device)

        self.save_hyperparameters()

    def forward(self, noisy, clean, sample_seed=None, return_z=False):
        noisy = noisy*2 - 1
        clean = clean*2 - 1

        noise = noisy - clean

        z, prompt = self.encode(noise)

        _clean = clean
        _clean = F.interpolate(_clean, scale_factor=0.5, mode='bicubic', antialias=True)
        _clean = F.pixel_unshuffle(_clean, 4)

        if sample_seed is not None:
            rand_generator = torch.Generator(device=z.device).manual_seed(sample_seed)
        else:
            rand_generator = None
        
        z = self.karras_sample(
            shape=z.shape, steps=1, device=noisy.device,
            sigma_max=self.sigma_max,sigma_min=self.sigma_min,
            rho=self.rho,sampler='onestep',
            prompt=prompt, clean=_clean,
            generator=rand_generator)

        if return_z:
            return z

        sampled_image = self.decode(z, clean)
        sampled_image = (sampled_image + 1.0) / 2.0
        sampled_image = torch.clamp(sampled_image, 0.0, 1.0)

        return sampled_image

    @torch.no_grad()
    def get_input(self, batch, config):
        if config.input_key in ['Gaussian']:
            x = self.get_z0(batch[config.target_key], init_type='gaussian').to(batch[config.target_key].device)
        elif config.input_key is None:
            x = None
        else:
            x = batch[config.input_key]
            x = x*2 - 1 # Rescale to [-1, 1]
            
        if config.target_key in ['Gaussian']:
            y = self.get_z0(x, init_type='gaussian').to(x.device)
        elif config.target_key is None:
            y = None
        else:
            y = batch[config.target_key]
            y = y*2 - 1 # Rescale to [-1, 1]
        return x, y
    
    def get_lr(self):
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler:
            return lr_scheduler.get_last_lr()[0]
        else:
            return self.optimizers().param_groups[0]['lr']
    
    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1
        
    def on_train_batch_start(self, batch, batch_idx):
        x = self.get_input(batch, self.data_config.train)[0]
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()
        
    def training_step(self, batch, batch_idx):
        noisy, clean = self.get_input(batch, self.data_config.train)

        noise = noisy - clean

        self.log("bs", self.global_batch_size, prog_bar=True,logger=False, rank_zero_only=True)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False, rank_zero_only=True)

        z, prompt = self.encode(noise)

        _clean = (clean + 1) / 2
        _clean = _clean + torch.randn_like(_clean) / 255. * 5.
        
        _clean = F.interpolate(_clean, scale_factor=0.5, mode='bicubic', antialias=True)
        _clean = F.pixel_unshuffle(_clean, 4)
        _clean = _clean * 2 - 1 

        out = self.improved_consistency_losses(x0=z, prompt=prompt, clean=(clean, _clean))

        self.log_dict(out, prog_bar=True)
        return out['train/loss']

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update(model_ema=self.model_ema, model_online=self.model, 
                        ema_decay_rate=self.ema_decay_rate)

    @torch.amp.autocast('cuda', enabled=False)
    def encode(self, x, clean=None):
        out0, out1, out2, out3 = self.encoder(x)

        p_g0 = out0[1][0]
        p_l = out0[1][1]
        
        ncm = F.pixel_unshuffle(p_l, 8)
        p0 = F.pixel_unshuffle(p_g0, 8)
        p1 = F.pixel_unshuffle(out1[1][0], 4)
        p2 = F.pixel_unshuffle(out2[1][0], 2)
        p3 = out3[1][0]
        
        z = out3[0]
        if self.ae_scale is not None:
            z = z * const_like(z, self.ae_scale).reshape(1, -1, 1, 1)
        if self.ae_bias is not None:
            z = z + const_like(z, self.ae_bias).reshape(1, -1, 1, 1)
            
        prompts = [ncm, p0, p1, p2, p3]
        
        if clean is not None:
            _clean = (clean + 1) / 2
            _clean = F.interpolate(_clean, scale_factor=0.5, mode='bicubic', antialias=True)
            _clean = F.pixel_unshuffle(_clean, 4)
            _clean = _clean * 2 - 1
            prompts.append(_clean)
            
        return z, prompts

    @torch.amp.autocast('cuda', enabled=False)
    def decode(self, z, clean): 
        if self.ae_bias is not None:
            z = z - const_like(z, self.ae_bias).reshape(1, -1, 1, 1)
        if self.ae_scale is not None:    
            z = z / const_like(z, self.ae_scale).reshape(1, -1, 1, 1)
        x = self.decoder(z, clean)
        return x

    @torch.no_grad()
    def ema_update(self, model_ema, model_online, ema_decay_rate):
        for p_ema, p_online in zip(model_ema.parameters(), model_online.parameters()):
            p_ema.data = ema_decay_rate * p_ema + (1 - ema_decay_rate) * p_online

    def configure_optimizers(self):
        optim_config = {}

        optim_config["optimizer"] = instantiate_from_config_with_arg(
            self.optimizer_config, [{'params': self.model.parameters()}])
        
        if self.scheduler_config:
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate_from_config_with_arg(
                    self.scheduler_config, optim_config["optimizer"]),
                "interval": 'step', "frequency": 1,}
        
        return optim_config

    @torch.no_grad()
    def log_image(self, key, image):
        for logger in self.loggers:
            if isinstance(logger, LocalImageLogger):
                logger.experiment.log_image('sampled_images', image, self.global_step+1)
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(key, image, self.global_step+1)
            if isinstance(logger, WandbLogger):
                logger.experiment.log({key: wandb.Image(ToPILImage()(image)),})

    def on_validation_start(self):
        self.sampled_images = []
        self.sample_steps_val = 50

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_dataset_names[dataloader_idx]
        val_config = self.data_config.validate[self.val_dataset_names[dataloader_idx]]
        self._validation_step(batch, batch_idx, val_config,
                              suffix=f"_ema_{val_name}",)
    
    def _validation_step(self, batch, batch_idx, val_config, suffix=""):
        noisy, clean = self.get_input(batch, val_config)
        assert noisy.shape[0] == 1
        noise = noisy - clean

        z, prompt = self.encode(noise)

        _clean = clean
        _clean = F.interpolate(_clean, scale_factor=0.5, mode='bicubic', antialias=True)
        _clean = F.pixel_unshuffle(_clean, 4)
        
        z1 = self.karras_sample(
            shape=z.shape, steps=1, device=noisy.device,
            sigma_max=self.sigma_max,sigma_min=self.sigma_min,
            rho=self.rho,sampler='onestep',
            prompt=prompt, clean=_clean,
            generator=torch.Generator(device=self.device).manual_seed(
                self.sample_seed) if self.sample_seed else None
            )

        z2 = self.karras_sample(
            shape=z.shape, steps=1, device=noisy.device,
            sigma_max=self.sigma_max,sigma_min=self.sigma_min,
            rho=self.rho,sampler='onestep',
            prompt=prompt, clean=_clean,
            generator=torch.Generator(device=self.device).manual_seed(
                self.sample_seed + 1) if self.sample_seed else None
            )

        sampled_image = self.decode(z1, clean)
        sampled_image = (sampled_image + 1.0) / 2.0
        sampled_image = torch.clamp(sampled_image, 0.0, 1.0)

        noisy = torch.clamp((noisy + 1.0) / 2.0, 0.0, 1.0)
        clean = torch.clamp((clean + 1.0) / 2.0, 0.0, 1.0)
        noise = noisy - clean
        fake_noise = sampled_image - clean
        
        losses = {}
        losses[f'val{suffix}/kld'] = calculate_kld(noise[0], fake_noise[0])
        losses[f'val{suffix}/akld'] = calculate_akld(noisy, sampled_image, clean)

        def min_max_norm(x, dim=None):
            if dim is None:
                return (x - x.min()) / (x.max() - x.min())
            else:
                return (x - x.amin(dim=dim, keepdim=True)) / (
                    x.amax(dim=dim, keepdim=True) - x.amin(dim=dim, keepdim=True))

        if batch_idx % 100 == 0:
            self.sampled_images.append(clean)
            self.sampled_images.append(noisy)
            self.sampled_images.append(min_max_norm(noise))
            self.sampled_images.append(sampled_image)
            self.sampled_images.append(min_max_norm(fake_noise))             

        self.log_dict(losses, prog_bar=True, sync_dist=True, add_dataloader_idx=False)               

    def on_validation_end(self):
        if len(self.sampled_images) > 0:
            imgs = torch.concat(self.sampled_images, dim=0)
            grid = make_grid(imgs, nrow=5)
            self.log_image('validation/sampled_images', grid)
            self.sampled_images.clear() # free memory

    def get_z0(self, batch, init_type='gaussian'):
      n,c,h,w = batch.shape 

      if init_type == 'gaussian':
          cur_shape = (n, c, h, w)
          return torch.randn(cur_shape)
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
    
    def improved_consistency_losses(self, x0, xT=None, prompt=None, clean=None, model_kwargs=None,):
        if model_kwargs is None:
            model_kwargs = {}

        dims = x0.ndim

        num_ts = self.discretization_fn(
            step=self.global_step, total_steps=self.trainer.max_steps, 
            disc_steps_min=self.disc_steps_min, disc_steps_max=self.disc_steps_max)

        ts = self.karras_schedule(num_timesteps=num_ts, device=x0.device)

        sampled_timestep_idx = self.lognormal_timestep_distribution(
            x0.shape[0], ts, self.lognormal_mean, self.lognormal_std
        )

        sigmas = ts

        def denoise_fn(x, prompt, clean, t, sigma):
            return self.denoise(self.model, x, prompt, clean, t, sigma, **model_kwargs)[1]
        
        if xT is None:
            xT = torch.randn_like(x0)

        t2 = ts[sampled_timestep_idx + 1]
        sigma_t2 = sigmas[sampled_timestep_idx + 1]
        x_t2 = x0 + xT * append_dims(sigma_t2, dims)

        dropout_state = (torch.get_rng_state(), torch.cuda.get_rng_state())
        distiller = denoise_fn(x_t2, prompt, clean[1], t2, sigma_t2)

        with torch.no_grad():
            t = ts[sampled_timestep_idx]
            sigma_t = sigmas[sampled_timestep_idx]
            x_t = x0 + xT * append_dims(sigma_t, dims)

            torch.set_rng_state(dropout_state[0])
            torch.cuda.set_rng_state(dropout_state[1])
            distiller_target = denoise_fn(x_t, prompt, clean[1], t, sigma_t).detach()

        weights = self.improved_loss_weighting(sigmas)[sampled_timestep_idx]

        if self.loss_norm == "pseudo_huber":
            c = 0.00054 * np.sqrt(np.prod(distiller_target.shape[1:]))
            diffs = torch.sqrt((distiller - distiller_target) ** 2 + c**2) - c
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            distiller = (self.decode(distiller, clean[0]) + 1) / 2.0
            distiller_target = (self.decode(distiller_target, clean[0]) + 1) / 2.0
            diffs = self.lpips(distiller, distiller_target)
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        result = {'train/num_timestep': num_ts, 'train/loss': loss.mean()}
        return result 
    
    def denoise(self, model, x_t, prompt, clean, t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim)
            for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        
        x_in = c_in * x_t
        rescaled_t = t
        prompt.append(clean)

        model_output = model(x_in, rescaled_t, prompt)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def karras_sample(
        self,
        shape,
        steps,
        x_T=None,
        prompt=None,
        clean=None,
        clip_denoised=False,
        progress=False,
        callback=None,
        model_kwargs=None,
        device=None,
        sigma_min=0.002,
        sigma_max=80,
        rho=7.0,
        sampler="onestep",
        generator=None,
        reverse=False,
    ):
        
        if x_T is None:
            x_T = randn_tensor(shape, generator=generator, device=device) * sigma_max

        if steps == 1:
            sigmas = self.karras_schedule(steps+1, device=device)
        else:
            sigmas = self.karras_schedule(steps, device=device)
        
        if not reverse:
            sigmas = sigmas.flip(0)

        sample_fn = {
            "onestep": self.sample_onestep,
        }[sampler]

        sampler_args = {}

        def denoiser(x_t, prompt, clean, t, sigma):
            _, denoised = self.denoise(self.model_ema, x_t, prompt, clean, t, sigma)
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

        x_0 = sample_fn(
            denoiser,
            x_T,
            prompt,
            clean,
            sigmas,
            generator,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
        return x_0

    def sample_onestep(
        self,
        distiller,
        x,
        prompt,
        clean, 
        sigmas,
        generator=None,
        progress=False,
        callback=None,
    ):
        """Single-step generation from a distilled model."""
        s_in = x.new_ones([x.shape[0]])
        return distiller(x, prompt, clean, sigmas[0] * s_in, sigmas[0] * s_in)

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min) * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def discretization_fn(self, step, total_steps, disc_steps_min, disc_steps_max):
        s0, s1 = disc_steps_min, disc_steps_max
        assert s0 <= s1

        total_steps_prime = np.floor(
            total_steps / (np.log2(np.floor(s1 / s0)) + 1)
        )
        num_timesteps = s0 * np.power(
            2, np.floor(step / total_steps_prime)
        )
        num_timesteps = np.minimum(num_timesteps, s1) + 1

        return num_timesteps

    def karras_schedule(self, num_timesteps, device=None,):
        steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
        
        rho_inv = 1.0 / self.rho
        sigmas = self.sigma_min**rho_inv + steps * (
            self.sigma_max**rho_inv - self.sigma_min**rho_inv
        )
        sigmas = sigmas**self.rho

        return sigmas

    def lognormal_timestep_distribution(
        self,
        num_samples: int,
        sigmas: torch.Tensor,
        mean: float = -1.1,
        std: float = 2.0,
    ) -> torch.Tensor:
        pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * np.sqrt(2))) - torch.erf(
            (torch.log(sigmas[:-1]) - mean) / (std * np.sqrt(2))
        )
        pdf = pdf / pdf.sum()

        timesteps = torch.multinomial(pdf, num_samples, replacement=True)

        return timesteps

    def improved_loss_weighting(self, sigmas: torch.Tensor) -> torch.Tensor:
        return 1 / (sigmas[1:] - sigmas[:-1])

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.remove_key is not None:
            self.remove_params(checkpoint, key=self.remove_key)

    def remove_params(self, checkpoint, key: list) -> None:
        del_keys = []
        for query in key:
            for k in list(checkpoint["state_dict"].keys()):
                if query in k:
                    del_keys.append(k)

        for k in del_keys:
            checkpoint["state_dict"].pop(k)
