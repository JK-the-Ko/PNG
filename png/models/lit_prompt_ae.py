# Lightning module for the prompt autoencoder.
from typing import Mapping, Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from png.models.loggers import TensorBoardLogger, WandbLogger, LocalImageLogger
from png.utils.common import instantiate_from_config, instantiate_from_config_with_arg
from png.utils.metrics import calculate_psnr_pt, calculate_ssim_pt, calculate_kld, calculate_akld

class LitPromptAE(pl.LightningModule):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        encoder_config: Mapping[str, Any],
        decoder_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        loss_config: Mapping[str, Any] = None,
        scheduler_config: Mapping[str, Any] = None,
        misc_config: Mapping[str, Any] = None,):
        super().__init__()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config
        self.loss = instantiate_from_config(loss_config)
        
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        
        if self.misc_config.compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            
        self.val_dataset_names = [k for k in self.data_config.validate.keys()]
        
        self.save_hyperparameters()

    def forward(self, noisy, clean):
        noisy = noisy*2 - 1
        clean = clean*2 - 1

        real_noise = noisy - clean
        latent_noise = self.encoder(real_noise)
        pred_noisy = self.decoder(latent_noise[-1][0], clean)

        return pred_noisy, latent_noise

    @torch.no_grad()
    def get_input(self, batch, config):
        x = batch[config.input_key]
        x = x*2 - 1 # Rescale to [-1, 1]
        
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

        latent_noise = self.encoder(noisy - clean)
        rec_noisy = self.decoder(latent_noise[-1][0], clean)
        
        lossess = {}
        lossess['train/recon'] = self.loss(rec_noisy, noisy)
        lossess['train/l2'] = torch.norm(latent_noise[-1][0])*self.misc_config.weight_l2
        lossess['train/total'] = sum(lossess.values())
        
        self.log("bs", self.global_batch_size, prog_bar=True,logger=False, rank_zero_only=True)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False, rank_zero_only=True)
        self.log_dict(lossess, prog_bar=True)
        return lossess['train/total']

    def configure_optimizers(self):
        optim_config = {}

        optim_config["optimizer"] = instantiate_from_config_with_arg(
            self.optimizer_config, [{'params': list(self.encoder.parameters()) + list(self.decoder.parameters())}])
        
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_dataset_names[dataloader_idx]
        val_config = self.data_config.validate[self.val_dataset_names[dataloader_idx]]
        self._validation_step(batch, batch_idx, val_config,
                              suffix=f"_pae_{val_name}",)
    
    def _validation_step(self, batch, batch_idx, val_config, suffix=""):
        noisy, clean = self.get_input(batch, val_config)
        
        real_noise = noisy - clean
        latent_noise = self.encoder(real_noise)
        pred_noisy = self.decoder(latent_noise[-1][0], clean)

        clean = torch.clamp((clean+1)/2, 0., 1.)
        noisy = torch.clamp((noisy+1)/2, 0., 1.)
        pred_noisy = torch.clamp((pred_noisy+1)/2, 0., 1.)
        
        real_noise = noisy - clean
        pred_noise = pred_noisy - clean

        results = {}
        results[f'val{suffix}/psnr'] = calculate_psnr_pt(noisy, pred_noisy, 0, test_y_channel=False).mean()
        results[f'val{suffix}/ssim'] = calculate_ssim_pt(noisy, pred_noisy, 0, test_y_channel=False).mean()
        results[f'val{suffix}/kld'] = calculate_kld(real_noise[0], pred_noise[0])
        results[f'val{suffix}/akld'] = calculate_akld(noisy, pred_noisy, clean)

        self.log_dict(results, prog_bar=True, sync_dist=True)
        
        def min_max_norm(x, dim=None):
            if dim is None:
                return (x - x.min()) / (x.max() - x.min())
            else:
                return (x - x.amin(dim=dim, keepdim=True)) / (
                    x.amax(dim=dim, keepdim=True) - x.amin(dim=dim, keepdim=True))

        if batch_idx % 100 == 0:
            self.sampled_images.append(clean)
            self.sampled_images.append(noisy)
            self.sampled_images.append(pred_noisy)
            self.sampled_images.append(min_max_norm(real_noise))
            self.sampled_images.append(min_max_norm(pred_noise))               

    def on_validation_end(self):
        if len(self.sampled_images) > 0:
            imgs = torch.concat(self.sampled_images, dim=0)
            grid = make_grid(imgs, nrow=5)
            self.log_image('validation/sampled_images', grid)
            self.sampled_images.clear() # free memory
