# Lightning module for supervised image denoising models.
from typing import Mapping, Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from lpips import LPIPS
from torchvision.utils import make_grid

from png.models.lit_prompt_cm import LitPromptCM
from png.models.loggers import TensorBoardLogger, WandbLogger, LocalImageLogger
from png.utils.common import instantiate_from_config, instantiate_from_config_with_arg
from png.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

class LitDenoising(pl.LightningModule):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        denoiser_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] = None,
        misc_config: Mapping[str, Any] = None,):
        super().__init__()

        # instantiate denoiser
        self.model = instantiate_from_config(denoiser_config)
        self.remove_key = ['lpips']

        if misc_config.generator_pth:
            self.generator = LitPromptCM.load_from_checkpoint(misc_config.generator_pth, strict=True)
            self.generator.freeze()
            self.remove_key.append('generator')
            del misc_config.generator_pth
        else:
            self.generator = None

        if misc_config.compile:
            self.model = torch.compile(self.model)
        self.loss = instantiate_from_config(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        self.val_dataset_names = [k for k in self.data_config.validate.keys()]

        self.save_hyperparameters()

    def forward(self, noisy):
        return self.model(noisy)
    
    @torch.no_grad()
    def get_input(self, batch, config):
        x = batch[config.input_key]
        y = batch[config.target_key]
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
        x, y = self.get_input(batch, self.data_config.train)
        self.log("bs", self.global_batch_size, prog_bar=True,logger=False, rank_zero_only=True)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False, rank_zero_only=True)

        losses = dict()
        if self.generator:
            x = self.generator(noisy=x, clean=y)
        pred = self.model(x)

        losses['train/loss'] = self.loss(pred, y)
        
        total_loss = sum(losses.values())
        losses['train/total'] = sum(losses.values())

        self.log_dict(losses, prog_bar=True)
        return total_loss
    
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
                logger.experiment.log({key: wandb.Image(image),})

    def on_validation_start(self):
        self.lpips_metric = LPIPS(net="alex").to(self.device)
        self.sampled_images = []
        self.sample_steps_val = 50

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_dataset_names[dataloader_idx]
        val_config = self.data_config.validate[self.val_dataset_names[dataloader_idx]]
        self._validation_step(batch, batch_idx, val_config,
                              suffix=f"/{val_name}",)
    
    def _validation_step(self, batch, batch_idx, val_config, suffix=""):
        x, y = self.get_input(batch, val_config)
        assert x.shape[0] == 1
        pred = self.model(x)
        pred = torch.clamp(pred, 0.0, 1.0)

        # evalutate using noise metrics
        losses = {}
        losses[f'val{suffix}/psnr'] = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean()
        losses[f'val{suffix}/ssim'] = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean()
        losses[f'val{suffix}/lpips'] = self.lpips_metric(pred, y, normalize=True).mean()

        # log the outputs!
        self.log_dict(losses, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        
        if batch_idx % 500 == 0:
            self.sampled_images.append(x[0].cpu())
            self.sampled_images.append(y[0].cpu())
            self.sampled_images.append(pred[0].cpu())
    
    def on_validation_end(self):
        grid = make_grid(self.sampled_images, nrow=3)
        self.log_image('validation/sampled_images', grid)
        self.sampled_images.clear() # free memory

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.remove_key is not None:
            print(f"{checkpoint['state_dict'].keys()} in state_dict ")
            self.remove_params(checkpoint, key=self.remove_key)
            print(f"{checkpoint['state_dict'].keys()} in state_dict after remove params\n")


    def remove_params(self, checkpoint, key: list) -> None:
        del_keys = []
        for query in key:
            for k in list(checkpoint["state_dict"].keys()):
                if query in k:
                    del_keys.append(k)
        print(f"{len(del_keys)} keys to remove")

        for k in del_keys:
            checkpoint["state_dict"].pop(k)
