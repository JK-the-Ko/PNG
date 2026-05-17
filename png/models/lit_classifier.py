# Lightning module for prompt-based and baseline noise classifiers.
from typing import Mapping, Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchmetrics import Accuracy

from png.utils.common import get_obj_from_str
from png.utils.common import instantiate_from_config, instantiate_from_config_with_arg

class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        classifier_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        encoder_config: Mapping[str, Any] = None,
        scheduler_config: Mapping[str, Any] = None,
        compile: bool = False,):
        super().__init__()

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config
        
        if encoder_config:
            encoder_path = encoder_config.get('ckpt_path', None)
            if encoder_path:
                del encoder_config['ckpt_path']
                self.ae = get_obj_from_str(encoder_config.target).load_from_checkpoint(encoder_path, strict=True, map_location=self.device)
                self.ae.freeze()
        else :
            self.ae = None

        self.classifier = instantiate_from_config(classifier_config)
        if self.ae is not None :
            self.classifier_type = classifier_config.params.prompt_type
        
        if compile:
            if self.ae is not None:
                self.ae = torch.compile(self.ae)
            self.classifier = torch.compile(self.classifier)
        
        self.save_hyperparameters()

    def forward(self, noisy, clean, label):
        loss, pred = self.compute_loss(noisy, clean, label)

        return loss, pred

    @torch.no_grad()
    def get_input(self, batch, config):
        x = batch[config.input_key]
        x = x*2 - 1 # Rescale to [-1, 1]
        
        y = batch[config.target_key]
        y = y*2 - 1 # Rescale to [-1, 1]

        label = batch[config.label_key]
        
        return x, y, label
    
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

    def compute_loss(self, noisy, clean, label) :
        # Feed-Forward Encoder
        latent_noise = self.ae.encoder(noisy-clean)
        
        # Feed-Foward Prompt-Classifier
        scale_0 = torch.cat(latent_noise[0][1], dim=1).detach() if len(latent_noise[0][1]) != 1 else latent_noise[0][1][0].detach()
        scale_1 = torch.cat(latent_noise[1][1], dim=1).detach() if len(latent_noise[1][1]) != 1 else latent_noise[1][1][0].detach()
        scale_2 = torch.cat(latent_noise[2][1], dim=1).detach() if len(latent_noise[2][1]) != 1 else latent_noise[2][1][0].detach()
        scale_3 = torch.cat(latent_noise[3][1], dim=1).detach() if len(latent_noise[3][1]) != 1 else latent_noise[3][1][0].detach()
        pred = self.classifier(scale_0, scale_1, scale_2, scale_3)
        
        # Compute Cross-Entrotpy Loss
        loss = F.cross_entropy(pred, label)

        return loss, pred

    def training_step(self, batch, batch_idx):
        noisy, clean, label = self.get_input(batch, self.data_config.train)

        loss, _ = self.compute_loss(noisy, clean, label)

        losses = {}
        losses['train/ce_loss'] = loss
        losses['train/total'] = losses['train/ce_loss']
        
        self.log("bs", self.global_batch_size, prog_bar=True,logger=False, rank_zero_only=True)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False, rank_zero_only=True)

        self.log_dict(losses, prog_bar=True)
        return losses['train/total']
    
    def configure_optimizers(self):
        optim_config = {}

        optim_config["optimizer"] = instantiate_from_config_with_arg(
            self.optimizer_config, [{'params': list(self.classifier.parameters())}])
        
        if self.scheduler_config:
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate_from_config_with_arg(
                    self.scheduler_config, optim_config["optimizer"]),
                "interval": 'step', "frequency": 1,}
        
        return optim_config
        
    def validation_step(self, batch, batch_idx):
        self.acc_top_1 = Accuracy(task="multiclass", num_classes=16, top_k=1).to(self.device)
        self.acc_top_3 = Accuracy(task="multiclass", num_classes=16, top_k=3).to(self.device)
        self.sensor_dict = {0:"S6", 1:"S6", 2:"S6", 3:"S6",
                            4:"GP", 5:"GP", 6:"GP", 7:"GP",
                            8:"N6", 9:"N6", 10:"N6", 11:"G4", 
                            12:"G4", 13:"IP", 14:"IP", 15:"IP"}
        self._validation_step(batch, batch_idx, suffix="")
    
    def _validation_step(self, batch, batch_idx, suffix=""):
        noisy, clean, label = self.get_input(batch, self.data_config.validate)
        
        _, pred = self.compute_loss(noisy, clean, label)
        
        overall_acc_top_1, overall_acc_top_3 = self.acc_top_1(pred, label), self.acc_top_3(pred, label)
        camera_model_acc = self.sensor_dict[torch.argmax(pred, dim=1).cpu().item()] == self.sensor_dict[label.cpu().item()]

        results = {}
        results["val/overall_acc_top_1"] = overall_acc_top_1
        results["val/overall_acc_top_3"] = overall_acc_top_3
        results["val/camera_model_acc"] = camera_model_acc

        self.log_dict(results, prog_bar=True, sync_dist=True)