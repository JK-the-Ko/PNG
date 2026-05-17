# Lightning data modules for training and validation dataset wiring.
from typing import Any, Tuple, Mapping

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from png.datasets.batch_transform import BatchTransform, IdentityBatchTransform
from png.datasets.utils import RepeatDataset
from png.utils.common import instantiate_from_config

class BIRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: str,
        val_config: str=None
    ) -> "BIRDataModule":
        super().__init__()
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config) if val_config else None

    def load_dataset(self, config: Mapping[str, Any]) -> Tuple[Dataset, BatchTransform]:
        dataset = instantiate_from_config(config["dataset"])
        batch_transform = (
            instantiate_from_config(config["batch_transform"])
            if config.get("batch_transform") else IdentityBatchTransform()
        )
        return dataset, batch_transform

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset, self.train_batch_transform = self.load_dataset(self.train_config)
            if self.val_config:
                self.val_dataset, self.val_batch_transform = self.load_dataset(self.val_config)
            else:
                self.val_dataset, self.val_batch_transform = None, None
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        self.trainer: pl.Trainer
        
        if self.trainer.training:
            return self.train_batch_transform(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            return self.val_batch_transform(batch)
        else:
            raise RuntimeError(
                "Trainer state: \n"
                f"training: {self.trainer.training}\n"
                f"validating: {self.trainer.validating}\n"
                f"testing: {self.trainer.testing}\n"
                f"predicting: {self.trainer.predicting}\n"
                f"sanity_checking: {self.trainer.sanity_checking}"
            )

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_config: str, val_config: str|ListConfig=None):
        super().__init__()

        self.train_config = OmegaConf.load(train_config)

        if type(val_config) == str:
            self.val_config = OmegaConf.load(val_config)
        elif type(val_config) == ListConfig:
            self.val_config = [OmegaConf.load(vc) for vc in val_config]
        else:
            self.val_config = None
            
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = instantiate_from_config(self.train_config["dataset"])
            
            repeat_rate = self.train_config["repeat_dataset"]["times"]
            if repeat_rate > 1:
                self.train_dataset = RepeatDataset(self.train_dataset, repeat_rate)

            if type(self.val_config) == DictConfig:
                self.val_dataset = instantiate_from_config(self.val_config["dataset"])
            elif type(self.val_config) == list:
                self.val_dataset = [instantiate_from_config(vc["dataset"]) for vc in self.val_config]
            else:
                self.val_dataset = None
        elif stage == "validate":
            if type(self.val_config) == DictConfig:
                self.val_dataset = instantiate_from_config(self.val_config["dataset"])
            elif type(self.val_config) == list:
                self.val_dataset = [instantiate_from_config(vc["dataset"]) for vc in self.val_config]
            else:
                self.val_dataset = None
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if type(self.val_config) == DictConfig:
            return DataLoader(dataset=self.val_dataset, **self.val_config["data_loader"])
        elif type(self.val_config) == list:
            return [DataLoader(dataset=vd, **vc["data_loader"]) 
                    for vd, vc in zip(self.val_dataset, self.val_config)]
        else:
            return None
