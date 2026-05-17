from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from png.utils.common import instantiate_from_config, load_state_dict, get_obj_from_str

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    if config.lightning.seed:
        pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)

    model_config = OmegaConf.load(config.model.config)
    if config.model.get("pl_resume"):
        model = get_obj_from_str(model_config.target).load_from_checkpoint(config.model.get("pl_resume"), strict=True, map_location="cpu")
    else:
        model = instantiate_from_config(model_config)

    if config.model.get("resume"):
        state_dict = torch.load(config.model.resume, map_location="cpu")# ["params"]
        if model_config.params.misc_config.compile == False:
            for key in list(state_dict.keys()):
                state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)
        load_state_dict(model, state_dict, strict=True)
        # load_state_dict(model.model, torch.load(config.model.resume, map_location="cpu")['model_state_dict'], strict=True)

    callbacks = []
    if 'callbacks' in config.lightning.keys():
        for callback_config in config.lightning.callbacks:
            callbacks.append(instantiate_from_config(callback_config))

    if not args.debug:
        loggers = []
        if 'loggers' in config.lightning.keys():
            for logger_config in config.lightning.loggers:
                logger = instantiate_from_config(logger_config)
                loggers.append(logger)
                if isinstance(logger, WandbLogger):
                    code = wandb.Artifact('code', type='code')
                    for path in Path('.').glob('**/*.py'):
                        code.add_file(path, name=str(path))
                    for path in Path('.').glob('**/*.yaml'):
                        code.add_file(path, name=str(path))
                    logger.experiment.log_artifact(code)

    if args.debug:
        loggers = []
        debug_logger = instantiate_from_config(
            {'target': 'png.models.loggers.LocalImageLogger',
             'params': {
                 'save_dir' : './logs/',
                 'name': 'LocalImageLogger',
                 'version': 'debug',
             }}
        )
        loggers.append(debug_logger)
        config.lightning.trainer.val_check_interval = 10000
        data_module.train_config.dataset.params.preload = False
        if type(data_module.val_config) == list:
            for vc in data_module.val_config:
                vc.dataset.params.preload = False
        else:
            data_module.val_config.dataset.params.preload = False

    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config.lightning.trainer)
    if config.lightning.mode == 'fit':
        trainer.fit(model, datamodule=data_module, ckpt_path=config.model.get("fit_resume"))
    elif config.lightning.mode == 'validate':
        trainer.validate(model, datamodule=data_module)
    else:
        assert False, f'unsupported mode : {config.lightning.mode}'


if __name__ == "__main__":
    main()
