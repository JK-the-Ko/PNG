import argparse
import multiprocessing as py_mp
import os
import sys
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm

# Allow running this file directly with:
#   python png/misc/generate_image.py
# Without this, Python starts from png/misc and cannot resolve the top-level
# package import "png.models...".
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from png.models.lit_prompt_cm import LitPromptCM
from png.utils.common import instantiate_from_config


def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(image).save(save_path)


def tensor_batch_to_uint8_images(images):
    images = images.detach().clamp(0.0, 1.0)
    images = images.mul(255.0).add(0.5).to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu().numpy()


def wait_for_save_futures(futures, wait_all=False):
    if not futures:
        return []

    done, pending = wait(
        futures,
        return_when=ALL_COMPLETED if wait_all else FIRST_COMPLETED,
    )
    for future in done:
        future.result()

    return list(pending)


def get_device(opt, rank):
    if opt.device == 'cpu':
        return torch.device('cpu')
    return torch.device(f'cuda:{rank}')


def setup_distributed(opt, rank, world_size):
    if world_size == 1:
        return

    # NCCL is the preferred backend for multi-GPU jobs. Gloo keeps the same
    # code path usable for CPU debugging.
    backend = 'gloo' if opt.device == 'cpu' else 'nccl'
    os.environ['MASTER_ADDR'] = opt.master_addr
    os.environ['MASTER_PORT'] = str(opt.master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def worker(rank, world_size, opt, dataset_config, dataloader_config):
    # One worker process owns one GPU. For single-GPU or CPU runs, rank is 0.
    setup_distributed(opt, rank, world_size)
    device = get_device(opt, rank)

    if opt.device != 'cpu':
        torch.cuda.set_device(device)

    try:
        model = LitPromptCM.load_from_checkpoint(opt.ckpt_path, strict=True, map_location=device)
        model.eval()
        model.freeze()
        model.to(device)
        
        train_dataset = instantiate_from_config(dataset_config)
        if world_size > len(train_dataset):
            raise ValueError(f'world_size={world_size} is larger than dataset size={len(train_dataset)}.')

        if world_size > 1:
            # Use a strided split instead of DistributedSampler. DistributedSampler
            # pads by duplicating samples when the dataset is not divisible by
            # world_size, which would create duplicated output images.
            train_dataset = Subset(train_dataset, range(rank, len(train_dataset), world_size))

        dataloader_config = dict(dataloader_config)
        dataloader_config['shuffle'] = False

        train_dataloader = DataLoader(dataset=train_dataset, **dataloader_config)
        
        # Only rank 0 shows progress so multiple processes do not write over each
        # other in the terminal.
        tqdm_dataloader = tqdm(train_dataloader, disable=rank != 0)

        save_workers = opt.save_workers
        if save_workers is None:
            save_workers = max(1, min(4, (os.cpu_count() or world_size) // world_size))
        if save_workers < 0:
            raise ValueError('--save_workers must be non-negative.')

        save_futures = []
        max_pending_saves = max(1, save_workers * 4)
        if save_workers > 0:
            save_pool_context = ProcessPoolExecutor(
                max_workers=save_workers,
                mp_context=py_mp.get_context('spawn'),
            )
        else:
            save_pool_context = nullcontext()

        with save_pool_context as save_pool:
            with torch.inference_mode():
                for batch in tqdm_dataloader :
                    noisy = batch['LQ'].to(device, non_blocking=True)
                    clean = batch['GT'].to(device, non_blocking=True)
                    name = batch['file_name']

                    gen_noisy = tensor_batch_to_uint8_images(model(noisy, clean))
                    for image, file_name in zip(gen_noisy, name):
                        save_path = os.path.join(opt.save_path, file_name)
                        if save_pool is None:
                            save_image(image, save_path)
                        else:
                            save_futures.append(save_pool.submit(save_image, image, save_path))

                    if len(save_futures) >= max_pending_saves:
                        save_futures = wait_for_save_futures(save_futures)

            wait_for_save_futures(save_futures, wait_all=True)
    finally:
        cleanup_distributed()


def main(opt, dataset_config, dataloader_config):
    os.makedirs(opt.save_path, exist_ok=True)

    if opt.device == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. Use --device cpu or run on a CUDA-enabled machine.')

        available_gpus = torch.cuda.device_count()
        # By default, use all visible GPUs. --num_gpus can restrict the run to
        # the first N CUDA devices.
        world_size = opt.num_gpus if opt.num_gpus is not None else available_gpus
        if world_size < 1:
            raise ValueError('--num_gpus must be at least 1.')
        if world_size > available_gpus:
            raise ValueError(f'--num_gpus={world_size} but only {available_gpus} CUDA device(s) are available.')
    else:
        world_size = 1

    if world_size > 1:
        # Spawn creates independent Python processes. This avoids GIL limits and
        # matches the usual PyTorch distributed inference pattern.
        mp.spawn(worker, args=(world_size, opt, dataset_config, dataloader_config), nprocs=world_size, join=True)
    else:
        worker(0, world_size, opt, dataset_config, dataloader_config)


if __name__ == '__main__' :
    args = argparse.ArgumentParser()
    args.add_argument('--dataroot', type=str, required=True)
    args.add_argument('--dataset_type', type=str, choices=['sidd', 'siddplus', 'polyu', 'nam'], required=True)
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--num_workers', type=int, default=10)
    args.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    args.add_argument('--num_gpus', type=int, default=None)
    args.add_argument('--save_workers', type=int, default=None)
    args.add_argument('--master_addr', type=str, default='127.0.0.1')
    args.add_argument('--master_port', type=int, default=29500)
    opt = args.parse_args()
    
    dataset_dict = {'sidd':'png.datasets.sidd.SIDDDataset',
                    'siddplus':'png.datasets.siddplus.SIDDPlusDataset',
                    'polyu':'png.datasets.polyu.PolyUDataset',
                    'nam':'png.datasets.nam.NAMDataset'}
    
    dataset_config = {'target': dataset_dict[opt.dataset_type],
                      'params': {
                          'dataroot': opt.dataroot,
                          'patch_size': 256,
                          'preload': False, 
                          'parallel_preload': False,
                          'augmentation': False,
                          'test': True,}}
    
    dataloader_config = {'batch_size': opt.batch_size,
                      	 'shuffle': False,
                         'num_workers': opt.num_workers,
                         'drop_last': False}
    
    main(opt, dataset_config, dataloader_config)
