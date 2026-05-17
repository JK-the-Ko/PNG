import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow running this file directly with:
#   python png/misc/compute_latent_statistics.py
# Without this, Python starts from png/misc and cannot resolve the top-level
# package import "png.models...".
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from png.models.lit_prompt_ae import LitPromptAE
from png.utils.common import instantiate_from_config


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


def reduce_stat(stat, device, world_size):
    # Each process accumulates channel-wise sums for only its dataset shard.
    # Reducing sums and counts gives the exact global average, including when
    # the final local batch is smaller than batch_size.
    latent_sum = torch.stack(stat['latent_sum']).sum(dim=0).to(device)
    latent_std_sum = torch.stack(stat['latent_std_sum']).sum(dim=0).to(device)
    sample_count = torch.tensor(stat['sample_count'], dtype=torch.float64, device=device)

    if world_size > 1:
        dist.all_reduce(latent_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(latent_std_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

    return {
        'latent_mean': latent_sum / sample_count,
        'latent_std': latent_std_sum / sample_count,
    }


def worker(rank, world_size, opt, dataset_config, dataloader_config):
    # One worker process owns one GPU. For single-GPU or CPU runs, rank is 0.
    setup_distributed(opt, rank, world_size)
    device = get_device(opt, rank)

    if opt.device != 'cpu':
        torch.cuda.set_device(device)

    model = LitPromptAE.load_from_checkpoint(opt.ckpt_path, strict=True, map_location=device)
    model.eval()
    model.freeze()
    model.to(device)
    
    train_dataset = instantiate_from_config(dataset_config)
    if world_size > len(train_dataset):
        raise ValueError(f'world_size={world_size} is larger than dataset size={len(train_dataset)}.')

    if world_size > 1:
        # Use a strided split instead of DistributedSampler. DistributedSampler
        # pads by duplicating samples when the dataset is not divisible by
        # world_size, which would bias the latent statistics.
        train_dataset = Subset(train_dataset, range(rank, len(train_dataset), world_size))

    dataloader_config = dict(dataloader_config)
    dataloader_config['shuffle'] = False

    train_dataloader = DataLoader(dataset=train_dataset, **dataloader_config)
    
    # Only rank 0 shows progress so multiple processes do not write over each
    # other in the terminal.
    tqdm_dataloader = tqdm(train_dataloader, disable=rank != 0)

    latent_stat = defaultdict(list)
    latent_stat['sample_count'] = 0

    with torch.inference_mode():
        for batch in tqdm_dataloader :
            noisy = batch['LQ'].to(device)
            clean = batch['GT'].to(device)

            latent_noise = model(noisy, clean)[-1][-1][0]
            # Average each sample over spatial dimensions first, then sum over
            # samples. This preserves correct weighting across uneven batches.
            latent_mean = latent_noise.mean(dim=(2, 3)).double()
            latent_std = latent_noise.std(dim=(2, 3)).double()

            latent_stat['latent_sum'].append(latent_mean.sum(dim=0))
            latent_stat['latent_std_sum'].append(latent_std.sum(dim=0))
            latent_stat['sample_count'] += latent_noise.shape[0]

    results = reduce_stat(latent_stat, device, world_size)

    if rank == 0:
        def summarize(x):
            return [round(i, 4) for i in x.detach().cpu().tolist()]

        for key, value in results.items():
            print(f'{key} : {summarize(value)}')

    cleanup_distributed()


def main(opt, dataset_config, dataloader_config):
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
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--num_workers', type=int, default=10)
    args.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    args.add_argument('--num_gpus', type=int, default=None)
    args.add_argument('--master_addr', type=str, default='127.0.0.1')
    args.add_argument('--master_port', type=int, default=29500)
    opt = args.parse_args()
    
    dataset_config = {'target': 'png.datasets.sidd.SIDDDataset',
                      'params': {
                          'dataroot': opt.dataroot,
                          'patch_size': 256,
                          'preload': False, 
                          'parallel_preload': False,
                          'augmentation': False,
                          'test': False,}}
    
    dataloader_config = {'batch_size': opt.batch_size,
                      	 'shuffle': False,
                         'num_workers': opt.num_workers,
                         'drop_last': False}
    
    main(opt, dataset_config, dataloader_config)