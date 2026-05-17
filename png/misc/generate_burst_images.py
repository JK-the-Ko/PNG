import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow running this file directly with:
#   python png/misc/generate_burst_images.py
# Without this, Python starts from png/misc and cannot resolve the top-level
# package import "png.models...".
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from png.models.lit_prompt_cm import LitPromptCM


def get_device(device_name, gpu_id):
    if device_name == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use --device cpu or run on a CUDA-enabled machine.')
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f'--gpu_id={gpu_id} but only {torch.cuda.device_count()} CUDA device(s) are available.')
    return torch.device(f'cuda:{gpu_id}')


def load_rgb_image(path):
    return Image.open(path).convert('RGB')


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_pil_image(tensor):
    tensor = tensor.detach().clamp(0.0, 1.0)
    tensor = tensor.mul(255.0).add(0.5).to(torch.uint8)
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def validate_image_pair(clean, noisy):
    if clean.size != noisy.size:
        raise ValueError(f'clean and noisy images must have the same size, got {clean.size} and {noisy.size}.')

    width, height = clean.size
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(
            'image width and height must be divisible by 8 for this model. '
            f'Got width={width}, height={height}.'
        )


def build_gif(frames, save_path, duration, loop):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )


def generate_burst(model, clean, noisy, num_images, batch_size, seed, device):
    clean_tensor = image_to_tensor(clean).to(device)
    noisy_tensor = image_to_tensor(noisy).to(device)

    frames = []
    with torch.inference_mode():
        for start_idx in tqdm(range(0, num_images, batch_size)):
            current_batch_size = min(batch_size, num_images - start_idx)
            clean_batch = clean_tensor.repeat(current_batch_size, 1, 1, 1)
            noisy_batch = noisy_tensor.repeat(current_batch_size, 1, 1, 1)
            generated = model(noisy_batch, clean_batch, sample_seed=seed + start_idx)

            for image in generated:
                frames.append(tensor_to_pil_image(image.unsqueeze(0)))

    return frames


def main(opt):
    if opt.num_images < 1:
        raise ValueError('--num_images must be at least 1.')
    if opt.batch_size < 1:
        raise ValueError('--batch_size must be at least 1.')
    if opt.duration < 1:
        raise ValueError('--duration must be at least 1 millisecond.')

    device = get_device(opt.device, opt.gpu_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    clean = load_rgb_image(opt.clean_path)
    noisy = load_rgb_image(opt.noisy_path)
    validate_image_pair(clean, noisy)

    model = LitPromptCM.load_from_checkpoint(opt.ckpt_path, strict=True, map_location=device)
    model.eval()
    model.freeze()
    model.to(device)

    frames = generate_burst(
        model=model,
        clean=clean,
        noisy=noisy,
        num_images=opt.num_images,
        batch_size=opt.batch_size,
        seed=opt.seed,
        device=device,
    )
    build_gif(frames, opt.save_path, opt.duration, opt.loop)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--clean_path', type=str, required=True)
    args.add_argument('--noisy_path', type=str, required=True)
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--num_images', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--duration', type=int, default=50)
    args.add_argument('--loop', type=int, default=0)
    args.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    args.add_argument('--gpu_id', type=int, default=0)
    opt = args.parse_args()

    main(opt)
