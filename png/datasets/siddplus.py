# Dataset loaders for the SIDD+ validation benchmark.
import random
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import join
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class SIDDPlusDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super().__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation
        
        # Get Dataset Instances
        self.noisy_dataset, self.clean_dataset = self.get_path_list()    

        self.gt_dirs = [join(self.clean_dataset[1], fn) for fn in self.clean_dataset[0]]
        self.lq_dirs = [join(self.noisy_dataset[1], fn) for fn in self.noisy_dataset[0]]

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.gt_images = list(executor.map(self.load_image, self.gt_dirs))
                with ThreadPoolExecutor() as executor:
                    self.lq_images = list(executor.map(self.load_image, self.lq_dirs))
            else:
                self.gt_images, self.lq_images = [], []
                for img_dir in self.gt_dirs:
                    self.gt_images.append(np.array(Image.open(img_dir).convert('RGB')))
                for img_dir in self.lq_dirs:
                    self.lq_images.append(np.array(Image.open(img_dir).convert('RGB')))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            noisy = self.lq_images[index]
            clean = self.gt_images[index]
        else:
            noisy = np.array(Image.open(self.lq_dirs[index]).convert("RGB"))
            clean = np.array(Image.open(self.gt_dirs[index]).convert("RGB"))

        h, w, _ = clean.shape
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisy_dataset[0][index]
        
        return img_item

    def __len__(self):
        return len(self.noisy_dataset[0])

    def get_path_list(self) :            
        noisy_path = join(self.dataroot, "noisy")
        clean_path = join(self.dataroot, "gt")
    
        # Create List Instance for Adding Dataset Path
        noisy_path_list = listdir(noisy_path)
        clean_path_list = listdir(clean_path)
        
        # Create List Instance for Adding File Name
        noisy_name_list = [image_name for image_name in noisy_path_list if ".png" in image_name]
        clean_name_list = [image_name for image_name in clean_path_list if ".png" in image_name]
        
        # Sort List Instance
        noisy_name_list = natsorted(noisy_name_list)
        clean_name_list = natsorted(clean_name_list)
        
        return (noisy_name_list, noisy_path), (clean_name_list, clean_path)
