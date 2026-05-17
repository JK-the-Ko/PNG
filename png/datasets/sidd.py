# Dataset loaders for SIDD image denoising and classification tasks.
import random
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import exists
from os.path import join
from pathlib import Path

import numpy as np
import torch
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

class SIDDDataset(Dataset) :
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
        # Support both the original SIDD layout and experiment-specific exports.
        candidate_dirs = [
            ("noisy", "clean"),
            ("input", "groundtruth")
        ]
        noisy_path, clean_path = None, None
        for noisy_dir, clean_dir in candidate_dirs:
            candidate_noisy = join(self.dataroot, noisy_dir)
            candidate_clean = join(self.dataroot, clean_dir)
            if exists(candidate_noisy) and exists(candidate_clean):
                noisy_path, clean_path = candidate_noisy, candidate_clean
                break

        if noisy_path is None or clean_path is None:
            raise FileNotFoundError(
                f"Could not find a supported SIDD directory pair under {self.dataroot}. "
                "Expected one of: noisy/clean, input/groundtruth, pinr_noisy/pae_clean."
            )
    
        # Create List Instance for Adding Dataset Path
        noisy_path_list = listdir(noisy_path)
        clean_path_list = listdir(clean_path)
        
        # Create List Instance for Adding File Name
        noisy_name_list = [image_name for image_name in noisy_path_list if ".png" in image_name or ".PNG" in image_name]
        clean_name_list = [image_name for image_name in clean_path_list if ".png" in image_name or ".PNG" in image_name]
        
        # Sort List Instance
        noisy_name_list = natsorted(noisy_name_list)
        clean_name_list = natsorted(clean_name_list)
        
        return (noisy_name_list, noisy_path), (clean_name_list, clean_path)


class SIDDSynDataset(Dataset) :
    def __init__(self, clean_dir:str, noisy_dir:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super().__init__()

        # Initialize Variables
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
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
        if not exists(self.noisy_dir):
            raise FileNotFoundError(f"Could not find noisy image directory: {self.noisy_dir}")
        if not exists(self.clean_dir):
            raise FileNotFoundError(f"Could not find clean image directory: {self.clean_dir}")

        # Create List Instance for Adding Dataset Path
        noisy_path_list = listdir(self.noisy_dir)
        clean_path_list = listdir(self.clean_dir)

        # Create List Instance for Adding File Name
        noisy_name_list = [image_name for image_name in noisy_path_list if ".png" in image_name or ".PNG" in image_name]
        clean_name_list = [image_name for image_name in clean_path_list if ".png" in image_name or ".PNG" in image_name]

        # Sort List Instance
        noisy_name_list = natsorted(noisy_name_list)
        clean_name_list = natsorted(clean_name_list)

        if len(noisy_name_list) != len(clean_name_list):
            raise ValueError(
                f"Clean/noisy image count mismatch: {len(clean_name_list)} clean images in "
                f"{self.clean_dir}, {len(noisy_name_list)} noisy images in {self.noisy_dir}."
            )

        return (noisy_name_list, self.noisy_dir), (clean_name_list, self.clean_dir)
    
class SIDDClsDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super().__init__()

        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation

        # Generate Meta-Data
        self.meta_data_list = self.filter_image()

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
    
    def filter_image(self) :
        # Create Dictionary Instance
        sensor_dict = {"S6":{400:0, 800:1, 1600:2, 3200:3},
                      "GP":{1600:4, 3200:5, 6400:6, 10000:7},
                      "N6":{400:8, 800:9, 3200:10},
                      "G4":{400:11, 800:12},
                      "IP":{1000:13, 1600:14, 2000:15}}
        
        # Create List Instace for Adding Meta-Data
        meta_data_list = []
        
        if self.test :
            f = open("meta-data/sidd/Benchmark-Info.txt", mode="r")
            for local_info in f.read().split("\n") :
                local_info_list = local_info.split("_")
                file_id, sensor_info, iso = local_info_list[0], local_info_list[2], int(local_info_list[3])
                if sensor_info in sensor_dict.keys() :
                    if iso in sensor_dict[sensor_info].keys() :
                        meta_data_list.append([file_id.split(" ")[0], sensor_info, iso, sensor_dict[sensor_info][iso]])
        else :
            f = open("meta-data/sidd/Scene-Instance.txt", mode="r")
            for local_info in f.read().split("\n") :
                local_info_list = local_info.split("_")
                file_id, sensor_info, iso = local_info_list[0], local_info_list[2], int(local_info_list[3])
                if sensor_info in sensor_dict.keys() :
                    if iso in sensor_dict[sensor_info].keys() :
                        meta_data_list.append([file_id, sensor_info, iso, sensor_dict[sensor_info][iso]])
                    
        return meta_data_list
    
    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            noisy = self.lq_images[index]
            clean = self.gt_images[index]
        else:
            noisy = np.array(Image.open(self.lq_dirs[index]).convert("RGB"))
            clean = np.array(Image.open(self.gt_dirs[index]).convert("RGB"))

        h, w, _ = clean.shape
        if self.patch_size < h and self.patch_size < w:
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else :
            clean_patch = clean
            noisy_patch = noisy

        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)

        if self.test:
            for meta_data in self.meta_data_list :
                if int(meta_data[0]) == int(self.noisy_dataset[0][index].split("-")[0]) :
                    label = meta_data[-1]
        else :
            for meta_data in self.meta_data_list :
                if meta_data[0] in self.noisy_dataset[0][index] :
                    label = meta_data[-1]

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisy_dataset[0][index]
        img_item['label'] = torch.tensor(label).type(torch.LongTensor)

        return img_item
    
    def __len__(self):
        return len(self.noisy_dataset[0])
    
    def get_path_list(self) :
        # Support both the original SIDD layout and experiment-specific exports.
        candidate_dirs = [
            ("noisy", "clean"),
            ("input", "groundtruth")
        ]
        noisy_path, clean_path = None, None
        for noisy_dir, clean_dir in candidate_dirs:
            candidate_noisy = join(self.dataroot, noisy_dir)
            candidate_clean = join(self.dataroot, clean_dir)
            if exists(candidate_noisy) and exists(candidate_clean):
                noisy_path, clean_path = candidate_noisy, candidate_clean
                break

        if noisy_path is None or clean_path is None:
            raise FileNotFoundError(
                f"Could not find a supported SIDD directory pair under {self.dataroot}. "
                "Expected one of: noisy/clean, input/groundtruth, pinr_noisy/pae_clean."
            )

        # Create List Instance for Adding Dataset Path
        noisy_path_list = listdir(noisy_path)
        clean_path_list = listdir(clean_path)
        
        # Create List Instance for Adding File Name
        if self.test :
            # Create List Instance for Adding File Name
            noisy_name_list, clean_name_list = [], []
            
            for image_name in noisy_path_list :
                if ".png" in image_name or ".PNG" in image_name :
                    for meta_data in self.meta_data_list : 
                        if int(image_name.split("-")[0]) == int(meta_data[0]) :
                            noisy_name_list.append(image_name)
            for image_name in clean_path_list :
                if ".png" in image_name or ".PNG" in image_name :
                    for meta_data in self.meta_data_list : 
                        if int(image_name.split("-")[0]) == int(meta_data[0]) :
                            clean_name_list.append(image_name)
        else :
            # Create List Instance for Adding File Name
            noisy_name_list, clean_name_list = [], []
            
            for meta_data in self.meta_data_list :
                for image_name in noisy_path_list :
                    if meta_data[0] in image_name :
                        if ".png" in image_name or ".PNG" in image_name :
                            noisy_name_list.append(image_name)            
                for image_name in clean_path_list :
                    if meta_data[0] in image_name :
                        if ".png" in image_name or ".PNG" in image_name :
                            clean_name_list.append(image_name)
                            
        # Sort List Instance
        noisy_name_list = natsorted(noisy_name_list)
        clean_name_list = natsorted(clean_name_list)

        return (noisy_name_list, noisy_path), (clean_name_list, clean_path)