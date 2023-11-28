import os
import glob

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from config import common_config as config


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        self.mode = mode
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.3), 
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2,p=0.8),
            A.MotionBlur(blur_limit=13, p=0.5),
            A.GaussNoise(var_limit=(0, 255), p=0.05),
            A.GridDistortion(num_steps=10, distort_limit=0.2, p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.4)
            ])
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        if self.mode == "train": 
            image = transform(image=image)["image"]
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
    
    
    
class ICDAR13Dataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        self.mode = mode
        np.random.seed(config["seed"]) # Set Numpy random seed for reproducable splits.
        mapping = {}
        datapath = r"data\icdar2013" # Path to ICDAR2013 directory
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()
        
        path_train = os.path.join(datapath, f'icdar2013_annotations_train_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        path_val = os.path.join(datapath, f'icdar2013_annotations_val_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        path_test = os.path.join(datapath, f'icdar2013_annotations_test_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        # Generate random train, val and test splits based on annotation files:
        if not (os.path.isfile(path_train) and os.path.isfile(path_val) and os.path.isfile(path_test)):
            annotation_files = ['icdar2013_annotations_train.txt', 'icdar2013_annotations_test.txt'] # Set original annotation file names
            collected_annotations = []
            for ann_file in annotation_files:
                with open(os.path.join(datapath, ann_file), "r") as f:
                    t = f.read().split("\n")
                    filtered_anns = [x for x in t if x != '']
                    collected_annotations += filtered_anns
            np.random.shuffle(collected_annotations)
            split_percentages = [0.7, 0.15, 0.15] # Must be array of length 2 or 3
            for idx, i in enumerate(split_percentages):
                split_percentages[idx] = int(np.ceil(i * len(collected_annotations)))
            if len(split_percentages) == 3:
                train_set = collected_annotations[:split_percentages[0]]
                val_set = collected_annotations[split_percentages[0]:sum(split_percentages[0:2])]
                test_set = collected_annotations[sum(split_percentages[0:2]):]
                print("Length of train, val, test: ", len(train_set), ", ", len(val_set), ", ", len(test_set))
            elif len(split_percentages) == 2:
                train_set = collected_annotations[:split_percentages[0]]
                val_set = collected_annotations[split_percentages[0]:]
                test_set = collected_annotations[split_percentages[0]:]
                print("Length of train, val = test: ", len(train_set), ", ", len(val_set), " = ", len(test_set))
            else:
                raise IndexError
            with open(path_train, "w+") as f:
                f.write('\n'.join(train_set))
            with open(path_val, "w+") as f:
                f.write('\n'.join(val_set))
            with open(path_test, "w+") as f:
                f.write('\n'.join(test_set))
        paths_file = None
        if mode == 'train':
            paths_file = path_train
        elif mode == 'dev':
            paths_file = path_val
        elif mode == 'test':
            paths_file = path_test
        paths = []
        texts = []
        with open(os.path.join(datapath, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(datapath, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.3), 
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2,p=0.8),
            A.MotionBlur(blur_limit=13, p=0.5),
            A.GaussNoise(var_limit=(0, 255), p=0.2),
            A.GridDistortion(num_steps=10, distort_limit=0.2, p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.4)
            ])
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        if self.mode == "train": 
            image = transform(image=image)["image"]
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def icdar13_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
    
    
class IIIT5KDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.displaycount = [5, 2]
        self.fig = plt.subplots(nrows=self.displaycount[0], ncols=self.displaycount[1], figsize=(15,15))

    def _load_from_raw_files(self, root_dir, mode):
        self.mode = mode
        np.random.seed(config["seed"]) # Set Numpy random seed for reproducable splits.
        mapping = {}
        datapath = r"data\IIIT5K" # Path to ICDAR2013 directory
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()
        
        path_train = os.path.join(datapath, f'iiitk5k_annotations_train_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        path_val = os.path.join(datapath, f'iiitk5k_annotations_val_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        path_test = os.path.join(datapath, f'iiitk5k_annotations_test_generated_seed_{config["seed"]}.txt') # Placeholder generated names
        # Generate random train, val and test splits based on annotation files:
        if not (os.path.isfile(path_train) and os.path.isfile(path_val) and os.path.isfile(path_test)):
            annotation_files = ['iiit5k_annotations_train.txt', 'iiit5k_annotations_test.txt'] # Set original annotation file names
            collected_annotations = []
            for ann_file in annotation_files:
                with open(os.path.join(datapath, ann_file), "r") as f:
                    t = f.read().split("\n")
                    filtered_anns = [x for x in t if x != '']
                    collected_annotations += filtered_anns
            np.random.shuffle(collected_annotations)
            split_percentages = [0.7, 0.15, 0.15] # Must be array of length 2 or 3
            for idx, i in enumerate(split_percentages):
                split_percentages[idx] = int(np.ceil(i * len(collected_annotations)))
            if len(split_percentages) == 3:
                train_set = collected_annotations[:split_percentages[0]]
                val_set = collected_annotations[split_percentages[0]:sum(split_percentages[0:2])]
                test_set = collected_annotations[sum(split_percentages[0:2]):]
                print("Length of train, val, test: ", len(train_set), ", ", len(val_set), ", ", len(test_set))
            elif len(split_percentages) == 2:
                train_set = collected_annotations[:split_percentages[0]]
                val_set = collected_annotations[split_percentages[0]:]
                test_set = collected_annotations[split_percentages[0]:]
                print("Length of train, val = test: ", len(train_set), ", ", len(val_set), " = ", len(test_set))
            else:
                raise IndexError
            with open(path_train, "w+") as f:
                f.write('\n'.join(train_set))
            with open(path_val, "w+") as f:
                f.write('\n'.join(val_set))
            with open(path_test, "w+") as f:
                f.write('\n'.join(test_set))
        paths_file = None
        if mode == 'train':
            paths_file = path_train
        elif mode == 'dev':
            paths_file = path_val
        elif mode == 'test':
            paths_file = path_test
        paths = []
        texts = []
        with open(os.path.join(datapath, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(datapath, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.3), 
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2,p=0.8),
            A.MotionBlur(blur_limit=13, p=0.5),
            A.GaussNoise(var_limit=(0, 255), p=0.2),
            A.GridDistortion(num_steps=10, distort_limit=0.2, p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.4)
            ])
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        #if self.mode == "test":    # Show image of augmentations
        #    if self.displaycount[0] == -1:
        #        plt.show()
        #        self.displaycount[0] -= 1
        #    if self.displaycount[0] >= 0:
        #        self.fig[1][self.displaycount[0] - 1, 0].imshow(image)
        #        if self.texts:
        #            self.fig[1][self.displaycount[0] - 1, 0].set_title(f"Image: {self.texts[index]}")
        #        transformimage = transform(image=image)["image"]
        #        self.fig[1][self.displaycount[0] - 1, 1].imshow(transformimage)
        #        if self.texts:
        #            self.fig[1][self.displaycount[0] - 1, 1].set_title(f"Augmentation:")
        #        self.displaycount[0] -= 1
        if self.mode == "train": 
            image = transform(image=image)["image"]
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def iiit5k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
    
    
class CocoTextV2Dataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            root_dir = "data/"
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        self.mode = mode
        np.random.seed(config["seed"]) # Set Numpy random seed for reproducable splits.
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'coco_annotations_train.txt'
        elif mode == 'dev':
            paths_file = 'coco_annotations_val.txt'
        elif mode == 'test':
            paths_file = 'coco_annotations_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=90, p=0.5), 
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2,p=0.5),
            A.MotionBlur(blur_limit=11, p=0.2),
            A.GaussNoise(var_limit=(0, 255), p=0.1),
            A.GridDistortion(num_steps=25, distort_limit=0.2, p=0.2),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2)
            ])
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)      
        #if self.mode == "train": # Additional rotational augmentation
            #image = image.rotate(np.random.choice([0, 90, 180, 270]), expand = True)
        
        #img_array = []
        #for i in [0, 90, 180, 270]:
        #    img_array.append(image.rotate(i, expand = False))
        #np.random.shuffle(img_array)
        #new_img = Image.merge('RGBA', img_array)
        #image = np.array(new_img)
        image = np.array(image)
        if self.mode == "train":
            image = transform(image=image)["image"]
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def cocotextv2_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
