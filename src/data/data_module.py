import pandas as pd
from pytorch_lightning import LightningDataModule
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

class DataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_dir = 'data/imgs'
        self.caption_csv = '/data/tilak/projects/diffarch/data/living_room_with_blip_captions.csv'

        self.df = pd.read_csv(self.caption_csv)
        # Create a dictionary to map image names to captions
        self.image_caption_map = dict(zip(self.df['image'], self.df['Blip_Caption']))

        # Split the dataset into training and validation sets (90-10 split)
        train_images, val_images = train_test_split(self.df['image'], test_size=0.1, random_state=42)

        self.train_images = train_images.tolist()
        self.val_images = val_images.tolist()

    def train_dataloader(self):
        # Define data loader for training
        train_dataset = Dataset(self.image_dir, self.train_images[:100], self.image_caption_map)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Define data loader for validation
        val_dataset = Dataset(self.image_dir, self.val_images, self.image_caption_map)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_list, img_caption_map):
        self.image_dir = image_dir
        self.images = image_list
        self.img_caption_map = img_caption_map

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load an image and apply transforms to it
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        image_np_arr = np.array(image)
        image = self.transform(image)

        low_threshold = 100
        high_threshold = 200

        canny_image = cv2.Canny(image_np_arr, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image)
        canny_image = self.transform(canny_image)

        # Return the transformed image along with its caption
        result = {}
        result['image'] = image
        result['caption'] = self.img_caption_map[self.images[idx]]
        result['hint'] = canny_image
        
        return result
