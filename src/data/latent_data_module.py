import pandas as pd
from pytorch_lightning import LightningDataModule
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class LatentDataModule(LightningDataModule):
    def __init__(self, batch_size=4, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_dir = 'data/imgs'
        self.latents_dir = 'data/latents'
        self.caption_csv = '/data/tilak/projects/diffarch/data/master.csv'

        self.df = pd.read_csv(self.caption_csv)

        # Perform train-test split
        self.train_images, self.test_images = train_test_split(
            self.df['image'].tolist(), test_size=0.02,
        )

        self.train_images = self.train_images

        # Create dictionaries for train and test image-caption pairs
        self.image_caption_map = dict(zip(self.df['image'], self.df['Blip_Caption']))
        self.train_image_caption_map = {img: self.image_caption_map[img] for img in self.train_images}
        self.test_image_caption_map = {img: self.image_caption_map[img] for img in self.test_images}

    def train_dataloader(self):
        train_dataset = Dataset(self.latents_dir, self.train_images, self.train_image_caption_map)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        test_dataset = Dataset(self.latents_dir, self.test_images, self.test_image_caption_map)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, latents_dir, image_list, img_caption_map):
        self.latents_dir = latents_dir
        self.image_list = image_list
        self.img_caption_map = img_caption_map

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_file = self.image_list[idx]
        img_name = img_file.rsplit('.')[0]
        latent_path = os.path.join(self.latents_dir, f"{img_name}.pt")

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        latent_tensor = torch.load(latent_path)

        image_path = os.path.join('data/imgs', img_file)
        image = Image.open(image_path).convert("RGB")
        image_np_arr = np.array(image)

        # Apply Canny edge detection
        canny_image = cv2.Canny(image_np_arr, 100, 200)
        canny_image = np.concatenate([canny_image[:, :, None]] * 3, axis=2)
        canny_image = Image.fromarray(canny_image)

        canny_image_tensor = transforms.ToTensor()(canny_image)

        return {
            'latent': latent_tensor,
            'caption': self.img_caption_map[img_file],
            'canny_img': canny_image_tensor,
        }

if __name__ == "__main__":
    batch_size = 512
    datamodule = LatentDataModule(batch_size=batch_size, num_workers=1)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    for data in train_loader:
        # print(data['latent'].shape, data['latent'].dtype)
        # print(len(data['caption']))
        # print(data['canny_img'].shape, data['canny_img'].dtype)
        print("pass")
    
    for data in test_loader:
        print("test pass")
    
    # for data in test_loader:
    #     print(data['latent'].shape, data['latent'].dtype)
    #     print(len(data['caption']))
    #     print(data['canny_img'].shape, data['canny_img'].dtype)