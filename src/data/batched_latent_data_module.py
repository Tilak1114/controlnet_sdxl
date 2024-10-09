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
    def __init__(self, batch_size=4, num_workers=8, test_size=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

        self.latent_file = '/data/tilak/projects/diffarch/data/latents/latent.pt'
        self.caption_csv = '/data/tilak/projects/diffarch/data/final_master.csv'
        self.img_dir_path = '/data/tilak/projects/diffarch/data/imgs'

        self.df = pd.read_csv(self.caption_csv)[:30000]
        
        # Split into train and test
        self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=42)
        
        # Load the latent tensor once
        print("Loading precomputed latents...")
        self.latent_tensor = torch.load(self.latent_file)

    def train_dataloader(self):
        train_dataset = LatentDataset(self.train_df, self.latent_tensor, self.img_dir_path)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        test_dataset = LatentDataset(self.test_df, self.latent_tensor, self.img_dir_path)
        return torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, df, latent_tensor, img_dir_path):
        self.df = df
        self.latent_tensor = latent_tensor
        self.img_dir_path = img_dir_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract the corresponding latent from the tensor
        latent_idx = idx  # Assuming latents and csv are aligned
        latent = self.latent_tensor[latent_idx]

        # Load and process the image
        image_path = f"{self.img_dir_path}/{row['img_id']}.png"
        image = Image.open(image_path).convert("RGB")
        image_np_arr = np.array(image)

        # Apply Canny edge detection
        canny_image = cv2.Canny(image_np_arr, 100, 200)
        canny_image = np.stack([canny_image] * 3, axis=-1)  # Make it 3-channel
        canny_image = Image.fromarray(canny_image)

        # Convert to tensor
        canny_image_tensor = transforms.ToTensor()(canny_image)

        return {
            'latent': latent,
            'caption': row['blip_caption'],  # Assuming caption is in 'blip_caption'
            'canny_img': canny_image_tensor,
        }


if __name__ == "__main__":
    batch_size = 512
    datamodule = LatentDataModule(batch_size=batch_size, num_workers=1)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # Iterate over train loader
    for data in train_loader:
        print(data['latent'].shape, data['latent'].dtype)
        print(len(data['caption']))
        print(data['canny_img'].shape, data['canny_img'].dtype)
        print("train pass")

    # Iterate over test loader
    for data in test_loader:
        print("test pass")
