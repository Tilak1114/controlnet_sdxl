import pandas as pd
from pytorch_lightning import LightningDataModule
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import re
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers=16, test_size=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

        self.image_dir = 'data/imgs'
        csv_file = '/data/tilak/projects/diffarch/data/final_master.csv'

        # Read CSV
        self.df = pd.read_csv(csv_file)[:30000]

        # Split the dataframe into train and test
        self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=42)

    def train_dataloader(self):
        # Use the train dataframe to create the dataset
        train_dataset = Dataset(self.image_dir, self.train_df)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        # Use the test dataframe to create the dataset
        test_dataset = Dataset(self.image_dir, self.test_df)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df):
        self.image_dir = image_dir
        self.df = df

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Fetch image id and blip caption from the dataframe
        row = self.df.iloc[idx]
        img_id = row['img_id']
        blip_caption = row['blip_caption']

        # Load the image
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        image_np_arr = np.array(image)
        image = self.transform(image)

        # Apply Canny edge detection
        canny_image = cv2.Canny(image_np_arr, 100, 200)
        canny_image = np.concatenate([canny_image[:, :, None]] * 3, axis=2)
        canny_image = Image.fromarray(canny_image)
        canny_image = transforms.ToTensor()(canny_image)

        # Return image, blip caption, and Canny edge result
        result = {
            'image': image,
            'img_id': img_id,
            'caption': blip_caption,
            'hint': canny_image
        }

        return result


def save_image_with_caption(image, canny_image, caption, img_id, output_dir="output"):
    # Convert the tensor images to PIL images
    image = transforms.ToPILImage()(image)
    canny_image = transforms.ToPILImage()(canny_image)
    
    # Combine images side by side
    combined_width = image.width + canny_image.width
    combined_height = max(image.height, canny_image.height) + 30  # Add extra space for the caption
    combined_image = Image.new('RGB', (combined_width, combined_height), (0, 0, 0))
    
    combined_image.paste(image, (0, 30))  # Offset by 30 pixels for caption space
    combined_image.paste(canny_image, (image.width, 30))  # Offset by 30 pixels for caption space
    
    # Add caption as title on top
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()  # Using default font; customize as needed
    
    # Get the bounding box of the text for positioning
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Draw the caption above the image
    draw.text(((combined_width - text_width) / 2, 5), caption, fill="white", font=font)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the combined image
    output_path = os.path.join(output_dir, f"{img_id}_combined.png")
    combined_image.save(output_path)
    print(f"Saved: {output_path}")

def main():
    # Create an instance of the DataModule and Dataset
    batch_size = 4
    data_module = DataModule(batch_size=batch_size)
    
    # Load a few samples from the training dataset
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        for i in range(len(batch['image'])):
            # Get individual images, canny images, and captions
            image = batch['image'][i]
            image = image * 0.5 + 0.5

            canny_image = batch['hint'][i]
            caption = batch['blip_caption'][i]
            img_id = batch['img_id'][i]
            
            # Save the image and canny result side by side
            save_image_with_caption(image, canny_image, caption, img_id)
        
        break  # Just process the first batch for testing


if __name__ == "__main__":
    main()
