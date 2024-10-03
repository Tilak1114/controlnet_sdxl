from data_module import DataModule
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import os

latents_save_dir = 'data/latents'
os.makedirs(latents_save_dir, exist_ok=True)

data_module = DataModule(batch_size=2)

# Load the VAE model
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder='vae',
).to('cuda')

def encode_image(image):
    # Move image to GPU and encode
    with torch.no_grad():
        image = image.to(device=vae.device)
        latent_dist = vae.encode(image).latent_dist
        latents = latent_dist.sample() * vae.config.scaling_factor
    return latents

for batch in tqdm(data_module.train_dataloader(), desc="Processing Images"):
    images = batch['image']
    img_ids = batch['img_id']

    # Encode the batch of images
    latents = encode_image(images).cpu()

    # Process each image in the batch
    for latent, img_id in zip(latents, img_ids):
       
        latent = latent.cpu()

        # Save the latent tensor as .pt file
        save_path = os.path.join(latents_save_dir, f"{img_id}.pt")
        torch.save(latent, save_path)
        # print(f"Saved latents for {img_id} at {save_path}")
