from data_module import DataModule
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import os

# Directory to save the final latents
latents_save_dir = 'data/latents'
os.makedirs(latents_save_dir, exist_ok=True)

# Initialize the DataModule
data_module = DataModule(batch_size=24)

# Load the VAE model
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder='vae',
).to('cuda:1')

# Function to encode a batch of images
def encode_image(image):
    with torch.no_grad():
        image = image.to(device=vae.device)
        latent_dist = vae.encode(image).latent_dist
        latents = latent_dist.sample() * vae.config.scaling_factor
    return latents

# List to hold all latents
all_latents = []

# Loop through batches and encode images
for batch in tqdm(data_module.train_dataloader(), desc="Processing Images"):
    images = batch['image']

    # Encode the batch of images
    latents = encode_image(images).cpu()

    # Append the latents to the list
    all_latents.append(latents)

# Concatenate all the latents into a single tensor
final_latents = torch.cat(all_latents, dim=0)

# Save the final concatenated latents
save_path = os.path.join(latents_save_dir, "latent.pt")
torch.save(final_latents, save_path)

print(f"Final latents saved to {save_path}")
