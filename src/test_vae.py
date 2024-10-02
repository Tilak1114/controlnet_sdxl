from PIL import Image
from diffusers import AutoencoderKL
import torchvision.transforms as transforms
import torch

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder='vae',
).to('cuda')

image = Image.open("/data/tilak/projects/diffarch/data/imgs/img_9991.jpg")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ],
)

# Encode the image into latent space
image_tensor = transform(image).unsqueeze(0).to(device='cuda')
with torch.no_grad():
    latent_dist = vae.encode(image_tensor).latent_dist
    latents = latent_dist.sample() * vae.config.scaling_factor

# Decode back to image
with torch.no_grad():
    decoded_images = vae.decode(latents/vae.config.scaling_factor, return_dict=False,)[0]

# Post-process the decoded image for saving
decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)  # Scale back from [-1,1] to [0,1]
decoded_images = (decoded_images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")

# Save the decoded image
Image.fromarray(decoded_images[0]).save("test_vae_de.png")
