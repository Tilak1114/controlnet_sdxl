from diffusers import StableDiffusionPipeline
import torch
from data_module import DataModule
from PIL import Image
import numpy as np

def check_latent_functionality(image, vae):
    latents = vae.encode(image.to(vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False,)[0]
    
    return image

if __name__ == '__main__':
    model_path = "/data/tilak/projects/sd-lora/checkpoints/v1-5-pruned.safetensors"
    pipe = StableDiffusionPipeline.from_single_file(
            model_path,
        ).to('cuda')
    vae = pipe.vae

    data_module = DataModule(batch_size=4)

    train_dl = data_module.train_dataloader()

    for batch in train_dl:
        images = batch['image'].to('cuda')
        
        for i, image in enumerate(images):
            original_image = image
            decoded_image = check_latent_functionality(image.unsqueeze(0), vae)
            decoded_image = (decoded_image).clamp(0, 1)
            # Convert tensors to PIL Images
            pil_original_image = Image.fromarray((original_image.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))
            pil_decoded_image = Image.fromarray((decoded_image.squeeze(0).detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))

            # Save the images side by side
            combined_image = Image.new('RGB', (pil_original_image.width + pil_decoded_image.width, max(pil_original_image.height, pil_decoded_image.height)))
            combined_image.paste(pil_original_image, box=(0, 0))
            combined_image.paste(pil_decoded_image, box=(pil_original_image.width, 0))

            filename = f"decoded_{i}.png"
            combined_image.save(filename)
            
            print(f"Saved image {filename}")
        break
