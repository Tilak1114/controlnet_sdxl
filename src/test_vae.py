
from PIL import Image
from diffusers import AutoencoderKL
import torchvision.transforms as transforms

vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder='vae'
        )
image = Image.open("/data/tilak/projects/diffarch/data/imgs/img_9991.jpg")


transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ],
        )

model_input = vae.encode(transform(image).unsqueeze(0)).latent_dist.sample()
model_input = model_input * vae.config.scaling_factor