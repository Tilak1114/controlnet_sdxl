# Create Inference Pipeline
import argparse

from PIL import Image
from sdxl_module import SDXLModule
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0",)
    parser.add_argument("--width", type=int, default=1024,)
    parser.add_argument("--height", type=int, default=1024,)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--guidance_scale", type=float, default=6.0,)
    parser.add_argument("--num_inference_steps", type=int, default=50,)
    return parser.parse_args()

def main():
    args = parse_args()

    #Path to tryonnet checkpoint
    sdxl_module = SDXLModule(args)
    ctrlnet_ckpt = '/data/tilak/projects/diffarch/checkpoints/controlnet-10.ckpt'

    checkpoint = torch.load(
        ctrlnet_ckpt, 
        map_location='cpu',
        )
    sdxl_module.controlnet.load_state_dict(
        checkpoint['controlnet_state_dict'],
        )
    sdxl_module.to(args.device)
    sdxl_module.eval()
    
    image_path = "/data/tilak/projects/diffarch/data/imgs/img_17.jpg"
    prompt = "an interior design of a living room featuring a large sectional couch"

    image = Image.open(image_path)
    image_np_arr = np.array(image)

    # Apply Canny edge detection
    canny_image = cv2.Canny(image_np_arr, 100, 200)
    canny_image = np.concatenate([canny_image[:, :, None]] * 3, axis=2)
    canny_image = Image.fromarray(canny_image)

    canny_image_tensor = transforms.ToTensor()(canny_image)

    with torch.no_grad():
        generations = sdxl_module.generate(
                    prompt=prompt,
                    control_net_input=canny_image_tensor.unsqueeze(0),
                    num_inference_steps=args.num_inference_steps,
                )

    for i in range(generations.shape[0]):
        gen_image = Image.fromarray(generations[i])

        # Concatenate images side by side (horizontally)
        combined_image = Image.new('RGB', (gen_image.width + canny_image.width, gen_image.height))
        combined_image.paste(canny_image, (0, 0))
        combined_image.paste(gen_image, (canny_image.width, 0))

        # Save the combined image
        combined_image.save(f'{args.output_dir}/inference_generation_10_{i}.png')


if __name__ == "__main__":
    main()