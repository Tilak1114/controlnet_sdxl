from sdxl_module import SDXLModule
from safetensors.torch import load_file
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import torchvision
import numpy as np

def main():
    original_image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    ).resize((1024, 1024))

    image = np.array(original_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    canny_image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(canny_image)

    canny_image.save("tempcanny.png")

    control_net_input = torchvision.transforms.ToTensor()(canny_image).unsqueeze(0)

    diff_arch_module = SDXLModule(None).to('cuda')
    img = diff_arch_module.generate("Modern white theme living room interior expensive",
                                    control_net_input=control_net_input)
    img.save('test.png')
    # diff_arch_module.test("a modern living room with minimalist design")

if __name__ == '__main__':
    main()