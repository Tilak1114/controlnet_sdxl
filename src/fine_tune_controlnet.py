from sdxl_module import SDXLModule
from safetensors.torch import load_file
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import torchvision
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--test_every_n_epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024,)
    parser.add_argument("--height", type=int, default=1024,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5, help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--train_batch_size", type=int, default=6,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--num_inference_steps", type=int, default=30,)
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False,)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt/tryon_net/",)
    parser.add_argument("--ckpt_name", type=str, default="")
    parser.add_argument("--save_checkpoint_every_n_epochs", type=int, default=5)

    return parser.parse_args()

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

    args = parse_args()
    diff_arch_module = SDXLModule(args).to('cuda')
    img = diff_arch_module.generate("modern indian style living room interior",
                                    control_net_input=control_net_input)
    img.save('test.png')
    # diff_arch_module.test("a modern living room with minimalist design")

if __name__ == '__main__':
    main()