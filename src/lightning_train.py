
import argparse
import os

import torch
import torch.distributed as dist
from lightning.fabric import Fabric
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
from sdxl_module import SDXLModule
from data.data_module import DataModule
from tqdm import tqdm
from torchsummary import summary
from PIL import Image
import torchvision.transforms as T

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--test_every_n_steps", type=int, default=250)
    parser.add_argument("--width", type=int, default=1024,)
    parser.add_argument("--height", type=int, default=1024,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5, help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=2,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=16)
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
    parser.add_argument("--resume_from_checkpoint", type=bool, default=True,)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/",)
    parser.add_argument("--ckpt_name", type=str, default="/data/tilak/projects/diffarch/checkpoints/controlnet-0.ckpt")
    parser.add_argument("--save_checkpoint_every_n_epochs", type=int, default=1)

    return parser.parse_args()

def is_master_process():
    """Check if the current process is the master process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def main() -> None:
    args = parse_args()
    
    data_module = DataModule(batch_size=args.train_batch_size)

    strategy = DeepSpeedStrategy(stage=2)

    fabric = Fabric(
        accelerator="gpu", 
        devices=-1,
        strategy=strategy,
        )  # Enable mixed precision training
    
    fabric.launch()

    with fabric.init_module():
        model = SDXLModule(args)

    # summary(model, input_size=[(4, 128, 128), (1,), (77, 2048), (3, 1024, 1024)])

    checkpoint_path = os.path.join(args.checkpoint_dir, args.ckpt_name) if args.ckpt_name else None

    # Load checkpoint if it exists
    start_epoch = 0
    global_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path) and args.resume_from_checkpoint:
        torch.cuda.empty_cache()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
        # Restore the model states
        model.controlnet.load_state_dict(checkpoint['controlnet_state_dict'])

        # Restore global step and epoch
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    optimizer = model.configure_optimizers()

    model, optimizer = fabric.setup(model, optimizer)
    
    train_dataloader = data_module.train_dataloader()
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    test_dataloader = data_module.test_dataloader()
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    test_sample = next(iter(test_dataloader))

    for epoch in range(start_epoch, args.num_train_epochs):
        fabric.call("on_train_epoch_start")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            fabric.call("on_train_batch_start", batch, batch_idx)

            is_accumulating = batch_idx % args.gradient_accumulation_steps != 0

            loss = model.training_step(batch)
            fabric.backward(loss)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

                fabric.barrier()

                global_step += 1
            
            if is_master_process() and (global_step + 1) % args.test_every_n_steps == 0:
                model.eval()
                with torch.no_grad():
                    generations = model._forward_module.generate(
                        test_sample['caption'],
                    control_net_input=test_sample['hint'],
                    num_inference_steps=args.num_inference_steps,
                    )
                    transform = T.ToPILImage()
                    for i in range(generations.shape[0]):
                        gen_image = Image.fromarray(generations[i])
                        canny_image = transform(test_sample['hint'][i].cpu())

                        # Concatenate images side by side (horizontally)
                        combined_image = Image.new('RGB', (gen_image.width + canny_image.width, gen_image.height))
                        combined_image.paste(canny_image, (0, 0))
                        combined_image.paste(gen_image, (canny_image.width, 0))

                        # Save the combined image
                        combined_image.save(f'{args.output_dir}/step_{global_step}_generation_{i}.png')

                model.train()

            fabric.call("on_train_batch_end", loss, batch, batch_idx)
                
        fabric.call("on_train_epoch_end")
        
        if is_master_process() and epoch % args.save_checkpoint_every_n_epochs == 0:
            print("Saving checkpoint...")
            state = {
                "controlnet_state_dict": model._forward_module.controlnet.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "hparams": args,
            }
            torch.save(state, f"{args.checkpoint_dir}/controlnet-{epoch}.ckpt")
        
        fabric.barrier()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()