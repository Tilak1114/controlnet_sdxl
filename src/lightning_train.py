
import argparse
import os

import torch
import torch.distributed as dist
from lightning.fabric import Fabric
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
from sdxl_module import SDXLModule
from data.data_module import DataModule
from tqdm import tqdm

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

def is_master_process():
    """Check if the current process is the master process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def main() -> None:
    args = parse_args()
    
    data_module = DataModule(batch_size=args.train_batch_size)

    strategy = DeepSpeedStrategy(stage=2)

    fabric = Fabric(
        accelerator="gpu", 
        devices=[0],
        strategy="ddp",
        precision="16-true"
        )  # Enable mixed precision training
    
    fabric.launch()

    with fabric.init_module():
        model = SDXLModule(args)

    optimizer = model.configure_optimizers()
    global_step = 0

    model, optimizer = fabric.setup(model, optimizer)
    
    train_dataloader = data_module.train_dataloader()
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    test_sample = next(iter(data_module.test_dataloader()))

    start_epoch = 0
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
                global_step += 1

                fabric.barrier()

            fabric.call("on_train_batch_end", loss, batch, batch_idx)
                
        fabric.call("on_train_epoch_end")
        
        if is_master_process() and epoch % args.test_every_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                model._forward_module.generate(
                    test_sample['caption'],
                 control_net_input=test_sample['hint'],
                 prefix = f'epoch_{epoch}',
                )
            model.train()
        
        if is_master_process() and epoch % args.save_checkpoint_every_n_epochs == 0:
            print("Saving checkpoint...")
            state = {
                "control_net": model._forward_module.controlnet.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "hparams": args,
            }
            torch.save(state, f"{args.checkpoint_dir}/controlnet-{epoch}.ckpt")
        
        fabric.barrier()

if __name__ == "__main__":
    main()