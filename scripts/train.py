import wandb
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from diffusers.models import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator, utils
import argparse
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import Repository
from oadg.training import train
from oadg.dataset import Channels


load_dotenv()
wandb.login(key=os.environ['WANDB_KEY'])


def main(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed(args.seed)

    if args.dataset == 'MNIST':
        transform = Compose([ToTensor(), Resize(args.image_size), lambda x: x > 0.5])
        train_dataset = Channels(root=args.data_root, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, num_workers=args.workers,
                                      batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)

    elif args.dataset == 'Channels':
        transform = Compose([ToTensor(), lambda x: x > 0.5])
        train_dataset = Channels(root=args.data_root, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, num_workers=args.workers,
                                      batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    else:
        return NotImplementedError

    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 128),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    accelerator = Accelerator(mixed_precision=args.precision, log_with=['wandb'])
    accelerator.init_trackers(args.wandb_project_name)
    device = accelerator.device

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer,
                                                                           train_dataloader, lr_scheduler)

    ema = EMAModel(model, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_value)

    train(model, optimizer, lr_scheduler,
          train_dataloader, accelerator, ema,
          args.total_steps, args.max_grad_norm,
          args.checkpoint_dir, args.checkpoint_prefix,
          args.save_every, device=device)

    with Repository(args.hf_hub_repository,
                    clone_from='porestar/' + args.hf_hub_repository,
                    use_auth_token=True).commit(commit_message="{0:}_step_{1:}".format(accelerator.trackers[0].run_name,
                                                                                       args.total_steps)):
        torch.save(accelerator.unwrap_model(ema.averaged_model).state_dict(), "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OADG Configuration Parser')
    parser.add_argument('--data-root', default=".", type=Path,
                        help='path to dataset')
    parser.add_argument('--dataset', default="Channels", type=str,
                        choices=['Channels', 'MNIST'],
                        help='path to dataset')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--image-size', default=64, type=int,
                        help='number of data loader workers')
    parser.add_argument('--warmup-steps', default=500, type=int,
                        help='number of data loader workers')
    parser.add_argument('--total-steps', default=1000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--learning-rate', default=3e-4, type=float,
                        help='weight decay')
    parser.add_argument('--max-grad-norm', default=1e2, type=float,
                        help='weight decay')
    parser.add_argument('--ema-inv-gamma', default=1.0, type=float,
                        help='weight decay')
    parser.add_argument('--ema-power', default=0.75, type=float,
                        help='weight decay')
    parser.add_argument('--ema-max-value', default=0.999, type=float,
                        help='weight decay')
    parser.add_argument('--precision', default='no', type=str, choices=['no', 'fp16', 'bf16'],
                        help='weight decay')
    parser.add_argument('--save-every', default=1000, type=int,
                        help='weight decay')
    parser.add_argument('--seed', default=0, type=int,
                        help='weight decay')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        help='path to checkpoint directory')
    parser.add_argument('--checkpoint-prefix', default='channels',
                        help='path to checkpoint directory')
    parser.add_argument('--wandb-project-name', default='order-agnostic-autoregressive-diffusion-channels',
                        help='name of wandb project')
    parser.add_argument('--hf-hub-repository', default='oadg_channels_64',
                        help='name of wandb project')

    args = parser.parse_args()
    main(args)
