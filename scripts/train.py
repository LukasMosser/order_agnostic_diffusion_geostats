import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from diffusers.models import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator, utils
from dataclasses import dataclass
from oadg.training import train
from oadg.dataset import Channels


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 64
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 3e-4
    lr_warmup_steps = 500
    weight_decay = 1e-6
    ema_inv_gamma = 1.0
    ema_power = 0.75
    ema_max_value = 0.999
    max_grad_norm = 100.0
    save_every = 10
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0


def main():
    config = TrainingConfig()

    utils.set_seed(config.seed)

    transform = Compose([ToTensor(), lambda x: x > 0.5])
    train_dataset = Channels(download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    model = UNet2DModel(
        sample_size=config.image_size,
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

    accelerator = Accelerator(mixed_precision=config.mixed_precision, log_with='wandb')
    accelerator.init_trackers('order-agnostic-autoregressive-diffusion-channels')
    device = accelerator.device

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(model, optimizer,
                                                                                            train_dataloader,
                                                                                            lr_scheduler)

    ema = EMAModel(model, inv_gamma=config.ema_inv_gamma, power=config.ema_power, max_value=config.ema_max_value)

    train(model, optimizer, lr_scheduler,
          train_dataloader, accelerator, ema,
          config.num_epochs, config.max_grad_norm,
          "./drive/MyDrive/oadg", "channels",
          config.save_every, device=device)


if __name__ == "__main__":
    main()
