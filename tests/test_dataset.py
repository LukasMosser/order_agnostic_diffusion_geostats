import itertools
import torch
from torch.utils.data import DataLoader, TensorDataset
from oadg.dataset import Channels


def test_channels_dataset():
    dataset = Channels(root=".", download=True)
    assert len(dataset) > 0


def test_cycle_dataset():
    X = torch.zeros((10, 1, 32, 32))
    y = torch.zeros((10, 1))
    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset, num_workers=0,
                                  batch_size=5, shuffle=True,
                                  pin_memory=True, drop_last=True)

    cycle = 0
    step = 0
    for _ in itertools.cycle(train_dataloader):
        if step % len(train_dataloader) == 0 and step >= len(train_dataloader):
            cycle += 1
        else:
            print(step, len(train_dataloader))

        if cycle == 2:
            break

        step += 1

    assert cycle > 1

