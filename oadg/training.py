import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path


def sample_random_path(batch_size, w, h, device='cuda'):
    random_path = torch.stack([torch.randperm(w * h, device=device, requires_grad=False) for _ in range(batch_size)],
                              axis=0)
    return random_path


def create_mask_at_random_path_index(sampled_random_path, idx, batch_size, w, h):
    mask = (sampled_random_path < idx).view(batch_size, 1, w, h).long()
    return mask


def create_sampling_location_mask(sampled_random_path, idx, w, h):
    sampling_location_mask = (sampled_random_path == idx).view(-1, 1, w, h).long()
    return sampling_location_mask


def predict_conditional_prob(realization, model, sampling_location_mask, idx):
    logits = F.log_softmax(model(sampling_location_mask * realization, idx.view(-1, ))['sample'], dim=1)
    conditional_prob = OneHotCategorical(logits=logits.permute(0, 2, 3, 1))
    return conditional_prob


def sample_from_conditional(conditional_prob):
    return conditional_prob.sample().permute(0, 3, 1, 2)


def insert_predicted_value_at_sampling_location(realization, sampled_realization, sampling_location_mask):
    realization = (1 - sampling_location_mask) * realization + sampling_location_mask * sampled_realization
    return realization


def compute_entropy(conditional_prob):
    return conditional_prob.entropy().unsqueeze(1)


def binary_entropy(p, eps=1e-6):
    h = -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p)

    if np.isnan(h):
        return eps
    return h


def one_hot_realization(realization):
    realization = torch.cat([1 - realization, realization], dim=1)
    return realization


def sample_random_index_for_sampling(batch_size, w, h, device='cuda'):
    idx = torch.randint(low=0, high=w * h + 1, size=(batch_size, 1, 1), device=device, requires_grad=False)
    return idx


def log_prob_of_realization(conditional_prob, realization):
    return conditional_prob._categorical.log_prob(torch.argmax(realization, dim=1))


def log_prob_of_unsampled_locations(log_prob, sampling_location_mask):
    log_prob_unsampled = ((1 - sampling_location_mask) * log_prob).sum(dim=(1, 2, 3))
    return log_prob_unsampled


def weight_log_prob(log_prob_unsampled, idx, w, h, ):
    log_prob_weighted = 1 / (w * h - idx + 1) * log_prob_unsampled
    return log_prob_weighted


def compute_average_loss_for_batch(log_prob_weighted):
    loss = -log_prob_weighted.mean()
    return loss


def elbo_objective(model, realization, device='cuda'):
    batch_size, _, w, h = realization.size()

    sampled_random_path = sample_random_path(batch_size, w, h, device=device)

    idx = sample_random_index_for_sampling(batch_size, w, h, device=device)

    random_path_mask = create_mask_at_random_path_index(sampled_random_path.view(-1, w, h), idx, batch_size, w, h)

    conditional_prob = predict_conditional_prob(realization, model, random_path_mask, idx)

    log_prob = log_prob_of_realization(conditional_prob, realization)
    log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, random_path_mask)
    log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, w, h)
    loss = compute_average_loss_for_batch(log_prob_weighted)

    return loss


def train(model, optimizer, lr_scheduler, train_dataloader,
          accelerator,
          ema, total_steps, max_grad_norm, path: Path, fname,
          save_every, device='cuda'):

    progress_bar = tqdm(range(total_steps), total=total_steps)
    realizations = []
    global_step = 0
    for batch_idx, realization in enumerate(train_dataloader):

        if global_step % save_every == 0:
            torch.save(accelerator.unwrap_model(ema.averaged_model).state_dict(),
                       path.joinpath(fname + "_step_{0:}.pth".format(global_step)))

        realization = one_hot_realization(realization.float())

        optimizer.zero_grad()

        loss = elbo_objective(model, realization, device=device)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        global_step += 1

        lr_scheduler.step()
        optimizer.step()
        ema.step(model)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        accelerator.log(logs, step=global_step)

        progress_bar.set_description(
            "Step {0:}, Loss: {1:.2f}, Learning Rate: {2:.3e}".format(logs['step'], logs['loss'], logs['lr']))
        progress_bar.update(1)

        if global_step == total_steps:
            break

    return realizations
