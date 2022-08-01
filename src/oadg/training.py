import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from tqdm.notebook import tqdm
import numpy as np


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


def sample(model,
           w=64, h=64,
           batch_size=36,
           idx_range_start=0,
           random_path=None,
           realization=None,
           device='cuda'):
    if random_path is None and realization is None:
        sampled_random_path = sample_random_path(batch_size, w, h, device=device)
        realization = torch.zeros(batch_size, 2, w, h, requires_grad=False, device=device).float()

    idx_range = torch.arange(start=idx_range_start, end=w * h, step=1, device=device, requires_grad=False)

    progress_bar = tqdm(idx_range, total=len(idx_range))
    for idx in progress_bar:
        mask = create_mask_at_random_path_index(sampled_random_path, idx)

        sampling_location_mask = create_sampling_location_mask(sampled_random_path, idx, w, h)
        with torch.inference_mode():
            conditional_prob = predict_conditional_prob(realization, model, mask, idx)

        sampled_realization = sample_from_conditional(conditional_prob)
        realization = insert_predicted_value_at_sampling_location(realization, sampled_realization,
                                                                  sampling_location_mask)

        progress_bar.set_description("Generating Sample. Step: {0:}".format(idx))

    realization = torch.argmax(realization, dim=1, keepdim=True).cpu()
    return realization


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

    random_path_mask = create_mask_at_random_path_index(sampled_random_path.view(-1, w, h), idx)

    conditional_prob = predict_conditional_prob(realization, model, random_path_mask, idx)

    log_prob = log_prob_of_realization(conditional_prob, realization)
    log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, random_path_mask)
    log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, w, h)
    loss = compute_average_loss_for_batch(log_prob_weighted)

    return loss


def train(model, optimizer, lr_scheduler, train_dataloader,
          accelerator,
          ema, epochs, max_grad_norm, path, fname,
          save_every, device='cuda'):

    realizations = []
    global_step = 0
    for epoch in range(epochs):

        if epochs % save_every == 0:
            torch.save(accelerator.unwrap_model(ema.averaged_model).state_dict(),
                       path + "/" + fname + "_step_{0:}.pth".format(global_step))

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (realization, _) in progress_bar:
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

    return realizations
