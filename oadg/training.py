import itertools
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


def sample_random_path(batch_size, w, h, device='cuda'):
    # Create a batch of random sampling paths
    random_path = torch.stack([torch.randperm(w * h, device=device, requires_grad=False) for _ in range(batch_size)],
                              axis=0)
    return random_path


def create_mask_at_random_path_index(sampled_random_path, idx, batch_size, w, h):
    # Create a mask that has 1s everywhere where we've sampled, and 0's everywhere else
    mask = (sampled_random_path < idx).view(batch_size, 1, w, h).long()
    return mask


def create_sampling_location_mask(sampled_random_path, idx, w, h):
    # Create a binary mask that as a 1 at the current location where we're sampling, and 0 otherwise
    sampling_location_mask = (sampled_random_path == idx).view(-1, 1, w, h).long()
    return sampling_location_mask


def predict_conditional_prob(realization, model, sampling_location_mask, idx):
    # We paramterize the Categorical Distribution by the output of our neural network.
    # The outputs are a one-hot-categorical distribution from which we can sample
    logits = F.log_softmax(model(sampling_location_mask * realization, idx.view(-1, ))['sample'], dim=1)
    conditional_prob = OneHotCategorical(logits=logits.permute(0, 2, 3, 1))
    return conditional_prob


def sample_from_conditional(conditional_prob):
    # Draw a sample from the categorical distribution
    return conditional_prob.sample().permute(0, 3, 1, 2)


def insert_predicted_value_at_sampling_location(realization, sampled_realization, sampling_location_mask):
    # combine the current state of the realization with the newly predicted value
    realization = (1 - sampling_location_mask) * realization + sampling_location_mask * sampled_realization
    return realization


def compute_entropy(conditional_prob):
    # We can directly compute the entropy of the Categorical Distribution
    return conditional_prob.entropy().unsqueeze(1)


def binary_entropy(p, eps=1e-6):
    h = -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p)

    if np.isnan(h):
        return eps

    return h


def one_hot_realization(realization):
    # We one-hot the binary image as an input to the neural network
    realization = torch.cat([1 - realization, realization], dim=1)
    return realization


def sample_random_index_for_sampling(batch_size, w, h, device='cuda'):
    # Sample a random index where we want to sample next
    idx = torch.randint(low=0, high=w * h + 1, size=(batch_size, 1, 1), device=device, requires_grad=False)
    return idx


def log_prob_of_realization(conditional_prob, realization):
    # Compute the log-probability of a given realization
    log_prob = conditional_prob._categorical.log_prob(torch.argmax(realization, dim=1))
    return log_prob


def log_prob_of_unsampled_locations(log_prob, sampling_location_mask):
    # Compute the total log probability of the unsampled locations, taking sum over log-probs
    log_prob_unsampled = ((1 - sampling_location_mask) * log_prob).sum(dim=(1, 2, 3))
    return log_prob_unsampled


def weight_log_prob(log_prob_unsampled, idx, w, h, ):
    # We compute the average log-probability over the unsampled locations
    log_prob_weighted = 1 / (w * h - idx + 1) * log_prob_unsampled
    return log_prob_weighted


def compute_average_loss_for_batch(log_prob_weighted):
    # We compute a (negative) average over the batch elements to compute an unbiased estimator of the loss
    loss = -log_prob_weighted.mean()
    return loss


def elbo_objective(model, realization, device='cuda'):
    """
    Computing the Evidence Lower Bound (Elbo) Objective for the ARDM model

    We are going to have our model minimize the average negative log-likelihood predicted for
    random unsampled locations in a batch of training examples.

    We want our model to predict a high-likelihood for the true values at those unsampled locations.
    """
    batch_size, _, w, h = realization.size()

    # Get a batch of random sampling paths
    sampled_random_path = sample_random_path(batch_size, w, h, device=device)

    # Sample a set of random sampling steps for each individual training image in the current batch
    idx = sample_random_index_for_sampling(batch_size, w, h, device=device)

    # We create a mask that masks the locations where we assume we've already sampled
    random_path_mask = create_mask_at_random_path_index(sampled_random_path.view(-1, w, h), idx, batch_size, w, h)

    # We predict the conditional probability for the current sampling step for each training image in the batch
    # Image 1: log p(x23 | x22, x21, x20, ..., x1)
    # Image 2: log p(5 | x4, x3, x2, x1)
    conditional_prob = predict_conditional_prob(realization, model, random_path_mask, idx)

    # Evaluate the value of the log probability for the given realization
    log_prob = log_prob_of_realization(conditional_prob, realization)

    # Compute the total log probability of the unsampled locations
    log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, random_path_mask)

    # Compute an average over all the unsampled locations for each image in the batch
    log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, w, h)

    # Compute an average loss i.e. negative average log likelihood over the batch elements
    loss = compute_average_loss_for_batch(log_prob_weighted)

    return loss


def train(model, optimizer, lr_scheduler, train_dataloader,
          accelerator,
          ema, total_steps, max_grad_norm, path: Path, fname,
          save_every, device='cuda'):

    progress_bar = tqdm(range(total_steps), total=total_steps)
    global_step = 0

    # We use an enumeration of the dataloader to continuously sample from the dataloader without epochs
    for realization in itertools.cycle(train_dataloader):

        if isinstance(realization, list):
            # Handle datasets with labels, assumes image first, label second.
            realization = realization[0]

        optimizer.zero_grad()

        if global_step % save_every == 0:
            torch.save(accelerator.unwrap_model(ema.averaged_model).state_dict(),
                       path.joinpath(fname + "_step_{0:}.pth".format(global_step)))

        # One-hot the batch of training images
        realization = one_hot_realization(realization.float())

        # Compute the elbo loss for the batch of training images
        loss = elbo_objective(model, realization, device=device)

        # Backpropagate the loss to the parameters of the model
        accelerator.backward(loss)

        # We clip the norm of the parameters of the model as in the paper
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Increment the global step and learning rate scheduler
        global_step += 1
        lr_scheduler.step()

        # Make a stochastic gradient descent step using the estimated gradients of the loss w.r.t. the parameters
        optimizer.step()

        # Update the exponential moving average of weights as in the publication
        ema.step(model)

        # Log the loss to the monitor
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        accelerator.log(logs, step=global_step)

        progress_bar.set_description(
            "Step {0:}, Loss: {1:.2f}, Learning Rate: {2:.3e}".format(logs['step'], logs['loss'], logs['lr']))
        progress_bar.update(1)

        if global_step == total_steps:
            break
