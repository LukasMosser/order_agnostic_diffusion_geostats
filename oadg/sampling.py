import numpy as np
import torch
from tqdm.auto import tqdm
from oadg.training import create_mask_at_random_path_index, create_sampling_location_mask, predict_conditional_prob
from oadg.training import insert_predicted_value_at_sampling_location, sample_from_conditional


def initalize_empty_realizations_and_paths(batch_size, w, h, device='cpu'):
    sigmas = []
    for _ in range(batch_size):
        sigma_range = np.arange(w * h)
        np.random.shuffle(sigma_range)

        sigma_conditioned = sigma_range
        sigmas.append(sigma_conditioned)

    sigmas = np.array(sigmas)

    t_range_start = 0
    sigma_conditioned = torch.from_numpy(sigmas).to(device)

    realization = torch.zeros((batch_size, 1, h, w)).to(device)
    return t_range_start, sigma_conditioned, realization


def make_conditional_paths_and_realization(conditioning_data, batch_size=16, device='cpu'):
    w, h = conditioning_data.shape

    flattened_img = (conditioning_data.flatten() / 255.).astype(int)
    conditioning_indices = np.argwhere(flattened_img > 0)[:, 0]

    sigmas = []
    for _ in range(batch_size):
        sigma_range = np.arange(w * h)
        np.random.shuffle(sigma_range)

        sigma_range[conditioning_indices] = np.arange(len(conditioning_indices))
        sigma_conditioned = sigma_range
        sigmas.append(sigma_conditioned)

    sigmas = np.array(sigmas)

    t_range_start = np.sum(flattened_img)
    sigma_conditioned = torch.from_numpy(sigmas).to(device)

    realization = torch.from_numpy(conditioning_data).view(1, 1, h, w).to(device)
    return t_range_start, sigma_conditioned, realization


def sample(model, image_size: int = 32, batch_size: int = 16,
           realization=None, t_range_start=None, sigma_conditioned=None, device='cpu'):
    print(realization.shape, t_range_start)
    w, h = image_size, image_size
    if realization is not None:
        w, h = realization.size()[-2:]

    realization = torch.cat([1 - realization, realization], dim=1).float()

    model.eval()

    sampled_random_path = sigma_conditioned
    idx_range = torch.arange(start=t_range_start, end=w * h, step=1, device=device, requires_grad=False)

    for idx in tqdm(idx_range):
        mask = create_mask_at_random_path_index(sampled_random_path, idx, batch_size, w, h)

        sampling_location_mask = create_sampling_location_mask(sampled_random_path, idx, w, h)

        with torch.inference_mode():
            conditional_prob = predict_conditional_prob(realization, model, mask, idx)

        sampled_realization = sample_from_conditional(conditional_prob)
        realization = insert_predicted_value_at_sampling_location(realization, sampled_realization,
                                                                  sampling_location_mask)

    return torch.argmax(realization, dim=1).cpu().numpy()
