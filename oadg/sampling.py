import numpy as np
import torch
from tqdm.auto import tqdm
from oadg.training import create_mask_at_random_path_index, create_sampling_location_mask, predict_conditional_prob
from oadg.training import insert_predicted_value_at_sampling_location, sample_from_conditional


def initialize_empty_realizations_and_paths(batch_size, w, h, device='cpu'):
    """
    We create the necessary sampling paths, each random for each bach element.
    """
    random_paths = []
    for _ in range(batch_size):

        # Linear increasing index steps for sampling paths.
        random_path = np.arange(w * h)

        # Shuffle the index array to create a sampling path throughout the realization
        np.random.shuffle(random_path)

        random_paths.append(random_path)

    random_paths = np.array(random_paths)

    # We start from an empty realization so we start at index 0 for generating samples x1 \sim p(x_1)
    idx_start = 0
    random_paths = torch.from_numpy(random_paths).to(device)
    realization = torch.zeros((batch_size, 1, h, w)).to(device)
    return idx_start, random_paths, realization


def make_conditional_paths_and_realization(conditioning_data, batch_size=16, device='cpu'):
    w, h = conditioning_data.shape

    # We turn the conditioning data into a vector
    flattened_img = conditioning_data.flatten()

    # And we find those locations where we have conditioning data (only foreground supported right now)
    conditioning_indices = np.argwhere(flattened_img > 0)[:, 0]
    unconditioned_indices = np.argwhere(flattened_img < 1)[:, 0]


    # Generate random paths for each batch element
    random_paths = []
    for _ in range(batch_size):
        # We need to take into account that we're gonna
        random_path = np.arange(len(conditioning_indices), w * h)
        np.random.shuffle(random_path)

        random_path_grid = np.zeros((w, h)).reshape(-1)

        # Where we have the conditioning data we set the indices to a range of n_0 to n_conditioning data
        random_path_grid[conditioning_indices] = np.arange(len(conditioning_indices))
        random_path_grid[unconditioned_indices] = random_path
        random_paths.append(random_path_grid)

    random_paths = np.array(random_paths)

    idx_start = np.sum(flattened_img)
    random_paths = torch.from_numpy(random_paths).to(device)

    realization = torch.from_numpy(conditioning_data).view(1, 1, h, w).to(device)
    return idx_start, random_paths, realization


def sample(model, image_size: int = 32, batch_size: int = 16,
           realization=None, idx_start=0, random_paths=None, device='cpu'):
    model.eval()

    w, h = image_size, image_size
    if realization is not None:
        w, h = realization.size()[-2:]

    realization = torch.cat([1 - realization, realization], dim=1).float()

    idx_range = torch.arange(start=idx_start, end=w * h, step=1, device=device, requires_grad=False)

    for idx in tqdm(idx_range):
        mask = create_mask_at_random_path_index(random_paths, idx, batch_size, w, h)

        sampling_location_mask = create_sampling_location_mask(random_paths, idx, w, h)

        with torch.inference_mode():
            conditional_prob = predict_conditional_prob(realization, model, mask, idx)

        sampled_realization = sample_from_conditional(conditional_prob)
        realization = insert_predicted_value_at_sampling_location(realization, sampled_realization,
                                                                  sampling_location_mask)

    return torch.argmax(realization, dim=1).cpu().numpy()
