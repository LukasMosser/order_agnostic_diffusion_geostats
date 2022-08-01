"""import numpy as np
import torch
from tqdm.notebook import tqdm
import matplotlib.animation as animation
from IPython.display import HTML


conditioning_data = np.zeros((32, 32)).astype(int)
conditioning_data[16:16 + 1, 16:16 + 1] = 1


flattened_img = conditioning_data.flatten()
conditioning_indices = np.argwhere(flattened_img > 0)[:, 0]


sigmas = []
for _ in range(batch_size):
    sigma_range = np.arange(w * h)
    np.random.shuffle(sigma_range)

    sigma_range[conditioning_indices] = np.arange(len(conditioning_indices))
    sigma_conditioned = sigma_range
    sigmas.append(sigma_conditioned)

sigmas = np.array(sigmas)


fig, ax = plt.subplots(6, 6, figsize=(8, 8))
for a, img in zip(ax.flatten(), sigmas):
    a.imshow(img.reshape(32, 32), cmap="inferno", vmin=0, vmax=w * h, interpolation='none')
    a.set_axis_off()
plt.show()


t_range_start = np.sum(flattened_img)
sigma_conditioned = torch.from_numpy(sigmas).to(device)

realization = torch.from_numpy(conditioning_data).view(1, 1, h, w).to(device)
realization = torch.cat([1 - realization, realization], dim=1).float()

plt.imshow(realization.cpu()[0, 1])
plt.show()


model.eval()

realizations = []
entropys = []

sampled_random_path = sigma_conditioned
idx_range = torch.arange(start=t_range_start, end=w * h, step=1, device=device, requires_grad=False)

progress_bar = tqdm(idx_range, total=len(idx_range))
for idx in progress_bar:
    mask = create_mask_at_random_path_index(sampled_random_path, idx)

    sampling_location_mask = create_sampling_location_mask(sampled_random_path, idx, w, h)

    with torch.inference_mode():
        conditional_prob = predict_conditional_prob(realization, model, mask, idx)

    sampled_realization = sample_from_conditional(conditional_prob)
    realization = insert_predicted_value_at_sampling_location(realization, sampled_realization, sampling_location_mask)
    conditional_entropy = compute_entropy(conditional_prob)

    realizations.append(torch.argmax(realization, dim=1).cpu().numpy())
    entropys.append(conditional_entropy.cpu().numpy())

    progress_bar.set_description("Generating Sample. Step: {0:}".format(idx))

    if idx % 50 == 0:
        plt.imshow(torch.argmax(realization, dim=1).cpu()[0])
        plt.show()


fig, ax = plt.subplots(6, 6)
for a, img in zip(ax.flatten(), realizations[-1]):
    a.imshow(img.reshape(w, h))
    a.set_axis_off()
plt.show()

fig = plt.figure(figsize=(6, 6))
plt.axis("off")
ims = [[plt.imshow(np.reshape(i[0], (w, h)), animated=True, cmap='gray', vmin=0, vmax=1)] for i in realizations]
ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


fig = plt.figure(figsize=(6, 6))
plt.axis("off")
ims = [[plt.imshow(np.reshape(i[0], (w, h)), animated=True, cmap='gray', vmin=0, vmax=1)] for i in entropys]
ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


conditioning_data = np.zeros((64, 64)).astype(int)
conditioning_data[10, 10] = 1
conditioning_data[10, 45] = 0
conditioning_data[45, 45] = 1
conditioning_data[45, 10] = 0

flattened_img = conditioning_data.flatten()
conditioning_indices = np.argwhere(flattened_img > 0)[:, 0]

sigmas = []
for _ in range(batch_size):
    sigma_range = np.arange(w * h)
    np.random.shuffle(sigma_range)

    sigma_range[conditioning_indices] = np.arange(len(conditioning_indices))
    sigma_conditioned = sigma_range
    sigmas.append(sigma_conditioned)

sigmas = np.array(sigmas)

fig, ax = plt.subplots(6, 6, figsize=(8, 8))
for a, img in zip(ax.flatten(), sigmas):
    a.imshow(img.reshape(w, h), cmap="inferno", vmin=0, vmax=w * h, interpolation='none')
    a.set_axis_off()
plt.show()

t_range_start = np.sum(flattened_img)
sigma_conditioned = torch.from_numpy(sigmas).to(device)

realization = torch.from_numpy(conditioning_data).view(1, 1, h, w).to(device)
realization = torch.cat([1 - realization, realization], dim=1).float()

plt.imshow(realization.cpu()[0, 1])
plt.show()
model.eval()

realizations = []
entropys = []

sampled_random_path = sigma_conditioned
idx_range = torch.arange(start=t_range_start, end=w * h, step=1, device=device, requires_grad=False)

progress_bar = tqdm(idx_range, total=len(idx_range))
for idx in progress_bar:
    mask = create_mask_at_random_path_index(sampled_random_path, idx)

    sampling_location_mask = create_sampling_location_mask(sampled_random_path, idx, w, h)

    with torch.inference_mode():
        conditional_prob = predict_conditional_prob(realization, model, mask, idx)

    sampled_realization = sample_from_conditional(conditional_prob)
    realization = insert_predicted_value_at_sampling_location(realization, sampled_realization, sampling_location_mask)
    conditional_entropy = compute_entropy(conditional_prob)

    realizations.append(torch.argmax(realization, dim=1).cpu().numpy())
    entropys.append(conditional_entropy.cpu().numpy())

    progress_bar.set_description("Generating Sample. Step: {0:}".format(idx))

    if idx % 500 == 0:
        plt.imshow(torch.argmax(realization, dim=1).cpu()[0])
        plt.show()

"""
