import torch
from cleanfid import fid
from matplotlib import pyplot as plt

from torchvision.utils import save_image

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Linearly interpolate the first two dimensions between -1 and 1. 
    # Keep the rest of the z vector for the samples to be some fixed value. 
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    z = torch.randn(100, 128, dtype=torch.half, device=gen.dense.weight.data.device)
    z[:, 0] = torch.linspace(-1, 1, 100, dtype=torch.half, device=gen.dense.weight.data.device)
    z[:, 1] = torch.linspace(-1, 1, 100, dtype=torch.half, device=gen.dense.weight.data.device)

    with torch.cuda.amp.autocast():
        generated_samples = gen.forward_given_samples(z)
    save_image(
        generated_samples.data.float(),
        path,
        nrow=10,
    )
