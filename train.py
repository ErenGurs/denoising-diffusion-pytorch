import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from utils import *
import argparse
import logging
from tqdm import tqdm

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
).cuda('cuda')  # .cuda('cuda:1')

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_Unconditional"
args.epochs = 500
args.batch_size = 12
args.image_size = 128
args.dataset_path = r"/mnt/task_runtime/ddpm/landscape_img_folder"
#args.device = "cuda:1"
args.device = "cuda"
args.lr = 3e-4

dataloader = get_data(args)


for epoch in range(args.epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(dataloader)
    for i, (training_images, _) in enumerate(pbar):
        training_images = training_images.to(args.device)
        # training_images = torch.rand(8, 3, 128, 128).cuda() # images are normalized from 0 to 1
        loss = diffusion(training_images)
        loss.backward()

    sampled_images = diffusion.sample(batch_size = 4)
    #grid_img_noised = torchvision.utils.make_grid(x_t, nrow=4)
    #torchvision.utils.save_image(grid_img_noised, 'eren_noised.png')
    grid_img = torchvision.utils.make_grid(sampled_images, nrow=4)
    torchvision.utils.save_image(grid_img, f"eren_{epoch}.png")

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
