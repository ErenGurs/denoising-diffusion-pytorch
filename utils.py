import os
import torch
import torchvision
#from PIL import Image
#from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def get_data(args):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Resize(160),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        #torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader