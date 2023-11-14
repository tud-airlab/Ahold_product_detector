# collapse-hide
from pathlib import Path

####### PACKAGES

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import PIL.Image as Image
from tqdm import tqdm


def pil_loader_rgb(path_to_img: Path):
    return Image.open(path_to_img).convert('RGB')


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for root, d_names, f_names in os.walk(data_dir):
            print(root)
            for f in f_names:
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.data.append(Path(root).joinpath(f))
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = pil_loader_rgb(self.data[idx])

        # augmentations
        if self.transform is not None:
            image = self.transform(image)

        return image


####### PARAMS

device = torch.device('cuda:0')
num_workers = 8
image_size = 80
batch_size = 8
data_path = './data/PMF_dataset'

# collapse-show
augs = transforms.Compose([
    transforms.Resize(size=(image_size, image_size), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])
# dataset
image_dataset = CustomDataset(data_dir=data_path,
                              transform=augs)

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

####### COMPUTE MEAN / STD

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])
count = 0

# loop through image_tensors
print("starting loop")
for inputs in tqdm(image_loader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

####### FINAL CALCULATIONS

# pixel count
count = len(image_loader.dataset) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))
