import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import os
import shutil
from pathlib import Path

def download_and_extract_dataset(zip_file_path="pizza_steak_sushi.zip", 
                                 target_dir=os.path.join('data', 'pizza_steak_sushi')):

  zip_file = Path(zip_file_path)
  data_dir = Path(target_dir)

  # Create the target directory if it doesn't exist
  data_dir.mkdir(parents=True, exist_ok=True)

  # Unzip the file
  shutil.unpack_archive(zip_file, data_dir)

  train_dir = data_dir / 'train'
  test_dir = data_dir / 'test'
  classes = ['pizza', 'steak', 'sushi']
  return classes, train_dir, test_dir

def visualize_batch(batch, classes):
  X, y = batch
  n_samples = len(X)
  grid_size = np.sqrt(n_samples)
  plt.figure(figsize=(16,16))
  
  for sample_index in range(n_samples):
    plt.subplot(grid_size, grid_size, sample_index+1)
    plt.imshow(X[sample_index].permute(1,2,0))
    plt.title(classes[y[sample_index]])

def visualize_sample(sample_id, dataset):
  plt.imshow(dataset[sample_id][0].permute(1,2,0))
  plt.axis('off')
  plt.title(dataset.classes[dataset[sample_id][1]])

def create_datasets(train_dir, test_dir, train_transforms=None, test_transforms=None):
  if train_transforms == None:
    train_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31)
    ])
  
  if test_transforms == None:
    test_transforms = transforms.Compose([
        transforms.Resize((64,64))
    ])

  train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
  test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
  return train_data, test_data

def create_dataloaders(train_dataset, test_dataset, batch_size=32, num_workers=os.cpu_count()):
  train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  return train_dl, test_dl

def run_all(batch_size=32, num_workers=os.cpu_count(), train_transforms=None, test_transforms=None):
  classes, train_dir, test_dir = download_and_extract_dataset()
  train_dataset, test_dataset = create_datasets(train_dir, test_dir, train_transforms, test_transforms)
  train_dl, test_dl = create_dataloaders(train_dataset, test_dataset, batch_size, num_workers)
  return train_dl, test_dl, classes
