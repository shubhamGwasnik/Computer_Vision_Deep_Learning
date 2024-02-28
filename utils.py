"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import os
from matpolotlib import pyplot as plt
from torch import random
from PIL import Image



def find_classes(directory:str) -> Tuple[list[str], Dict[str, int]]:
    """finds the class folder names in a target directory"""
    # 1. get the class names by scanning the directory
    classes=sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()])

    # 2. raise an error id class names could not be found
    if not classes:
        raise FileNotFoundError(f"couldn't find any class in {directory}...please check file structure.")
    
    #. create a dictionary of class names
    class_to_idx={class_name : i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)


def plot_transformed_images(image_paths: list, transforms, n=3, seed=None):
  """
  Selects random images from the path of images and loads/transforms them then plots original vs transformed version
  """
  if seed:
    random.seed(seed)
  random_image_paths=random.sample(images_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig,ax=plt.subplots(nrows=1, ncols=2)
      #plot original
      ax[0].imshow(f)
      ax[0].set_title(f"original\nSize:{f.size}")
      ax[0].axis("off")
      # plot transformed
      transformed_image=transforms(f) # this changes the shape to c,h,w, but matplotlib supports h,w,c
      transformed_image=transformed_image.permute(1,2,0)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
      ax[1].axis("off")
      # fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary 
    
    Args: 
        results (dict): Dicionary containing list of values eg.
            {"train_loss": [...],
             "train_accuracy": [...],
             "test_loss": [...],
             "test_accuracy": [...]}    
    """
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_accuracy']
    test_accuracy = results['test_accuracy']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
