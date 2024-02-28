"""contains functionality for creating Pytorch's DataLoader;s for image classification data """

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

NUM_WORKERS=os.cpu_count()

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transforms:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS):
    """
    Creates training and testing DataLoader
    Takes in a training directory and a test directory and turns then into 
    pytorch datasets and then into pytorch DataLoaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to test directory
        transforms: torchvision transforms to perform on training and testing data
        batch_size: Number of samples per batch in each of DataLoaders
        num_workers: an integer number for number of workers per DataLoader.

    Returns:
        A Tuple of (train_dataloader, test_dataloader, class_names).
        where class_names is the list of the target classes
        Example usage:
            train_dataloader, test_dataloader, class_names=create_dataloaders(train_dir=path/to/train_dir,test_dir=path/to/test_dir, transforms=some transforms, batch_size=32, num_workers=os.cpu_count())
            
    """
    # use ImageFolder to create dataset(s)
    train_data=ImageFolder(root=train_dir, transform=transforms, target_transform=None)
    test_data=ImageFolder(root=test_dir, transform=transforms, target_transform=None)

    # get class names
    class_names=train_data.classes

    # turn images into Pytorch's DataLoader
    train_dataloader=DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_dataloader=DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,num_workers=os.cpu_count(),pin_memory=True)

    return train_dataloader, test_dataloader, class_names

    

