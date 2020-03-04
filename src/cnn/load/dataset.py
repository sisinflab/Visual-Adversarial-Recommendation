from torch.utils.data import Dataset
from skimage import io
import os

class CustomDataset(Dataset):
    """
    This class represents a dataset. You can load it from the memory (by specifying its path)
    and use it for training/testing purposes.

    Attributes:
        root_dir (str): dataset path
        filenames (list): list of dataset filenames
        num_samples (int): number of samples inside the dataset
        transform: pre processing operations to perform on dataset
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): dataset path
        """
        self.root_dir = root_dir
        self.num_samples = 0
        self.filenames = []
        self.transform = transform

        try:
            self.filenames = os.listdir(self.root_dir)
            self.filenames.sort()
            self.num_samples = len(self.filenames)

        except OSError:
            print("Path: %s does not exist. Provide a valid path." %self.root_dir)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = io.imread(self.filenames[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample