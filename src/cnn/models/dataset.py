from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
from PIL import Image
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        self.transform = transform
        self.filenames = os.listdir(self.root_dir)
        self.filenames.sort(key=lambda x: int(x.split(".")[0]))
        self.num_samples = len(self.filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = Image.open(self.root_dir + self.filenames[idx])

        try:
            sample.load()
        except:
            print(self.filenames[idx])

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.transform:
            # to_tensor = transforms.ToTensor()
            # sample = to_tensor(sample)
            sample = self.transform(sample)

        return sample, self.filenames[idx]


class CustomDataLoader:
    """
    This class represents a data loader. You can specify the batch size, shuffling mode and other
    parameters.

    Attributes:
        dataset: pytorch-like dataset
        batch_size (int): batch size
        shuffle (bool): True for dataset shuffling, otherwise False.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataloader = DataLoader(dataset=self.dataset,
                                     shuffle=self.shuffle,
                                     batch_size=self.batch_size)
