import os
from torch.utils import data
from PIL import Image


class Dataset(data.Dataset):
    """
    This version of the loader assumes that labels is a dict in the shape of {index: {property: value}}
    """
    def __init__(self, data_location, list_IDs, labels, transform):
        self.data_location = data_location
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        filename = '{}.png'.format(ID)

        filepath = os.path.join(self.data_location, filename)
        image = Image.open(filepath)
        X = self.transform(image)
        y = self.labels[ID]

        # print(y)

        return X, y

class TensorLabelDataset(data.Dataset):
    """
    This version of the loader assumes that labels is a two dimensional Torch tensor
    """
    def __init__(self, data_location, list_IDs, labels, transform):
        self.data_location = data_location
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        filename = '{}.png'.format(ID)

        filepath = os.path.join(self.data_location, filename)
        image = Image.open(filepath)
        X = self.transform(image)
        y = self.labels[ID, :]

        # print(y)

        return X, y
