import os
from torch.utils import data
from PIL import Image


# we use the Torch Dataset class to handle the data
class Dataset(data.Dataset):
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
