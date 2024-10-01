import os

from utils.ImageDataSet import ImageDataSet
from utils.PathToData import PathToData


class GujaratiOCRDataset:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_path = os.path.join(root_path, "data_set", "Gujarati", "Train")
        testing_path = os.path.join(root_path, "data_set", "Gujarati", "Test")
        self.data_path = PathToData(training_path, testing_path)
        self.training = self.get_training_data_set()
        self.testing = self.get_testing_data_set()

    def get_training_data_set(self):
        return ImageDataSet(self.data_path.path_training_data)

    def get_testing_data_set(self):
        return ImageDataSet(self.data_path.path_testing_data)
