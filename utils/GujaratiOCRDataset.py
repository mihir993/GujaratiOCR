import os

from utils.PathToData import PathToData
from keras import utils as kutils


class GujaratiOCRDataset:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_path = os.path.join(root_path, "data_set", "Gujarati", "Train")
        testing_path = os.path.join(root_path, "data_set", "Gujarati", "Test")
        self.data_path = PathToData(training_path, testing_path)
        self.training, self.valid = self.get_training_data_set()
        self.testing = self.get_testing_data_set()

    def get_training_data_set(self):
        train, valid = kutils.image_dataset_from_directory(self.data_path.path_training_data,
                                                           labels="inferred",
                                                           label_mode="categorical",
                                                           color_mode="grayscale",
                                                           batch_size=32,
                                                           image_size=(32, 32),
                                                           seed=42,
                                                           validation_split=0.2,
                                                           subset='both',
                                                           verbose=True)
        return train, valid

    def get_testing_data_set(self):
        return kutils.image_dataset_from_directory(self.data_path.path_testing_data,
                                                   labels="inferred",
                                                   label_mode="categorical",
                                                   color_mode="grayscale",
                                                   batch_size=32,
                                                   image_size=(32, 32),
                                                   verbose=True)
