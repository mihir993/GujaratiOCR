# import os
from dataclasses import dataclass


@dataclass
class PathToData:
    path_training_data: str
    path_testing_data: str

    def __init__(self, training_path, testing_path):
        self.path_training_data = training_path
        self.path_testing_data = testing_path

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# training_path = os.path.join(root_path, "data_set", "Gujarati", "Train")
# testing_path = os.path.join(root_path, "data_set", "Gujarati", "Test")
# data_path = PathToData(training_path, testing_path)
