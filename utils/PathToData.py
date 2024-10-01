from dataclasses import dataclass


@dataclass
class PathToData:
    path_training_data: str
    path_testing_data: str

    def __init__(self, training_path, testing_path):
        self.path_training_data = training_path
        self.path_testing_data = testing_path
