from dataclasses import dataclass
import numpy as np
import os
import cv2

@dataclass
class ImageDataClass:
    image: list[list[list[int]]]
    label: str

    def __init__(self, image, label: str):
        self.image = image
        self.label = label


class ImageDataSet:
    def __init__(self, path_to_dataset):
        self.images = np.empty(shape=(0, 32, 32))
        self.labels = np.empty(shape=0)
        for _, folders, _ in os.walk(path_to_dataset):
            for folder in folders:
                label = folder
                print(f"scanning for label {label}")
                folder_path = os.path.join(path_to_dataset, folder)
                for _, _, files in os.walk(folder_path):
                    for data_file in files:
                        file_path = os.path.join(folder_path, data_file)
                        img = self.read_image(file_path)
                        self.images = np.append(self.images, [img], axis=0)
                        self.labels = np.append(self.labels, [label], axis=0)

    def read_image(self, data):
        temp = cv2.imread(data)
        return cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
