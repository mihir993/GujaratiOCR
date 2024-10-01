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
        self.arr_samples = np.empty(shape=0)
        self.images = np.empty(shape=(0, 32, 32, 3))
        self.labels = np.empty(shape=0)
        for _, folders, _ in os.walk(path_to_dataset):
            for folder in folders:
                label = folder
                folder_path = os.path.join(path_to_dataset, folder)
                for _, _, files in os.walk(folder_path):
                    for data_file in files:
                        file_path = os.path.join(folder_path, data_file)
                        img = self.read_image(file_path)
                        ids = ImageDataClass(img, label)
                        self.arr_samples = np.append(self.arr_samples, [ids], axis=0)
                        self.images = np.append(self.images, [img], axis=0)
                        self.labels = np.append(self.labels, [label], axis=0)

    def read_image(self, data):
        return cv2.imread(data)
