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
        self.arr_samples = np.array()
        self.arr_images = np.array()
        self.arr_labels = np.array()
        for _, folders, _ in os.walk(path_to_dataset):
            for _, folder, _ in os.walk(folders):
                label = folder.name
                for _, _, files in os.walk(folder):
                    for data_file in files:
                        img = self.read_image(data_file)
                        ids = ImageDataClass(img, label)
                        np.append(self.arr_samples, ids, axis=0)
                        np.append(self.arr_images, img)
                        np.append(self.arr_labels, label)

    def __getattr__(self, images):
        return self.arr_images

    def __getattr__(self, labels):
        return self.arr_labels

    def read_image(self, data):
        return cv2.imread(data)
