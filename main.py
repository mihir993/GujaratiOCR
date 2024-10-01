# This is a sample Python script.
from model.KerasCNN import KerasCNN
from utils.GujaratiOCRDataset import GujaratiOCRDataset
import numpy as np


# Press the green button in the gutter to run the script.
def get_model(image_sample, num_categories):
    return KerasCNN(input_image_size=image_sample.shape[0:2],
                    input_channels=image_sample.shape[2],
                    num_conv_layers=2,
                    output_categories=num_categories)


def get_data_set():
    return GujaratiOCRDataset()


if __name__ == '__main__':
    data_set = get_data_set()
    print(f"dataset: {data_set.training.images.shape}")
    print(f"dataset: {data_set.testing.images.shape}")
    cnn_model = get_model(data_set.training.images[0], np.unique(data_set.training.labels).size)
    print(cnn_model.model.summary())
    # cnn_model.train(data_set.training)
    # print("------------> training finished")
    # cnn_model.test(data_set.test)
    # print("------------> testing finished")
    # print("")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
