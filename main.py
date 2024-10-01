# This is a sample Python script.
from model.KerasCNN import KerasCNN
from utils.GujaratiOCRDataset import GujaratiOCRDataset
import numpy as np
from keras import datasets, ops
from sklearn.preprocessing import OneHotEncoder


# Press the green button in the gutter to run the script.
def get_model(image_sample, num_categories):
    return KerasCNN(input_image_sample=image_sample,
                    num_conv_layers=2,
                    output_categories=num_categories)


def get_data_set():
    return GujaratiOCRDataset()


if __name__ == '__main__':
    data_set = get_data_set()
    train_labels = data_set.training.labels
    onehot = OneHotEncoder()
    onehot.fit(train_labels.reshape(-1, 1))

    print(f"dataset: {data_set.training.images.shape}")
    print(f"dataset: {data_set.testing.images.shape}")
    cnn_model = get_model(data_set.training.images[0], np.unique(data_set.training.labels).size)
    cnn_model.label_encoder = onehot
    print(cnn_model.model.summary())
    cnn_model.train(data_set.training)
    print("------------> training finished")
    cnn_model.test(data_set.testing)
    print("------------> testing finished")
    # print(cnn_model.evaluate)
    cnn_model.model.save("./model.keras")
    cnn_model.model.save_weights("./model.weights.h5")
    print("finished saving")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
