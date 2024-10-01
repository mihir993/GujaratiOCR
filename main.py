# This is a sample Python script.
from model.KerasCNN import KerasCNN
from utils.GujaratiOCRDataset import GujaratiOCRDataset


# Press the green button in the gutter to run the script.
def get_model(image_sample):
    return KerasCNN(input_image_size=image_sample.size[0:2],
                    input_channels=image_sample.size[2],
                    num_conv_layers=2)


def get_data_set():
    return GujaratiOCRDataset()


if __name__ == '__main__':
    data_set = get_data_set()
    print(f"dataset: {data_set.training.images.shape}")
    print(f"dataset: {data_set.testing.images.shape}")
    # cnn_model = get_model(data_set.trainng.images[0])
    # cnn_model.train(data_set.trainng)
    # print("------------> training finished")
    # cnn_model.test(data_set.test)
    # print("------------> testing finished")
    # print("")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
