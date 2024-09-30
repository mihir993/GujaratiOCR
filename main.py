# This is a sample Python script.
from model.KerasCNN import KerasCNN


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def get_model(image_sample):
    return KerasCNN(input_image_size=image_sample.size[0:2],
                    input_channels=image_sample.size[2],
                    num_conv_layers=2)


class ImageDataSet:
    pass


def get_data_set():
    return ImageDataSet()


if __name__ == '__main__':
    data_set = get_data_set()
    cnn_model = get_model(data_set.trainng.images[0])
    cnn_model.train(data_set.trainng)
    print("------------> training finished")
    cnn_model.test(data_set.test)
    print("------------> testing finished")
    print("")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
