from model.KerasCNN import KerasCNN
from utils.GujaratiOCRDataset import GujaratiOCRDataset


def get_model(image_sample, num_categories):
    return KerasCNN(input_shape=image_sample,
                    num_conv_layers=2,
                    output_categories=num_categories)


def get_data_set():
    return GujaratiOCRDataset()


if __name__ == '__main__':
    data_set = get_data_set()
    cnn_model = get_model(data_set.training.element_spec[0].shape[1:], len(data_set.training.class_names))
    print(cnn_model.model.summary())
    cnn_model.train(data_set.training, data_set.valid)
    print("------------> training finished")
    cnn_model.test(data_set.testing)
    print("------------> testing finished")
    cnn_model.model.save("./model_saved/model.keras")
    cnn_model.model.save_weights("./model_saved/model.weights.h5")
    print("finished saving")
