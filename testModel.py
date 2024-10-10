from model.KerasCNN import KerasCNN
from keras.api.models import load_model
from utils.GujaratiOCRDataset import GujaratiOCRDataset


def load_saved_model(path_to_model, path_to_weights):
    modl = load_model(path_to_model)
    modl.load_weights(path_to_weights)
    return modl


def get_data_set():
    return GujaratiOCRDataset()


def test(model, testing_set):
    eval = model.evaluate(testing_set,
                               verbose=1,
                               return_dict=True)
    print(eval)


if __name__ == '__main__':
    path_weight = "./model_saved/model.weights.h5"
    path_model = "./model_saved/model.keras"
    cnn_model = load_saved_model(path_model, path_weight)
    print(cnn_model.summary())
    data_set = get_data_set()
    test(cnn_model, data_set.testing)
    print("------------> testing finished")
