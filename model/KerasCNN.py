from keras import models, layers, losses
from keras import utils as kutils
import matplotlib.pyplot as plt

kutils.set_random_seed(812)

class KerasCNN:
    def __init__(self, input_image_sample, num_conv_layers, output_categories):
        self.label_encoder =None
        self.model = models.Sequential()
        self.add_input_convolution_layer(16, input_image_sample.shape)
        for i in range(num_conv_layers - 1):
            self.add_convolution_layer(8)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(8, input_dim=4, activation='relu'))
        self.model.add(layers.Dense(output_categories, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=losses.categorical_crossentropy,
                           metrics=['accuracy'])

    def add_input_convolution_layer(self, filters: int, input_shape: tuple([int, int, int])):
        self.model.add(layers.Conv2D(filters=filters,
                                     kernel_size=(2, 2),
                                     activation='relu',
                                     input_shape=input_shape,
                                     ))
        self.model.add(layers.MaxPool2D(2, 2))

    def add_convolution_layer(self, filters: int):
        self.model.add(layers.Conv2D(filters=filters,
                                     kernel_size=(2, 2),
                                     activation='relu'
                                     ))
        self.model.add(layers.MaxPool2D(2, 2))

    def train(self, training_set):
        history = self.model.fit(training_set.images,
                                 self.encode_labels(training_set),
                                 epochs=10,  verbose=1, batch_size=15,
                                 validation_data=(training_set.images, self.encode_labels(training_set))
                                 )
        print(history)
        self.plot_training_history(history)

    def test(self, testing_set):
        eval = self.model.evaluate(testing_set.images,
                                   self.encode_labels(testing_set),
                                   verbose=1,
                                   return_dict=True)
        print(eval)

    def encode_labels(self, data):
        return self.label_encoder.transform(data.labels.reshape(-1, 1)).toarray()

    def plot_training_history(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()
