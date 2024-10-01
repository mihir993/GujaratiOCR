from keras import models, layers, losses
from keras import utils as kutils

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
                                 epochs=10,  verbose=True,
                                 validation_data=(training_set.images, self.encode_labels(training_set))
                                 )
        print(history)

    def test(self, testing_set):
        eval = self.model.evaluate(testing_set.images,
                                   self.encode_labels(testing_set))
        print(eval)

    def encode_labels(self, data):
        return self.label_encoder.transform(data.labels.reshape(-1, 1)).toarray()
