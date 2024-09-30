from keras import models, layers, losses


class KerasCNN:
    def __init__(self, input_image_size, input_channels, num_conv_layers):
        self.model = models.Sequencial()
        self.add_input_convolution_layer(16, (input_image_size[0], input_image_size[1], input_channels))
        for i in range(num_conv_layers - 1):
            self.add_convolution_layer(8)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(8, input_dim=4, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def add_input_convolution_layer(self, filters: int, input_shape: tuple(int, int, int)):
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
        self.model.train(training_set)
        self.model.fit(training_set.image, training_set.label, epochs=10,
                       validation_data=(training_set.image, training_set.label))

    def test(self, testing_set):
        self.model.evaluate(testing_set.image, testing_set.label)
