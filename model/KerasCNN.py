from keras import models, layers, losses, callbacks
from keras import utils as kutils
import matplotlib.pyplot as plt

kutils.set_random_seed(740)


class KerasCNN:
    def __init__(self, input_shape, num_conv_layers, output_categories):
        self.label_encoder =None
        self.model = models.Sequential()
        self.add_input_convolution_layer(12, input_shape)
        self.add_convolution_layer(36)
        self.add_convolution_layer(72)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(750, activation='relu'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(500, activation='relu'))
        self.model.add(layers.Dropout(0.2))
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

    def train(self, training_set, validation_set=None):
        if not validation_set:
            validation_set = training_set

        callback = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )
        history = self.model.fit(training_set,
                                 epochs=25,  verbose=1, batch_size=15,
                                 validation_data=validation_set,
                                 callbacks=[callback]
                                 )
        print(history)
        self.plot_training_history(history)

    def test(self, testing_set):
        eval = self.model.evaluate(testing_set,
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
        plt.savefig("../training_progress.png")
        # plt.show()
