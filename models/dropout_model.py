from models.model import Model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, BatchNormalization

class DropoutModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        self.model = models.Sequential()

        self.model.add(tf.keras.layers.Rescaling(scale=1./255, offset=0.0))

        self.model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.Dropout(0.2)) 
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(3, activation='softmax'))

        # self.model = <model definition>
        pass
    
    def _compile_model(self):
        # Your code goes here
        self.model.compile(
            optimizer='rmsprop',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        # self.model.compile(<configuration properties>)
        pass
