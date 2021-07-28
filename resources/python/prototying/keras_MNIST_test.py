from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# MNIST Dataset
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images  = np.expand_dims(train_images.astype(np.float32) / 255.0, axis=3)
test_images = np.expand_dims(test_images.astype(np.float32) / 255.0, axis=3)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Training parameters
batch_size = 128
n_epochs = 5
n_classes = 10

# Create the model
model = Sequential()

# Convolution Layers
model.add(Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(16, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

# Dense Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

# Optimizer
optimizer = Adam(lr=1e-4)

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

# Train model
model.fit(train_images,
          train_labels,
          epochs=n_epochs,
          batch_size=batch_size,
          validation_data=(test_images, test_labels)
          )

# Show Sample Predictions
predictions = model.predict(test_images[:25])
predictions = np.argmax(predictions, axis=1)
f, axarr = plt.subplots(5, 5, figsize=(25,25))
for idx in range(25):
    axarr[int(idx/5), idx%5].imshow(np.squeeze(test_images[idx]), cmap='gray')
    axarr[int(idx/5), idx%5].set_title(str(predictions[idx]),fontsize=50)
