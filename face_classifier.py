import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

print("Designing model...")
model = tf.keras.models.Sequential([
    # Convolutional neural network
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])

print("Compiling model...")
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])

print("Getting training and validation data ready...")
print("Data from https://hackernoon.com/binary-face-classifier-using-pytorch-2d835ccb7816")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_gen = train_datagen.flow_from_directory(
    os.getcwd()+'/training-data/',
    target_size=(150,150),
    batch_size=128,
    class_mode = 'binary'
)

validation_gen = validation_datagen.flow_from_directory(
    os.getcwd()+'/validation-data',
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary'
)

print("Training model and calculating its accuracy with validation data...")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=10,
    epochs=8,
    verbose=1,
    validation_data= validation_gen,
    validation_steps = 8
)

print("Testing model...")
for index, fileName in enumerate(os.listdir(os.getcwd()+'/test-data')):
    fullpathFileName = os.getcwd() + "/test-data/" + fileName
    img = image.load_img(fullpathFileName, target_size=(150,150))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    images = np.vstack([img_arr])
    classes = model.predict(images,batch_size=10)
    if( classes[0] > 0.5):
        print(fileName + ' [NOT FACE]')
    else:
        print(fileName + ' [FACE]')