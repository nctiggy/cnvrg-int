import tensorflow as tf
import os
import argparse
from tensorflow.keras.optimizers import RMSprop
import datetime
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import preprocessing as keras_pre

help_msg = "This loads in a trained modeal and returns a prediction"
parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument("-e",
                    "--epochs",
                    type=int,
                    help="Number of epochs")
parser.add_argument("-d",
                    "--data_path",
                    type=str,
                    help="Path to the data source")
args = parser.parse_args()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,
                           (3, 3),
                           activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# train_datagen = ImageDataGenerator(rescale=1/255,
#                                   validation_split=0.2)

train_generator = keras_pre.image_dataset_from_directory(
        args.data_path,
        labels="inferred",
        label_mode="binary",
        seed=45,
        image_size=(300, 300),
        class_names=["horses", "humans"],
        batch_size=32,
        validation_split=0.2,
        subset='training')

validation_generator = keras_pre.image_dataset_from_directory(
        args.data_path,
        labels="inferred",
        label_mode="binary",
        image_size=(300, 300),
        seed=45,
        class_names=["horses", "humans"],
        batch_size=32,
        validation_split=0.2,
        subset='validation')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(
      train_generator,
      epochs=args.epochs,
      validation_data=validation_generator,
      verbose=1)

print('cnvrg_tag_test_accuracy: ', history.history['val_acc'][-1])
print('cnvrg_tag_test_loss: ', history.history['val_loss'][-1])
if not os.path.exists('output'):
    os.mkdir('output')
model.save('output/imagizer.model.h5')
