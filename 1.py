import keras
import numpy as np
from keras.datasets import mnist
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape, batch_size=2),
        layers.Flatten(),
        layers.Dense(28 * 28, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Define callbacks
checkpoint_cb = ModelCheckpoint("best_model_first.keras", save_best_only=True)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
tensorboard_cb = TensorBoard(log_dir="logs", histogram_freq=1)
lr_scheduler_cb = ReduceLROnPlateau(factor=0.5, patience=5)

optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

try:
    # Train the model with callbacks
    history = model.fit(x_train, y_train,
                        epochs=30,
                        validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb, lr_scheduler_cb])
except [Exception]:
    print("Training interrupted.")
    pass

# Evaluate on the test set
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("multiclass_classification_model.keras")
