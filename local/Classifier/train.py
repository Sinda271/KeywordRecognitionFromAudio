import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model1.h5"
LEARNING_RATE = 0.000008
EPOCHS = 120
BATCH_SIZE = 32
NUM_KEYWORDS = 5


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs X and targets y
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    return X, y


def get_data_splits(data_path, test_size=0.1, validation_size=0.1):
    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)

    # convert inputs from 2d to 3d arrays
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    # build network
    model = keras.Sequential()
    # conv layer1
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                                  input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))  # down sample the output of the conv

    # conv layer2
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer3
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output and feed it to a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.25))  # regularisation to avoid overfitting

    # output layer
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=['accuracy'])

    # print model overview
    model.summary()

    return model


def main():
    # load train/validation/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)

    # build CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # ( nb of segments, nb of MFCCs = 13, 1)
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[es])

    # evaluate model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"test error: {test_error}, test accuracy: {test_accuracy}")

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
