import numpy as np
from keras import Input
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, Lambda, Dense, Reshape
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


from layers import (
    PrimaryCaps, CapsuleLayer, ReconstructionMask, capsule_length)
from keras.callbacks import (
    TensorBoard, LearningRateScheduler, ModelCheckpoint, CSVLogger)

# config
BATCH_SIZE = 100
LR_DECAY = 0.9
SHIFT = 0.1
EPOCHS = 50
CSV_LOGS_PATH = 'capsnet_log.csv'
TENSORBOAD_LOGS_PATH = 'tensorboard-logs'


def margin_loss(lambda_=0.5, m_plus=0.9, m_minus=0.1):
    def margin(y_true, y_pred):
        loss = K.sum(
            y_true * K.square(K.maximum(0., m_plus - y_pred)) +
            lambda_ * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus)),
            axis=1,
        )
        return loss
    return margin


def reconstruction_loss(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), -1)


def load_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train / 255.

    x_test = x_test.reshape(-1, 28, 28, 1)
    x_test = x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(
        width_shift_range=shift_fraction,
        height_shift_range=shift_fraction)
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while True:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])


def get_capsule_network(input_shape=(28, 28, 1)):
    # encoder network
    input_tensor = Input(shape=input_shape, dtype='float32', name='data')
    conv1 = Conv2D(
        kernel_size=(9, 9), strides=(1, 1), filters=256,
        activation='relu')(input_tensor)
    primary_caps = PrimaryCaps(
        capsule_dim=8, filters=256, kernel_size=(9, 9),
        strides=(2, 2))(conv1)
    capsule_layer = CapsuleLayer(
        output_capsules=10, capsule_dim=16)(primary_caps)
    lengths = Lambda(
        capsule_length, output_shape=(10,), name='digits')(capsule_layer)

    input_mask = Input(shape=(10,), name='mask')
    reconstruction_mask = ReconstructionMask()
    masked_from_labels = reconstruction_mask([capsule_layer, input_mask])
    masked_by_length = reconstruction_mask(capsule_layer)

    # decoder network
    decoder = Sequential(name='decoder')
    decoder.add(Dense(512, activation='relu', input_shape=(160,)))
    decoder.add(Dense(1024, activation='relu'))
    decoder.add(Dense(784, activation='sigmoid'))
    decoder.add(Reshape(input_shape))

    training_model = Model(
        [input_tensor, input_mask], [lengths, decoder(masked_from_labels)])
    inference_model = Model(input_tensor, [lengths, decoder(masked_by_length)])

    return training_model, inference_model


if __name__ == '__main__':

    # load data
    x_train, y_train, x_test, y_test = load_mnist()

    # define callbacks
    lr_schedule = LearningRateScheduler(
        schedule=lambda epoch, lr: lr * LR_DECAY)
    csvlog = CSVLogger(CSV_LOGS_PATH)
    tb = TensorBoard(log_dir=TENSORBOAD_LOGS_PATH, batch_size=BATCH_SIZE)
    checkpoint = ModelCheckpoint(
        'weights-{epoch:02d}.h5',
        monitor='val_digits_acc',
        save_best_only=True,
        save_weights_only=True,
        period=1,
        verbose=1,
    )

    # get and compile the newtork
    capsnet_train, capsnet_inference = get_capsule_network()

    capsnet_train.compile(
        loss={'digits': margin_loss(), 'decoder': reconstruction_loss},
        loss_weights={'digits': 1, 'decoder': 0.0005},
        optimizer='adam',
        metrics={'digits': 'acc'}
    )

    # train
    history = capsnet_train.fit_generator(
        generator=train_generator(x_train, y_train, BATCH_SIZE, SHIFT),
        steps_per_epoch=int(y_train.shape[0] / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=[[x_test, y_test], [y_test, x_test]],
        callbacks=[lr_schedule, csvlog, tb, checkpoint],
    )

    # evaluate
    y_pred, x_reconstruct = capsnet_inference.predict(x_test)
    acc = np.sum(
        np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]

    print(f'Test acc: {acc}')
