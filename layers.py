from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.layers import Activation, Conv2D, Lambda, Reshape
from keras.utils.generic_utils import get_custom_objects


def capsule_length(x):
    return K.sqrt(K.sum(K.square(x), axis=-1))


def squash(x):
    l2_norm = K.sum(K.square(x), axis=-1, keepdims=True)
    return l2_norm / (1 + l2_norm) * (x / (K.sqrt(l2_norm + K.epsilon())))


get_custom_objects().update({'squash': Activation(squash)})


def PrimaryCaps(capsule_dim, filters, kernel_size, strides=1, padding='valid'):

    conv2d = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )

    def eval_primary_caps(input_tensor):
        x = conv2d(input_tensor)
        reshaped = Reshape((-1, capsule_dim))(x)
        return Lambda(squash)(reshaped)

    return eval_primary_caps


class CapsuleLayer(Layer):
    def __init__(
            self,
            output_capsules,
            capsule_dim,
            routing_iterations=3,
            kernel_initializer='glorot_uniform',
            activation='squash',
            **kwargs):
        self.output_capsules = output_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = Activation(activation)
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            name='kernel',
            shape=(
                input_shape[1],
                self.output_capsules,
                input_shape[2],
                self.capsule_dim,
            ),
            initializer=self.kernel_initializer,
            trainable=True
        )

        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = K.expand_dims(inputs, axis=2)
        inputs = K.repeat_elements(inputs, rep=self.output_capsules, axis=2)
        U = K.map_fn(
            lambda x: K.batch_dot(x, self.kernel, axes=[2, 2]), inputs)

        # initialize matrix of b_ij's
        input_shape = K.shape(inputs)
        B = K.zeros(
            shape=(input_shape[0], input_shape[1], self.output_capsules))
        for i in range(self.routing_iterations):
            V, B_updated = self._routing_single_iter(B, U, i, input_shape)
            B = B_updated

        return V

    def _routing_single_iter(self, B, U, i, input_shape):
        C = K.softmax(B, axis=-1)
        C = K.expand_dims(C, axis=-1)
        C = K.repeat_elements(C, rep=self.capsule_dim, axis=-1)
        S = K.sum(C * U, axis=1)
        V = self.activation(S)
        # no need to update b_ij's on last iteration
        if i != self.routing_iterations:
            V_expanded = K.expand_dims(V, axis=1)
            V_expanded = K.tile(V_expanded, [1, input_shape[1], 1, 1])
            B = B + K.sum(U * V_expanded, axis=-1)
        return V, B

    def compute_output_shape(self, input_shape):
        return None, self.output_capsules, self.capsule_dim

    def get_config(self):
        config = {
            'output_capsules': self.output_capsules,
            'capsule_dim': self.capsule_dim,
            'routing_iterations': self.routing_iterations,
            'kernel_initializer': self.kernel_initializer,
            'activation': self.activation,
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(**base_config, **config)


class ReconstructionMask(Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) == list and len(inputs) == 2:
            x, mask = inputs[0], inputs[1]
        else:
            x = inputs
            len_x = K.sqrt(K.sum(K.square(x), -1))
            mask = K.one_hot(indices=K.argmax(len_x, 1),
                             num_classes=K.shape(x)[1])

        return K.batch_flatten(x * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_):
        if type(input_) == list and len(input_) == 2:
            input_shape = input_[0]
            return None, input_shape[2]
        else:
            return None, input_[1] * input_[2]

    def get_config(self):
        config = super(ReconstructionMask, self).get_config()
        return config
