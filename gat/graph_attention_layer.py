from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, constraints, regularizers, initializers, \
    activations
import numpy as np

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attention_heads=1,
                 heads_combination='concat',  # ['concat', 'average']
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert heads_combination in ['concat', 'average'], 'Possbile combination methods: \'concat\', \'average\''
        self.F_ = F_
        self.attention_heads = attention_heads  # K
        self.heads_combination = heads_combination
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = False

        if heads_combination == 'concat':
            self.output_dim = self.F_ * self.attention_heads  # Output will be K*F'
        else:
            self.output_dim = self.f_  # Output will be F'

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = self.F = input_shape[-1]

        self.kernels = []
        self.biases = []
        for head in range(self.attention_heads):
            kernel = self.add_weight(shape=(self.F, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                       initializer=self.bias_initializer,
                                       name='bias_%s' % head,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
            else:
                bias = None
            self.biases.append(bias)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super(GraphAttention, self).build(input_shape)

    def call(self, inputs):
        X = inputs  # B x F
        # TODO: N = inputs[1]  # B x F
        B = K.shape(X)[0]  # Get batch size at runtime
        # TODO: N = K.shape(neighbors)[0]

        outputs = []
        for head in range(self.attention_heads):
            kernel = self.kernels[head]
            linear_transf = K.dot(X, kernel)  # B x F'
            # TODO: linear_transformation of neighbors
            # Repeat feature vectors: [[1], [2]] becomes [[1], [1], [2], [2]]
            repeated = K.reshape(K.tile(linear_transf, [1, N]), (-1, self.F_))  # B*N x F'

            # TODO: tile feature vectors of neighbors B times
            # Tile feature vectors of neighbors: [[1], [2]] becomes [[1], [2], [1], [2]]
            tiled = K.tile(linear_transf, [B, 1])  # B*N x F'

            # Build combinations
            combinations = K.concatenate([repeated, tiled])  # N*B x 2F'

            # Compute output features for each node in the batch
            # TODO: change the for loop to a loop over tf.unstack(combinations)
            combination_slices = tf.unstack(K.reshape(combinations, (B, -1, 2 * self.F_)))
            output_features = []
            for slice in combination_slices:
                dense = Dense(1)(slice)  # N x 1 (basically "a(Wh_i, Wh_j)" in the paper)
                # TODO: masking
                e_i = K.reshape(dense, (1, -1))  # 1 x N (e_i in the paper)
                softmax = K.squeeze(K.softmax(e_i))  # N (alpha_i in the paper)
                softmax_broadcast = K.transpose(K.reshape(K.tile(softmax, [self.F_]), [self.F_, -1]))
                node_features = K.sum(softmax_broadcast * linear_transf, axis=0)
                if self.use_bias:
                    output = K.bias_add(node_features, self.bias)
                if self.heads_combination == 'concat' and self.activation is not None:
                    node_features = self.activation(node_features)
                output_features.append(node_features)

            output_features = K.stack(output_features)
            outputs.append(output_features)

        if self.heads_combination == 'concat':
            output = K.concatenate(outputs)
        else:
            output = K.mean(K.stack(outputs), axis=0)
            if self.activation is not None:
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)