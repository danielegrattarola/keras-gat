from __future__ import absolute_import

from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import constraints, regularizers, initializers, activations, Dropout


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # ['concat', 'average']
                 attn_dropout=0.5,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 att_kernel_constraint=None,
                 **kwargs):
        assert attn_heads_reduction in ['concat', 'average'], \
            'Possbile reduction methods: \'concat\', \'average\''

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.attn_kernel_constraint = constraints.get(att_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # Attention kernel
            attention_kernel = self.add_weight(shape=(2 * self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               name='att_kernel_%s' % head,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint)
            self.attn_kernels.append(attention_kernel)
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        # Parameters
        N = K.shape(X)[0]  # Number of nodes in the graph

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            repeated = K.reshape(K.tile(linear_transf_X, [1, N]), (N * N, self.F_))  # (N^2 x F')
            tiled = K.tile(linear_transf_X, [N, 1])  # (N^2 x F')
            combinations = K.concatenate([repeated, tiled])  # (N^2 x 2F')
            combination_slices = K.reshape(combinations, (N, -1, 2 * self.F_))  # (N x N x 2F')

            # Attention head
            dense = K.squeeze(K.dot(combination_slices, attention_kernel), -1)  # a(Wh_i, Wh_j) in the paper (N x N x 1)

            # Mask values before activation (Vaswani et al., 2017)
            comparison = tf.equal(A, tf.constant(0, dtype=tf.float32))
            mask = tf.where(comparison, tf.ones_like(A) * -10e9, tf.zeros_like(A))
            masked = dense + mask

            # Feed masked values to softmax
            softmax = K.softmax(masked)  # (N x N)
            dropout = Dropout(self.attn_dropout)(softmax)  # (N x N)

            # Linear combination with neighbors' features
            node_features = K.dot(dropout, linear_transf_X)  # (N x F')

            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape