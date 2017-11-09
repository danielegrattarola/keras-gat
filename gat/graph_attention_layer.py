from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import constraints, regularizers, initializers, activations

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attention_heads=1,
                 attention_heads_reduction='concat',  # ['concat', 'average']
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
        assert attention_heads_reduction in ['concat', 'average'], 'Possbile reduction methods: \'concat\', \'average\''
        self.F_ = F_  # Dimensionality of the output features (F' in the paper)
        self.attention_heads = attention_heads  # Number of attention heads to run in parallel (K in the paper)
        self.attention_heads_reduction = attention_heads_reduction  # 'concat' or 'average' (Equations 5 and 6 in the paper)
        self.activation = activations.get(activation)  # Optional nonlinearity to apply to the weighted node features (Equation 4 in the paper)
        self.use_bias = use_bias  # Not used in the paper, so it defaults to False
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

        if attention_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attention_heads  # Output will be KF'-dimensional
        else:
            self.output_dim = self.F_  # Output will be F'-dimensional

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = self.F = input_shape[0][-1]

        self.kernels = []  # Stores layer kernels for each attention head
        self.attention_kernels = []  # Stores attention kernels for each attention heads
        self.biases = []  # Stores biases for each attention head
        for head in range(self.attention_heads):
            kernel = self.add_weight(shape=(self.F, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            attention_kernel = self.add_weight(shape=(2 * self.F_, 1),
                                               initializer=self.kernel_initializer,
                                               name='att_kernel_%s' % head,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)
            self.attention_kernels.append(attention_kernel)

            if self.use_bias:
                raise NotImplementedError  # TODO
            else:
                bias = None
            self.biases.append(bias)
        super(GraphAttention, self).build(input_shape)

    def call(self, inputs):
        X = inputs[0]  # input graph (B x F)
        G = inputs[1]  # full graph (N x F) (this is necessary in code, but not in theory. Check section 2.2 of the paper)
        B = K.shape(X)[0]  # Get batch size at runtime
        N = K.shape(G)[0]  # Get number of nodes in the graph at runtime

        outputs = []  # Will store the outputs of each attention head (B x F' or B x KF')
        for head in range(self.attention_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attention_kernels[head]  # Attention network a in paper (2*F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # B x F'
            linear_transf_G = K.dot(G, kernel)  # N x F'

            # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
            repeated = K.reshape(K.tile(linear_transf_X, [1, N]), (-1, self.F_))  # B*N x F'
            # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
            tiled = K.tile(linear_transf_G, [B, 1])  # B*N x F'
            # Build combinations
            combinations = K.concatenate([repeated, tiled])  # N*B x 2F'
            combination_slices = K.reshape(combinations, (B, -1, 2 * self.F_))  # B x N x 2F'

            # Attention head
            dense = K.dot(combination_slices, attention_kernel)  # B x N x 1 (a(Wh_i, Wh_j) in the paper)
            dense = K.squeeze(dense, -1)  # B x N
            dense = K.softmax(dense)  # B x N

            # TODO: masking with Vaswani method (add -inf to masked coefficients)

            # Linear combination with neighbors' features
            node_features = K.dot(dense, linear_transf_G)  # B x F'

            if self.use_bias:
                raise NotImplementedError
            if self.attention_heads_reduction == 'concat' and self.activation is not None:
                # In the case of concatenation, we compute the activation here (Equation 5)
                node_features = self.activation(node_features)

            outputs.append(node_features)  # Add output of attention head to final output

        # Reduce the attention heads output according to the reduction method
        if self.attention_heads_reduction == 'concat':
            output = K.concatenate(outputs)
        else:
            output = K.mean(K.stack(outputs), axis=0)
            if self.activation is not None:
                # In the case of mean reduction, we compute the activation now
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)