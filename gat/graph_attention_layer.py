from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import constraints, regularizers, initializers, activations, Dropout

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attention_heads=1,
                 attention_heads_reduction='concat',  # ['concat', 'average']
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 att_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 att_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 att_kernel_constraint=None,
                 **kwargs):
        assert attention_heads_reduction in ['concat', 'average'], \
            'Possbile reduction methods: \'concat\', \'average\''

        self.F_ = F_  # Dimensionality of the output features (F' in the paper)
        self.attention_heads = attention_heads  # Number of attention heads to run in parallel (K in the paper)
        self.attention_heads_reduction = attention_heads_reduction  # 'concat' or 'average' (Equations 5 and 6 in the paper)
        self.activation = activations.get(activation)  # Optional nonlinearity to apply to the weighted node features (Equation 4 in the paper)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.att_kernel_initializer = initializers.get(att_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.att_kernel_regularizer = regularizers.get(att_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.att_kernel_constraint = constraints.get(att_kernel_constraint)
        self.supports_masking = False

        if attention_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attention_heads  # Output will be KF'-dimensional
        else:
            self.output_dim = self.F_  # Output will be F'-dimensional

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.F = input_shape[0][-1]
        self.kernels = []  # Stores layer kernels for each attention head
        self.attention_kernels = []  # Stores attention kernels for each attention heads

        # Initialize kernels for each attention head
        for head in range(self.attention_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(self.F, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # Attention kernel
            attention_kernel = self.add_weight(shape=(2 * self.F_, 1),
                                               initializer=self.att_kernel_initializer,
                                               name='att_kernel_%s' % head,
                                               regularizer=self.att_kernel_regularizer,
                                               constraint=self.att_kernel_constraint)
            self.attention_kernels.append(attention_kernel)

        super(GraphAttention, self).build(input_shape)

    def call(self, inputs):
        X = inputs[0]  # input graph (B x F)
        G = inputs[1]  # full graph (N x F) (this is necessary in code, but not in theory. Check section 2.2 of the paper)
        A = inputs[2]  # Adjacency matrix of the graph, used for masking (B x N)
        B = K.shape(X)[0]  # Get batch size at runtime
        N = K.shape(G)[0]  # Get number of nodes in the graph at runtime

        outputs = []  # Will store the outputs of each attention head (B x F' or B x KF')
        for head in range(self.attention_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attention_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # B x F'
            linear_transf_G = K.dot(G, kernel)  # N x F'

            # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
            repeated = K.reshape(K.tile(linear_transf_X, [1, N]), (B * N, self.F_))  # (BN x F')
            # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
            tiled = K.tile(linear_transf_G, [B, 1])  # (BN x F')
            # Build combinations
            combinations = K.concatenate([repeated, tiled])  # (BN x 2F')
            combination_slices = K.reshape(combinations, (B, -1, 2 * self.F_))  # (B x N x 2F')

            # Attention head
            dense = K.squeeze(K.dot(combination_slices, attention_kernel), -1)  # a(Wh_i, Wh_j) in the paper (B x N x 1)
            masked = dense - A  # Masking technique by Vaswani et al., section 2.2 of paper (B x N)
            softmax = K.softmax(masked)  # (B x N)
            dropout = Dropout(0.5)(softmax)  # Apply dropout to normalized attention coefficients (B x N)

            # Linear combination with neighbors' features
            node_features = K.dot(dropout, linear_transf_G)  # (B x F')

            # In the case of concatenation, we compute the activation here (Equation 5)
            if self.attention_heads_reduction == 'concat' and self.activation is not None:
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attention_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (B x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0) # (B x F')
            if self.activation is not None:
                # In the case of mean reduction, we compute the activation now
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)