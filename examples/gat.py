from __future__ import division

from keras_gat import GraphAttention
from keras_gat.utils import load_data
from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard

# TODO: the exact config of the paper by Velickovic et al. may not fit on GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Read data
A_train, X_train, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')

# Parameters
N = X_train.shape[0]  # Number of nodes in the graph
F = X_train.shape[1]  # Original feature dimesnionality
F_ = 8  # Output dimension of first GraphAttention layer
n_classes = Y_train.shape[1]  # Number of classes
dropout_rate = 0.6  # Dropout rate applied to the input of GraphAttention layers
l2_reg = 5e-4  # Regularization rate for l2
learning_rate = 0.005  # Learning rate for SGD
epochs = 2000  # Number of epochs to run for
es_patience = 100

# Preprocessing operations
X_train /= X_train.sum(1).reshape(-1, 1)
A_train = A_train.toarray()

# Model definition (as per Section 3.3 of the paper)
X = Input(shape=(F, ))
A = Input(shape=(N, ))

dropout1 = Dropout(dropout_rate)(X)
graph_attention_1 = GraphAttention(F_,
                                   attn_heads=8,
                                   attn_heads_reduction='concat',
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg))([dropout1, A])
dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   activation='softmax',
                                   kernel_regularizer=l2(0.0005))([dropout2, A])

# Build model
model = Model(inputs=[X, A], outputs=graph_attention_2)
optimizer = Adam(lr=0.005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Callbacks
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = TensorBoard(batch_size=N)

# Train model
validation_data = ([X_train, A_train], Y_val, idx_val)
model.fit([X_train, A_train],
          Y_train,
          sample_weight=idx_train,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[es_callback, tb_callback])

# Evaluate model
eval_results = model.evaluate([X_train, A_train],
                              Y_test,
                              sample_weight=idx_test,
                              batch_size=N,
                              verbose=0)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
