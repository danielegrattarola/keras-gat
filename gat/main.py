from __future__ import division
from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from graph_attention_layer import GraphAttention
from utils import load_data, get_splits, evaluate_preds
import numpy as np

# Read data
X_train, A_train, Y = load_data()
Y_train, Y_val, Y_test, idx_train, idx_val, idx_test, train_mask = get_splits(Y)

# Normalize X
X_train /= X_train.sum(1).reshape(-1, 1)

# Convert adacency matrix to masking matrix for softmax (Vaswani et al., 2017)
A_train = A_train.toarray()
A_train[A_train == 0] = 1e09
A_train[A_train == 1] = 0.0  # Connections are now valued 0, the rest is inf

# Hyperparameters and constants
F = X_train.shape[-1]  # Original feature dimesnionality
F_ = 8  # Output dimension of first GraphAttention layer
N = X_train.shape[0]  # Number of nodes in the graph
n_classes = Y_train.shape[-1]  # Number of classes
dropout_rate = 0.5  # Dropout rate applied to the input of GraphAttention layers
epochs = 10  # Number of epochs to run for

# Model definition (as per Section 3.3 of the paper)
X = Input(shape=(F, ))
G = Input(shape=(F, ))
A = Input(shape=(N, ))

graph_attention_1 = GraphAttention(F_,
                                   attention_heads=8,
                                   attention_heads_reduction='concat',
                                   activation='elu',
                                   kernel_regularizer=l2(0.0005))([Dropout(dropout_rate)(X), G, A])

graph_attention_2 = GraphAttention(n_classes,
                                   attention_heads=1,
                                   attention_heads_reduction='average',
                                   activation='softmax',
                                   kernel_regularizer=l2(0.0005))([Dropout(dropout_rate)(graph_attention_1), graph_attention_1, A])

# Build and compile the model
model = Model(inputs=[X, G, A], outputs=graph_attention_2)
model.compile(Adam(lr=0.005), 'categorical_crossentropy', metrics=['acc'])
model.summary()

# Main training loop
for epoch in range(epochs):
    model.fit([X_train, X_train, A_train],
              Y_train,
              sample_weight=train_mask,
              epochs=1,
              batch_size=X_train.shape[0],
              verbose=0)

    # Test validation loss and accuray
    preds = model.predict([X_train, X_train, A_train],
                          batch_size=X_train.shape[0])
    train_val_loss, train_val_acc = evaluate_preds(preds,
                                                   [Y_train, Y_val],
                                                   [idx_train, idx_val])
    print "Epoch: {:04d}".format(epoch), \
          "train_loss= {:.4f}".format(train_val_loss[0]), \
          "train_acc= {:.4f}".format(train_val_acc[0]), \
          "val_loss= {:.4f}".format(train_val_loss[1]), \
          "val_acc= {:.4f}".format(train_val_acc[1])


# Final testing
preds = model.predict([X_train, X_train, A_train],
                      batch_size=X_train.shape[0])
test_loss, test_acc = evaluate_preds(preds, [Y_test], [idx_test])
print "Test set results:", \
      "loss= {:.4f}".format(test_loss[0]), \
      "accuracy= {:.4f}".format(test_acc[0])