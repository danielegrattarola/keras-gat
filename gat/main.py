from keras.models import Sequential
from graph_attention_layer import GraphAttention

model = Sequential()
model.add(GraphAttention(3, 1, input_shape=(5, )))

model.compile('adam', 'mse')
