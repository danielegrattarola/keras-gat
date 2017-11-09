from keras.models import Model
from keras.layers import Input
from graph_attention_layer import GraphAttention

F = 3
F_ = 2
N = 10
B = 5

X = Input(shape=(F, ))
G = Input(shape=(F, ))
graph_attention_1 = GraphAttention(F_,
                                   attention_heads=2,
                                   attention_heads_reduction='average',
                                   activation='softmax')([X, G])

model = Model(inputs=[X, G], outputs=graph_attention_1)
model.compile('adam', 'mse')
