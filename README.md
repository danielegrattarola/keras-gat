# Keras Graph Attention Network
This is a Keras implementation of the Graph Attention Network (GAT)
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

## Premise
This code implements the exact model and experimental setup described in
the paper, but I haven't been able to reproduce their exact results yet.
I get really close, but I can't fit eight attention heads at the same
time on my laptop's GPU.

Ideally the model should reach a 83.5% accuracy on the Cora dataset,
with the experimental setup described in the paper and implemented in
the code (140 training nodes, 500 validation nodes, 1000 test nodes).
If you manage to run the same setup of the paper, let me know your
results.

## Acknowledgements
I have no affiliation with the authors of the paper and I am
implementing this code for non-commercial reasons.
You should cite the paper if you use any of this code for your research:
```
@article{article,
         author = {Veličković, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Liò, Pietro and Bengio, Y},
         year = {2017},
         month = {10},
         pages = {},
         title = {Graph Attention Networks}
}
```
If you would like to give me credit, feel free to link to my
[Github profile](https://github.com/danielegrattarola),
[blog](https://danielegrattarola.github.io), or
[Twitter](https://twitter.com/riceasphait).

I also copied the code in `utils.py` almost verbatim from [this repo by
Thomas Kipf](https://github.com/tkipf/gcn), who I thank sincerely for
sharing his work on GCNs and GAEs, and for giving me a few pointers on
the data splits.

I do not own the rights to the datasets distributed with this code, but
they are widely available on the internet so it didn't feel wrong to
share. If you have problems with this, feel free to contact me.

## Installation
To install as a module:
```
$ git clone https://github.com/danielegrattarola/keras-gat.git
$ cd keras-gat
$ pip install -e .
$ python
>>> from keras_gat import GraphAttention
```

Or you can just copy and paste `graph_attention_layer.py` into your
project.

I tested this with Tensorflow 1.4.0 and Keras 2.0.9 on Python 2.7.12,
and I don't provide support for any other versions for now.

**Note**: since version 1.1, the module only works with Tensorflow.
Theano support is not planned in the near future.

## Replicating experiments
If you wish to replicate the experimental results of the paper, simply
run:
```sh
$ python examples/gat.py
```

from the base folder.
If you want to try and run it on a GPU or TPU, just comment out these
lines (12 and 13) from the same file:
```py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## Graph Attention Networks
I'm working on a blog post explaining GATs, so [stay tuned](https://danielegrattarola.github.io).
(Also, I lied and I'm not actually working on the post _RIGHT NOW_. But
I'll get around to it, I promise).
