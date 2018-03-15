# Keras Graph Attention Network
This is a Keras implementation of the Graph Attention Network (GAT)
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

## Acknowledgements
I have no affiliation with the authors of the paper and I am
implementing this code for non-commercial reasons.  
The authors published their official Tensorflow implementation 
[here](https://github.com/PetarV-/GAT), so check it out for something that is 
guaranteed to work as intended. From what I gather, their implementation is 
slightly different than mine, so that may be something I will investigate in the 
future.  
You should also cite the paper if you use any of this code for your research:
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={Accepted as poster},
}
```
If you would like to give me credit, feel free to link to my
[Github profile](https://github.com/danielegrattarola),
[blog](https://danielegrattarola.github.io), or
[Twitter](https://twitter.com/riceasphait).

I also copied the code in `utils.py` almost verbatim from [this repo by
Thomas Kipf](https://github.com/tkipf/gcn), whom I thank sincerely for
sharing his work on GCNs and GAEs, and for giving me a few pointers on
the data splits.

A big thank you to [matthias-samwald](https://github.com/matthias-samwald), who 
was able to run the full model on a better GPU than mine and report the 
performance of the exact config described in the paper.  
He got a 81.7% test accuracy, so pretty close to the 83.5% reported in the 
paper (the difference in implementation w.r.t the authors' published code 
might be playing a role here). 

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
