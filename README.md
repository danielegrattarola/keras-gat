# Keras Graph Attention Network
This is a Keras implementation of the Graph Attention Network (GAT)
model by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

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
how to split the data into train/test/val sets.

A big thank you to [matthias-samwald](https://github.com/matthias-samwald),
who was able to run the full model on a better GPU than mine and report
the performance of the config described in the paper (see section below).

Credits for commit f4974ac go to user [mawright](https://github.com/mawright) 
who forked the code implementing attention to make it way more fast
and memory efficient.

I also thank [vermaMachineLearning](https://github.com/vermaMachineLearning)
for suggesting the edits of commit 7959bd8, which made the layer 4 times
faster and finally brought the test accuracy to the same values reported
in the paper.

## Disclaimer
I do not own any rights to the datasets distributed with this code, but
they are publicly available at the following links:

- CORA: [https://relational.fit.cvut.cz/dataset/CORA](https://relational.fit.cvut.cz/dataset/CORA)
- PubMed: [https://catalog.data.gov/dataset/pubmed](https://catalog.data.gov/dataset/pubmed)
- CiteSeer: [http://csxstatic.ist.psu.edu/about/data](http://csxstatic.ist.psu.edu/about/data)

## Installation
To install as a module:
```
$ git clone https://github.com/danielegrattarola/keras-gat.git
$ cd keras-gat
$ pip install .
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

[matthias-samwald](https://github.com/matthias-samwald) got a 81.7% test
accuracy with v1.1, so pretty close to the 83.5% reported by the authors
(the difference in implementation w.r.t the authors might be responsible).  
[vermaMachineLearning](https://github.com/vermaMachineLearning) was
finally able to get an 83.5% by correctly adding self-loops to the input
graph. Since commit f4974ac (v1.3), the example script should replicate
the results of the paper.
