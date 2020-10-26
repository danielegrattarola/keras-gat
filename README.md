# Keras Graph Attention Network

|**DEPRECATED**|
|:------------|
| This implementation of GAT is no longer actively maintained and may not work with modern versions of Tensorflow and Keras. Check out [Spektral](https://graphneural.network) and its [GAT example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py) for a Tensorflow/Keras implementation of GAT.| 

This is a Keras implementation of the Graph Attention Network (GAT) model by Veličković et al. (2017, [[arXiv link]](https://arxiv.org/abs/1710.10903)).

## Acknowledgements
I have no affiliation with the authors of the paper and I am implementing this code for non-commercial reasons.  
The authors published their [reference Tensorflow implementation here](https://github.com/PetarV-/GAT), so check it out for something that is guaranteed to work as intended. Their implementation is slightly different than mine, so that may be something to keep in mind.
You should cite the paper if you use any of this code for your research:
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
If you would like to give me credit, feel free to link to my [Github profile](https://github.com/danielegrattarola), [blog](https://danielegrattarola.github.io), or [Twitter](https://twitter.com/riceasphait).

I also copied the code in `utils.py` almost verbatim from [this repo by Thomas Kipf](https://github.com/tkipf/gcn), who I thank sincerely for sharing his work on GCNs and GAEs, and for giving me a few pointers on how to split the data into train/test/val sets.

Thanks to [mawright](https://github.com/mawright), [matthias-samwald](https://github.com/matthias-samwald), and [vermaMachineLearning](https://github.com/vermaMachineLearning) for helping me out with bugs, performance improvements, and running experiments.

## Disclaimer
I do not own any rights to the datasets distributed with this code, but they are publicly available at the following links:

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

Or you can just copy and paste `graph_attention_layer.py` into your project.

## Replicating experiments
To replicate the experimental results of the paper, simply run:
```sh
$ python examples/gat.py
```
