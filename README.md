# Keras Graph Attention Network
This is a Keras implementation of the Graph Attention Network model
presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

## Premise
This code implements the exact model and experimental setup described in
the paper, but I haven't been able to reproduce their results yet.

Part of the problem is that the memory requirements of the algorithm are
in the order of O(V^2), where V is the number of nodes in the graph, so
I am forced to run everything on the i7-7700HQ CPU of my laptop.
**If you have GPU resources to spare, please let me know the results you
get**.
Ideally the model should reach a 83.5% accuracy on the Cora dataset,
with the experimental setup described in the paper and implemented in
the code (140 training nodes, 500 validation nodes, 1000 test nodes).

## Disclaimer
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
Make also sure to give me credit by linking to this repo.

I also copied the code in `utils.py` almost verbatim from [this repo by Thomas Kipf](https://github.com/tkipf/gcn),
who I thank sincerely for sharing his work on GCNs and GAEs.
The Cora dataset is also shamelessly copy-pasted from his repository
(`README.md` and all), I hope I'm not breaking any licensing laws.

## Installation
To install as a module:
```
$ git clone https://github.com/danielegrattarola/keras-gat.git
$ cd keras-get
$ pip install -e .
$ python
>>> from gat import GraphAttention  # That's all you can import
```

Or you can just copy and paste `graph_attention_layer.py` into your
project.

I tested this with Tensorflow 1.4.0 and Keras 2.0.9 on Python 2.7.12,
and I don't provide support for any other versions for now.

## Graph Attention Networks
I'm working on a blog post detailing how GATs work, so for now I'll
just put a schematic view of my GraphAttention implementation here below
and leave the details for later.
In this schematic:
- N is the number of nodes in the graph
- B is the number of nodes fed as input to the model; this detail is not
implemented yet in the code, so B = N for now
- F is the number of input node features
- F' is the number of output node features
- || indicates a sort of concatenation operation; why this operation
yields a B x N x 2F' output is explained a bit better in the source code,
and it's for purely practical reasons. Check out section 2.2 of the
paper (right after the bullet points) for even more details.

Note that for the sake of clarity I didn't explicitly represent
different attention heads in parallel. Think of this whole diagram as an
attention head on its own.

![GraphAttention layer](GAT.png?raw=True)
