# order-embeddings-wordnet
Code for the hypernym completion experiments on WordNet from the paper ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361). See [the other repo](https://github.com/ivendrov/order-embeddings) for the other experiments.

## Dependencies
- Python 2 with a recent version of Numpy and [nltk 3.0](http://www.nltk.org/) for easy access to WordNet.
- [Torch7](http://torch.ch/) with the [argparse](https://github.com/mpeterv/argparse) package.

## Create Datasets
Run
```
python preprocessWordnet.py
th createDatasets.lua
```

## Training the Model
To train with default hyperparameters (the order-embedding model from the paper), run
```
th main.lua --epochs 20 --name "myfirstmodel"
```

or train your own version by setting any of the the flags in `main.lua`.

The resulting weights are stored in `weights.t7`. You can view traces of training
and validation error by navigating to the `vis_training` directory, running

```
python -m SimpleHTTPServer
```
and pointing your browser to the server (usually `localhost:8000`).
