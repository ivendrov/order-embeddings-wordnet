# order-embeddings-wordnet
Code for the hypernym completion experiment from the paper ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361). See [the other repo](https://github.com/ivendrov/order-embedding) for the caption-image ranking and textual entailment experiments.

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

## Reference

If you found this code useful, please cite the following paper:

Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun. **"Order-Embeddings of Images and Language."** *arXiv preprint arXiv:1511.06361 (2015).*

    @article{vendrov2015order,
      title={Order-embeddings of images and language},
      author={Vendrov, Ivan and Kiros, Ryan and Fidler, Sanja and Urtasun, Raquel},
      journal={arXiv preprint arXiv:1511.06361},
      year={2015}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
