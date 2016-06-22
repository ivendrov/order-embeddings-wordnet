# hembeddings
Learning Hierarchical Embeddings TODO: Paper

## Dependencies
- Python 2 with a recent version of Numpy and NLTK.
- Torch7 with the 'argparse' package.


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

The resulting weights are stored in `weights.t7`, and you can view traces of training
and validation error by navigating to the `vis_training` directory, running

```
python -m SimpleHTTPServer
```
and pointing your browser to the server (usually `localhost:8000`).
