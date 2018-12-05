# story-generation

Story generation project using Seq2Seq networks

## Requirements

- [Python 3.x](https://www.python.org/downloads/)
- [pytorch](https://pytorch.org/)
- [nltk](https://www.nltk.org/)
- [pymeteor](https://github.com/zembrodt/pymeteor)

***Note***: to install pymeteor you must use the Test PyPi server:<br/>
`pip install --index-url https://test.pypi.org/simple/ pymeteor`

Tested with Python 3.6.6, PyTorch 0.4.1, and Cuda 9.0.176

Download the Wikipedia 2014 + Gigaword 5 pre-trained vectors from [GloVe](https://nlp.stanford.edu/projects/glove/), download link [here](http://nlp.stanford.edu/data/glove.6B.zip).<br/>
Unzip the text files to the location `data/glove.6B/`

### Usage

`python3 story_generation.py` or `./story_generation.py`<br/>
All command line arguments are optional, and any combination (beides `-h, --help`) can be used.<br/>
Arguments:

- `-h, --help` : Provides help on command line parameters
- `--epoch <epoch_value>` : specify an epoch value to train the model for or load a checkpoint from
- `--embedding <embedding_type>` : specify an embedding to use from: `[glove, sg, cbow]`
- `--loss <loss_dir>` : specify a directory to load loss values from (requires files `loss.dat` and `validation.dat`)

## Current Results

Trained three word2vec embeddings on all Harry Potter texts: Skip-Gram and Continuous Bag of Words trained for 15 iterations, and Continuous Bag of Words trained for 300 iterations.<br/>
With these 3 word2vec embeddings, the previous GloVe embedding, and the default random embedding, trained five models for 500 epochs on the data.<br/>
The models still seem to be underfitting, with the word2vec embeddings outperforming the random embedding. GloVe embedding still performs the best. See the results for loss values in the figure below:<br/>
<img src="https://i.imgur.com/YZZjo1f.png" alt="Loss graph, 500 epochs, for random, GloVe, word2vec-sg-15, word2vec-cbow-15, word2vec-cbow-300" />

## Todo
- [x] Retrain the Continuous Bag of Words word2vec embedding for 500 iterations
- [x] Update method of parsing text lines into sentences
- [ ] Retrain model on the new word2vec-cbow-500 embedding, and the previous GloVe embedding, for 500 epochs with the updated text parsing
- [ ] Run perplexity study on the two new models

## Previous Results

### 11/9/2018

Currently getting a minimum loss value of **2.977** until the loss spikes around the 30th epoch, as you can see in the figure below:<br/>
<img src="https://i.imgur.com/QyFyXIT.png" alt="Loss graph, 100 epochs" />

This seems to be due to the model beginning with the values from the above word embeddings, then breaking out and not being able to find the local optimum for the Harry Potter texts. An idea to correct this is to train our own word2vec on the Harry Potter texts.<br />
The model is also underfitting (on the 100th epoch) when evaluated, predicting sentences with repeated words.<br/>
Evaluating this model with beam search (k=1), the average perplexity is **100,562.7942**.<br/>
Evaluating this model with beam search (k=5), the average perplexity is **93,277.5684**.

Retraining this model on 40 epochs, we get an minimum loss value of **2.959** at the 23rd epoch.<br/>
Evaluating this model at k=1 gives us a perplexity value of **675.8603**.<br/>
Evaluating this model at k=5 gives us a perplexity value of **669.7439**.<br/>
Viewing prediction results at this point in training the model, it is apparent that the model is not yet underfitting.<br/>
Refer to the figure below to see the chosen minimum loss value at epoch=22, before the loss value spikes at epoch=30.<br/>
<img src="https://i.imgur.com/NhScaLG.png" alt="Loss graph, 40 epochs" />