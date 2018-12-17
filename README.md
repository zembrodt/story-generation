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

Along with `story_generation.py`, several other files can be executed as standalone scripts:

- `perplexity_study.py` allows the user to gather perplexity results from the best model saved in the `obj` directory, or a specific embedding type using the `--embedding` parameter
- `storygen/book.py` provides use to parse or filter standalone text into new files. `./book.py -h` for more information
- `util/display_loss.py` allows the user to display the loss values for select word embeddings, with or without validation values. `./display_loss.py -h` for more information
- `util/loss_analysis.py` allows the user to view min/max loss values of a given file, or find loss values at a specific epoch. `./loss_analysis.py -h` for more information

## Current Results

Determined loading from checkpoints was not working correctly, and causing spikes in loss values when a model was loaded.<br/>
Trained a Continuous Bag of Words word2vec embedding for 500 epochs. With this word2vec embedding, and the pre-trained GloVe embedding, trained 2 models for 500 epochs, without loading any checkpoints.<br/>
The GloVe embedding seems to heavily outperform the word2vec embedding, with a minimum loss of value of **2.983** at the 250th epoch, and a final loss value of **3.083**.<br/>
The results for each model is below:
<img src="https://i.imgur.com/4d6hhTp.png" alt="Model trained with CBOW word2vec word embedding" /><br/>
<img src="https://i.imgur.com/C3lChph.png" alt="Model trained with GloVe word embedding" /><br/>
A perplexity study was also ran on the model with the GloVe embedding, with the results in the below table:<br/>
| Epochs |                  | Training data | Testing data |
|--------|------------------|---------------|--------------|
| 250    | Actual sentences | 42.9893       | 736176.1818  |
|        | Random words     | 146455354.9   | 147674868.5  |
|        | Random sentences | 295889.4025   | 299692.5997  |
| 500    | Actual sentences | 37.0408       | 4872462.827  |
|        | Random words     | 1270328937    | 6201033178   |
|        | Random sentences | 27962670.27   | 2313469.597  |<br/>
Here, <em>actual sentences</em> refers to the score of given an input sentence, the perplexity of the model forcing it to evaluate to the real target sentence. <em>Random words</em> is such that given an input sentence, the perplexity of the model forcing it to evaluate a random target sentence of the same length as the real target sentence, but where each word was replaced by a random word selected from the corpus. Lastly, <em>random sentences</em> is such that given the input sentence, and forced to evaluate a target sentence of the same length as the real target sentence, but randomly chosen from the data.<br/>
Out of the three categories of studies, <em>actual sentences</em> should perform the best, as they are the real sentences to follow the input sentences. This will be followed by <em>random sentences</em>, as they are full real sentences, meaning they have a real sentence structure, but may have nothing to do with the input. Finally, <em>random words</em> should perform the worse, as they will likely not have a real sentence structure. <br/>
The above table shows that this is the case: training data usually outperforms testing data, <em>actual sentences</em> outperforms <em>random sentences</em>, which then outperformed <em>random words</em>. The 250th epoch model is taken from the minimum point in figure D, with an <em>actual sentences</em> value of **42.989**, this perplexity value was actually reduced to **37.041** for the 500th epoch of the same model. In fact, all training data perplexity values between the two models was decreased. However, the testing perplexity values all increased. For the extra 250 epochs the model trained for, it seemed to continue learning, but fitted too exactly to the training data, and became more "perplexed" by data it had not seen before. Finally, the actual perplexity themselves were extremely high for the testing data, further showing that perhaps the model hasn't learned enough to handle unseen data. One cause of this could be due to the lack for training data, with just over 5,000 sentence pairs. 

## Future work

Beyond correcting current drawbacks, such as checkpoint loading issues, high perplexity values, and a minimum loss value of 3, future work could include:
- Training and testing a working model on corpora of different types, such as news articles or song lyrics
- Training more custom embeddings, either current ones for much longer, or using GloVe to train custom word embeddings rather than word2vec

## Previous Results

### 11/15/2018

Trained three word2vec embeddings on all Harry Potter texts: Skip-Gram and Continuous Bag of Words trained for 15 epochs, and Continuous Bag of Words trained for 300 epochs.<br/>
With these 3 word2vec embeddings, the previous GloVe embedding, and the default random embedding, trained five models for 500 epochs on the data.<br/>
The models still seem to be underfitting, with the word2vec embeddings outperforming the random embedding. GloVe embedding still performs the best. See the results for loss values in the figure below:<br/>
<img src="https://i.imgur.com/YZZjo1f.png" alt="Loss graph, 500 epochs, for random, GloVe, word2vec-sg-15, word2vec-cbow-15, word2vec-cbow-300" />

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