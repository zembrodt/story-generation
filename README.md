# story-generation
Story generation project using Seq2Seq networks

### Requirements
- [pytorch](https://pytorch.org/)
- [nltk](https://www.nltk.org/)
- [pymeteor](https://github.com/zembrodt/pymeteor)<br/>
***Note***: to install pymeteor you must use the Test PyPi server:<br/>
`pip install --index-url https://test.pypi.org/simple/ pymeteor`

Download the Wikipedia 2014 + Gigaword 5 pre-trained vectors from [GloVe](https://nlp.stanford.edu/projects/glove/), download link [here](http://nlp.stanford.edu/data/glove.6B.zip).<br/>
Unzip the text files to the location `data/glove.6B/`

### Current Results
Currently getting a minimum loss value of 2.977 until the loss spikes around the 30th epoch, as you can see in the figure below:<br/>
<img src="https://i.imgur.com/QyFyXIT.png" alt="Loss graph" />

This seems to be due to the model beginning with the values from the above word embeddings, then breaking out and not being able to find the local optimum for the Harry Potter texts. An idea to correct this is to train our own word2vec on the Harry Potter texts.<br />
The model is also underfitting (on the 100th epoch) when evaluated, predicting sentences with repeated words. [PERPLEXITY VALUE?]

### Todo
- Evaluate a checkpointed model before the loss value spikes to see what the perplexity value is.
- Train our own word2vec embeddings on all Harry Potter texts to use.
- Determine the loss values of a model trained without pre-trained embeddings for comparison.