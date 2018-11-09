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

### Todo
- Train our own word2vec embeddings on all Harry Potter texts to use.
- Determine the loss values of a model trained without pre-trained embeddings for comparison.