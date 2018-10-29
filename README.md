# story-generation
Story generation project using Seq2Seq networks

### Requirements
- [pytorch](https://pytorch.org/)
- [nltk](https://www.nltk.org/)
- [pymeteor](https://github.com/zembrodt/pymeteor)<br/>
***Note***: to install pymeteor you must use the Test PyPi server:<br/>
`pip install --index-url https://test.pypi.org/simple/ pymeteor`

Download the Wikipedia 2014 + Gigaword 5 pre-trained vectors from [GloVe](https://nlp.stanford.edu/projects/glove/), download link [here](http://nlp.stanford.edu/data/glove.6B.zip).<br/>
Unzip the text files to the location "data/glove.6B/".