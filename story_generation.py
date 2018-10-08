# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Consts
SOL_token = 0
EOL_token = 1

MAX_LENGTH = 1024

class Book:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        # Start Of Line, End Of Line
        self.index2word = {0: "SOL", 1: "EOL"}
        self.n_words = 2  # Count SOL and EOL

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Reads the dict from a file, replaces contracted words with their expanded version
def convertContraction(word):
    with open('contractions_dictionary.txt', 'r') as f:
        s = f.read()
        contractions = eval(s)
    if word in contractions and len(contractions[word]) > 0:
        return contractions[word]
    else:
        return word

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Removes unneeded punctuation in the data:
# mr. -> mr, etc
# hyphens between words, ex: A -- B -> A B
# commas, double quotes, hyphens, colons, semi-colons
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"\s-+\s", ' ', s)
    s = re.sub(r"[,\"\-:;]", '', s)
    s = re.sub(r"mr\.", "mr", s)
    s = re.sub(r"mrs\.", "mrs", s)
    s = re.sub(r"ms\.", "ms", s)
    return s

## Perplexity code

import nltk
import collections

# Create a prediction model from all sentences in the corpus
def unigram(sentences):
    model = collections.defaultdict(lambda: 0.01)
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        for f in tokens:
            try:
                model[f] += 1
            except KeyError:
                model[f] = 1
                continue
        for word in model:
            model[word] = model[word]/float(sum(model.values()))
    return model

# Return the perplexity of a word from the model
def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity
##

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readBook(book_title):

    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s.txt' % book_title, encoding='utf-8').\
        read().strip().split('\n')
    lines = [normalizeString(line) for line in lines if len(line) > 0]

    """ old sentences code
    # Split into sentences:
    sentences = []
    currSentence = []
    for line in lines:
        for word in line.strip().split(' '):
            if re.search(r"[.!?]", word):
                currSentence.append(word[:-1])
                currSentence.append(word[-1:])
                if len(currSentence) < MAX_LENGTH:
                    sentences.append(' '.join(currSentence))
                currSentence = []
            else:
                currSentence.append(word)
    """

    ### new sentences code ###
    
    # Split into sentences:
    parsedLines = []
    for line in lines:
        lineList = line.strip().split(' ')

        # pre-parsing
        for i, word in enumerate(lineList):
            # Separate sentences if multiple periods exist based on the following conditions:
            # First char after periods is capital, and not "I" or "I'*".
            if re.search(r"[\.]{2,}$", word):
                # Check the next word
                if i+1 < len(lineList):
                    # First char is capital, and not "I", "I'*", etc
                    if re.search(r"^[A-Z]", lineList[i+1]) and not re.search(r"^(I'.+|I[.!?]*$)", lineList[i+1]):
                        # Replace "..+" with "."
                        word = re.sub(r"[\.]{2,}$", ".", word)
                    else:
                        # Replace "..+" with " "
                        word = re.sub(r"[\.]{2,}$", "", word)
                else:
                    # EOL, replace "..+" with " ", TODO: should we check the first word in next line??
                    word = re.sub(r"[\.]{2,}$", "", word)

            # Removing quotes for words surrounded in single quotes
            if re.match(r"^'.+'$", word):
                word = re.sub(r"'", '', word)
            # Remove trailing periods with single period
            # REPLACED???
            #if re.match(r"^.+\.\.\.$", word):
            #    word = re.sub(r"\.\.\.$", ".", word)

            lineList[i] = word

        parsedLines.append(' '.join(lineList))

    # Parse all lines into sentences
    # A ".", "!", or "?" terminates a sentence
    # All dialogue (double quotes) is separated into its own sentence
    sentences = []
    currSentence = []
    for line in parsedLines:
        for word in line.split(' '):
            # Search for dialogue to find double quotes
            if re.search(r"[\"]", word):
                inDialogue = not inDialogue
                # Remove double quotes from word
                word = re.sub(r"[\"]", '', word)
                # If we're already in dialogue, and have processed words since being in it, we have found the end of the dialogue
                if inDialogue and len(currSentence) > 0:
                    sentences.append(' '.join(currSentence))
                    currSentence = [word]
                # If we're not in dialogue, we found the start of a dialogue
                elif not inDialogue:
                    currSentence.append(word)
                    sentences.append(' '.join(currSentence))
                    currSentence = []
                elif inDialogue and len(currSentence) == 0:
                    currSentence.append(word)

            # Find sentence terminators ('.', '!', and '?')
            elif re.search(r"[.!?]", word):
                currSentence.append(word[:-1])
                # Keep terminator if it is not a period
                if word[-1:] != '.':
                    currSentence.append(word[-1:])
                sentences.append(' '.join(currSentence))
                currSentence = []
            else:
                currSentence.append(word)

    # Final pass of all sentence to unconstruct all contractions
    for i, sentence in enumerate(sentences):
        if "'" in sentence:
            sentenceSplit = sentence.split(' ')
            for j, word in enumerate(sentenceSplit):
                if "'" in word:
                    convertedWords = convertContraction(word).split(' ')
                    sentenceSplit[j] = convertedWords[0]
                    for k, convertedWord in enumerate(convertedWords[1:]):
                        sentenceSplit.insert(j+k+1, convertedWord)
            sentences[i] = ' '.join(sentenceSplit)
    ###                    ###
    
    # Split every line into pairs and normalize
    pairs = [[sentences[i], sentences[i+1]] for i,_ in enumerate(sentences[:len(sentences)-1])]

    print('Num of sentences: %d\nNum of pairs: %d'%(len(sentences),len(pairs)))
    #exit(0)

    perplexity_model = unigram(sentences)

    input_book = Book("input_%s" % book_title)
    output_book = Book("output_%s" % book_title)
    return input_book, output_book, pairs, perplexity_model


### DO WE NEED? ###
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
###################



######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#
def prepareData(book_title):
    input_book, output_book, pairs, perplexity_model = readBook(book_title)
    print("Read %s sentence pairs" % len(pairs))

    # TODO: add trimming?
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))

    # Create training data and test data HERE?
    random.shuffle(pairs)
    train_size = int(len(pairs)*0.8)
    train_pairs = pairs[:train_size]
    test_pairs = pairs[train_size:]

    print("Counting words...")
    for pair in pairs:
        input_book.addSentence(pair[0])
        output_book.addSentence(pair[1])

    print("Counted words:")
    print(input_book.name, input_book.n_words)
    print(output_book.name, output_book.n_words)

    return input_book, output_book, train_pairs, test_pairs, perplexity_model



######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOL token to both sequences.
#

def indexesFromSentence(book, sentence):
    return [book.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(book, sentence):
    indexes = indexesFromSentence(book, sentence)
    indexes.append(EOL_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_book, output_book):
    input_tensor = tensorFromSentence(input_book, pair[0])
    target_tensor = tensorFromSentence(output_book, pair[1])
    return (input_tensor, target_tensor)


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOL>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOL_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOL_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, n_iters, epochs, train_pairs, input_book, output_book, print_every=1000, plot_every=100, learning_rate=0.01):
    #n_iters == iterations
    #epochs = iterations / num of examples
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    training_pairs = [tensorsFromPair(random.choice(train_pairs), input_book, output_book) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    
    #reachedLoss = False

    # while epochs != 100 (param)
        # for i in train
            # run network
        # epochs++
        # don't break if loss < 1, let it run for all epochs for now
    # new code
    """
    for i in range(epochs):
        for j, train_pair in enumerate(train_pairs):
            pair = tensorsFromPair(train_pair, input_book, output_book)
            input_tensor = pair[0]
            target_tensor = pair[1]
            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if j % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                #print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                #                             iter, iter / n_iters * 100, print_loss_avg))
                now = time.time()
                print('%s, epoch=%d, (%d%%) loss_avg=%.4f' % (asMinutes(now - start), i+1, 100*int(j / len(train_pairs)), print_loss_avg))
            #if j % plot_every == 0:
            #    plot_loss_avg = plot_loss_total / plot_every
            #    plot_losses.append(plot_loss_avg)
            #    plot_loss_total = 0
            #if loss < 1.0:
            #    reachedLoss = True
        # DON'T use for now
        #if reachedLoss:
        #    break
    """
    # original code
    #"""
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    #"""
    #showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOL token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, input_book, output_book, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_book, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOL_token]], device=device)  # SOL

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOL_token:
                decoded_words.append('<EOL>')
                break
            else:
                decoded_words.append(output_book.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

from nltk.translate.bleu_score import sentence_bleu

def calculateBleu(candidate, reference, n_gram=2):
    # looks at ration of n-grams between 2 texts
    # Break candidate/reference into the format below
    candidate = candidate.split(' ')
    reference = reference.split(' ')
    return sentence_bleu(reference, candidate)#, weights=(1,0,0,0))

def evaluateRandomly(encoder, decoder, test_pairs, input_book, output_book, perplexity_model, n=10):
    i = 0
    print('Printing first 10 evaluations:')
    bleu_total = 0.0
    perplexity_total = 0.0
    for test_pair in test_pairs:
        output_words, attentions = evaluate(encoder, decoder, test_pair[0], input_book, output_book)
        output_sentence = ' '.join(output_words)
        bleu_score = calculateBleu(output_sentence, test_pair[1])
        perplexity_score = perplexity(output_sentence, perplexity_model)
        if i < 10:
            print('> [%s]'%test_pair[0])
            print('= [%s]'%test_pair[1])
            print('< [%s]'%output_sentence)
            print('BLEU: %.4f'%bleu_score)
            print('Perplexity: %.4f'%perplexity_score)
        i += 1
        # Calculate BLEU score
        bleu_total += bleu_score#calculateBleu(output_sentence, test_pair[1])
        perplexity_total += perplexity_score
        
    avg_bleu = bleu_total / len(test_pairs)
    avg_perplexity = perplexity_total / len(test_pairs)
    print('Average BLEU score = ' + str(avg_bleu))
    print('Average Perplexity score = ' + str(avg_perplexity))
    return avg_bleu, avg_perplexity
######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

import pickle
from pathlib import Path

######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#
"""
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())
"""

######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOL>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, input_book, output_book, encoder, decoder):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence, input_book, output_book)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

def createStory(input_sentence, input_book, output_book, encoder, decoder):
    input_sentence = normalizeString(input_sentence)
    story = [input_sentence]
    continueStory = True
    i = 0
    while continueStory:
        output_words, attentions = evaluate(
            encoder, decoder, input_sentence, input_book, output_book)
        print('input =', input_sentence)
        if output_words[len(output_words)-1] == '<EOL>':
            output_words = output_words[:len(output_words)-1]
        input_sentence = ' '.join(output_words)
        print('output =', input_sentence)
        story.append(input_sentence)
        i += 1
        if i > 10:
            continueStory = False
    return story




def main():


    input_book, output_book, train_pairs, test_pairs, perplexity_model = prepareData('1_sorcerers_stone')

    #stuff
    
    hidden_size = 256
    encoder_filename_format = "data/encoder_%d.p"
    decoder_filename_format = "data/decoder_%d.p"

    epoch_sizes = [0]#[100, 200, 400, 600]
    for epoch_size in epoch_sizes:
        print('Epoch size: %d' % epoch_size)
        encoder_filename = encoder_filename_format%epoch_size
        decoder_filename = decoder_filename_format%epoch_size
        
        encoder_file = Path(encoder_filename)
        encoder1 = None
        attn_decoder1 = None
    
        if encoder_file.is_file():
            print("Loading encoder and decoder from files...")
            encoder1 = pickle.load(open(encoder_filename, "rb"))
            attn_decoder1 = pickle.load(open(decoder_filename, "rb"))
        else:
            print("Training encoder and decoder...")
            encoder1 = EncoderRNN(input_book.n_words, hidden_size).to(device)
            attn_decoder1 = AttnDecoderRNN(hidden_size, output_book.n_words, dropout_p=0.1).to(device)

            iter_amount = 100000
            trainIters(encoder1, attn_decoder1, iter_amount, epoch_size, train_pairs, input_book, output_book, print_every=5000)
        
            ## Dump to pickle ##
            print("Dumping encoder and decoder to files...")
            pickle.dump(encoder1, open(encoder_filename, "wb"))
            pickle.dump(attn_decoder1, open(decoder_filename, "wb"))
    
        bleu_score, perplexity_score = evaluateRandomly(encoder1, attn_decoder1, test_pairs, input_book, output_book, perplexity_model)
        print('BLEU Score for %d epochs: %.4f' % (bleu_score, epoch_size))
        print('Perplexity score for %d epochs: %.4f' % (perplexity_score, epoch_size))

        # Generate test story
        story1 = createStory("You're a wizard Harry .", input_book, output_book, encoder1, attn_decoder1)
        for line in story1:
            print('> %s' % line)

main()

