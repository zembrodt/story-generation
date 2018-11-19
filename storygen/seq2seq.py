# seq2seq.py
import copy
import time
import random
import time
import math
import os
import operator
import re
from pathlib import Path

import pymeteor.pymeteor as pymeteor

import torch
import torch.nn as nn
from torch import optim

from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from storygen import book
from storygen.book import START_ID
from storygen.book import STOP_ID
from storygen import encoder
from storygen import decoder
from storygen import glove
from storygen import log

# Filename format: obj/{encoder or decoder}_{epoch-size}_{embedding-size}_{hidden-size}_{max-length}.torch
ENCODER_FILE_FORMAT = '{}/encoder_{}_{}_{}_{}.torch'
DECODER_FILE_FORMAT = '{}/decoder_{}_{}_{}_{}.torch'

CHECKPOINT_DIR = 'obj'

LOSS_FILE_FORMAT = '{}loss.dat'
VALIDATION_LOSS_FILE_FORMAT = '{}validation.dat'

## HELPER FUNCTIONS ##
# Converts a sentence into a list of indexes
def indexesFromSentence(book, sentence):
    return [book.word2index[word] for word in sentence.split()]

# Converts an index (integer) to a pytorch tensor
def tensorFromIndex(index, device):
    return torch.tensor(index, dtype=torch.long, device=device).view(-1, 1)

# Converts a sentence to a pytorch tensor
def tensorFromSentence(book, sentence, device):
    indexes = indexesFromSentence(book, sentence)
    indexes.append(STOP_ID)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Converts a pair of sentences into a pair of pytorch tensors
def tensorsFromPair(pair, book, device):
    input_tensor = tensorFromSentence(book, pair[0], device)
    target_tensor = tensorFromSentence(book, pair[1], device)
    return (input_tensor, target_tensor)

    # Calculates the BLEU score
def calculateBleu(candidate, reference, n_gram=2):
    # looks at ration of n-grams between 2 texts
    # Break candidate/reference into the format below
    candidate = candidate.split()
    reference = reference.split()
    return sentence_bleu(reference, candidate)#, weights=(1,0,0,0))
    
######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {:.2f}s'.format(m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '{} (- {})'.format(asMinutes(s), asMinutes(rs))
######################

# Object used in Beam Search to keep track of the results at each depth of the search tree
class BeamSearchResult:
        # score:   float of the sentence's current score
        # item:    the item (or list of items) to create the sentence with
        # hidden:  the current decoder hidden layer
        # stopped: has the sentence reached EOL or not?
        # words:   the items decoded into their corresponding strings
        def __init__(self, score, item, hidden):
                if isinstance(item, list):
                        self.items = item
                else:
                        self.items = [item]
                self.score = score
                self.hidden = hidden
                self.stopped = False
                self.words = []
        # Create a new BeamSearchResult with the values of 'self' and 'result'
        def add_result(self, result):
                new_result = BeamSearchResult(self.score + result.score, self.items + result.items, result.hidden)
                new_result.words += self.words
                return new_result
        # Returns the last element in the items list
        def get_latest_item(self):
                return self.items[-1]
        # Performs the perplexity calculation where summation = score and N = length of items
        def calculate_perplexity(self):
                return pow(math.e, -self.score / len(self.items))
        def __repr__(self):
                return 'BeamSearchResult: score={:.4f}, stopped={}, words="{}"'.format(self.score, str(self.stopped), ' '.join(self.words))

######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.

# Represents a Sequence to Sequence network made up of an encoder RNN and decoder RNN with attention weights
class Seq2Seq:
    def __init__(self, book, max_length, hidden_size, embedding_size, device):
        self.encoder = None
        self.decoder = None
        self.book = book
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.criterion = None
        self.log = log.Log()
        #for testing
        self.i = 0
        self.j = 0

    def loadFromFiles(self, encoder_filename, decoder_filename):
        # Check that the path for both files exists
        os.makedirs(os.path.dirname(encoder_filename), exist_ok=True)
        os.makedirs(os.path.dirname(decoder_filename), exist_ok=True)
        
        encoder_file = Path(encoder_filename)
        decoder_file = Path(decoder_filename)

        if encoder_file.is_file() and decoder_file.is_file():
            print("Loading encoder and decoder from files...")
            self.encoder = encoder.EncoderRNN(self.book.n_words, self.hidden_size, self.embedding_size).to(self.device)
            self.encoder.load_state_dict(torch.load(encoder_file, map_location=self.device))
            
            self.decoder = decoder.DecoderRNN(self.book.n_words, self.hidden_size, self.embedding_size, self.max_length).to(self.device)
            self.decoder.load_state_dict(torch.load(decoder_file, map_location=self.device))

            return True
        return False

    def saveToFiles(self, encoder_filename, decoder_filename):
        # Check that the path for both files exists
        os.makedirs(os.path.dirname(encoder_filename), exist_ok=True)
        os.makedirs(os.path.dirname(decoder_filename), exist_ok=True)

        encoder_file = Path(encoder_filename)
        decoder_file = Path(decoder_filename)

        torch.save(self.encoder.state_dict(), encoder_file)
        torch.save(self.decoder.state_dict(), decoder_file)
    
    def _train(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        encoder_hidden = self.encoder.initHidden(self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            if ei < self.max_length:
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            else:
                print('Somehow we got ei={} for range({}) where max_length={}'.format(ei, input_length, self.max_length))

        decoder_input = torch.tensor([[START_ID]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == STOP_ID:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length
    
    # Calculates loss value without updating weights. Used for validation
    def _calculate_loss(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        # Make all copies
        start = time.time()
        encoder_optimizer_copy = copy.deepcopy(self.encoder_optimizer)
        decoder_optimizer_copy = copy.deepcopy(self.decoder_optimizer)
        encoder_copy = copy.deepcopy(self.encoder)
        decoder_copy = copy.deepcopy(self.decoder)
        criterion_copy = copy.deepcopy(self.criterion)
        stop = time.time()
        if self.j == 0:
            t = (stop-start) * 1000
            print('Copies took {:.4f} ms'.format(t))
            self.j+=1
        
        # Begin regular function with object copies
        encoder_optimizer_copy.zero_grad()
        decoder_optimizer_copy.zero_grad()
    
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
    
        encoder_hidden = encoder_copy.initHidden(self.device)
        encoder_outputs = torch.zeros(self.max_length, encoder_copy.hidden_size, device=self.device)
    
        loss = 0
    
        for ei in range(input_length):
            # Create a copy of the encoder?
            if ei < self.max_length:
                encoder_output, encoder_hidden = encoder_copy(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            else:
                print('(In _calculate_loss): Somehow we got ei={} for range({}) where max_length={}'.format(ei, input_length, self.max_length))
    
        decoder_input = torch.tensor([[START_ID]], device=self.device)
        decoder_hidden = encoder_hidden
    
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                # Create a copy of the decoder?
                decoder_output, decoder_hidden, decoder_attention = decoder_copy(decoder_input, decoder_hidden, encoder_outputs)
                # Create a copy of the criterion?
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
    
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # Create a copy of the decoder?
                decoder_output, decoder_hidden, decoder_attention = decoder_copy(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1) # TODO: add beam search here? 
                decoder_input = topi.squeeze().detach()  # detach from history as input
    
                # Create a copy of the criterion?
                loss += criterion_copy(decoder_output, target_tensor[di])
                if decoder_input.item() == STOP_ID:
                    break
    
        # What does this do?
        loss.backward()
    
        # Don't do this?
        encoder_optimizer_copy.step()
        decoder_optimizer_copy.step()
    
        return loss.item() / target_length
    
    ######################################################################
    # The whole training process looks like this:
    #
    # -  Start a timer
    # -  Initialize optimizers and criterion
    # -  Create set of training pairs
    # -  Start empty losses array for plotting
    #
    # Then we call "train" many times and occasionally print the progress (%
    # of examples, time so far, estimated time) and average loss.
    def train_model(self, train_pairs, epochs, validation_size=0.1, validate_every=10, embedding_type=None, save_temp_models=False, checkpoint_every=25, loss_dir=None, print_every=1000, learning_rate=0.01):
        logfile = self.log.create('seq2seq-train-model')

        # Set folder for checkpoints
        if embedding_type is not None:
            if embedding_type == 'glove':
                CHECKPOINT_DIR = 'obj_glove'
            elif embedding_type == 'sg':
                CHECKPOINT_DIR = 'obj_sg'
            elif embedding_type == 'cbow':
                CHECKPOINT_DIR = 'obj_cbow'
            else:
                print('Incorrect embedding type given! Please choose one of ["glove", "sg", "cbow"]')
                exit()

        # Check if any checkpoints for this model exist:
        encoders = set()
        decoders = set()
        re_format = '{}_(\d+)_{}_{}_{}.torch'

        for filename in os.listdir('{}/'.format(CHECKPOINT_DIR)):
            r_enc = re.search(re_format.format('encoder', self.embedding_size, self.hidden_size, self.max_length), filename)
            if r_enc:
                encoders.add(int(r_enc.group(1)))
            else:
                r_dec = re.search(re_format.format('decoder', self.embedding_size, self.hidden_size, self.max_length), filename)
                if r_dec:
                    decoders.add(int(r_dec.group(1)))
        # A checkpoint needs a valid encoder and decoder 
        checkpoints = encoders.intersection(decoders)
        print('Checkpoints found at: {}'.format(checkpoints))
        self.log.debug(logfile, 'Checkpoints found at: {}'.format(checkpoints))
        start_epoch = 0

        found_max_checkpoint = False
        while not found_max_checkpoint:
            if len(checkpoints) > 0:
                max_val = max(checkpoints)
                if max_val < epochs:
                    start_epoch = max_val
                    print('Found checkpoint at epoch={}'.format(start_epoch))
                    self.log.debug(logfile, 'Found checkpoint at epoch={}'.format(start_epoch))
                    found_max_checkpoint = True
                else:
                    checkpoints.remove(max_val)
            else:
                found_max_checkpoint = True # the max is 0 (none exists)
                print('No checkpoint found')
                self.log.debug(logfile, 'No checkpoint found')

        loss_avgs = []
        validation_loss_avgs = []

        # If we didn't load the encoder/decoder from files: create new ones or load checkpoint to train
        if self.encoder is None or self.decoder is None:
            if start_epoch > 0:
                # Load the encoder/decoder for the starting epoch checkpoint
                encoder_filename = ENCODER_FILE_FORMAT.format(CHECKPOINT_DIR, start_epoch, self.embedding_size, self.hidden_size, self.max_length)
                decoder_filename = DECODER_FILE_FORMAT.format(CHECKPOINT_DIR, start_epoch, self.embedding_size, self.hidden_size, self.max_length)
                if self.loadFromFiles(encoder_filename, decoder_filename):
                    self.log.info(logfile, 'Loaded encoder/decoder from files at checkpoint {}'.format(start_epoch))
                else:
                    self.log.error(logfile, 'Tried to load checkpoint encoder/decoder at epoch={}, but it failed!'.format(start_epoch))
                    print('Checkpoint loading error!')
                    exit(1)
                # Load the loss values from files, if given
                if loss_dir is not None:
                    # Add a forward slash to end of directory path
                    if not loss_dir[-1] == '/':
                        loss_dir += '/'
                    self.log.info(logfile, 'Attempting to load loss files from {}'.format(loss_dir))
                    loss_filename = LOSS_FILE_FORMAT.format(loss_dir)
                    validation_loss_filename = VALIDATION_LOSS_FILE_FORMAT.format(loss_dir)
                    # Add (epoch, loss value) pairs to the loss lists
                    if os.path.isfile(loss_filename):
                        self.log.debug(logfile, 'Loading loss file: {}'.format(loss_filename))
                        print('Loading loss file: {}'.format(loss_filename))
                        with open(loss_filename, 'r') as f:
                            for line in f.readlines(): # should just have one line
                                for pair in line.split('\t'):
                                    if len(pair) > 0:
                                        epoch, value = pair.strip().split(',', 1)
                                        loss_avgs.append((int(epoch), float(value)))
                        # Save the values in the new log directory
                        with open(LOSS_FILE_FORMAT.format(self.log.dir), 'w+') as f:
                            for item in loss_avgs:
                                f.write('{},{}\t'.format(item[0], item[1]))
                    if os.path.isfile(validation_loss_filename):
                        self.log.debug(logfile, 'Loading validation loss file: {}'.format(validation_loss_filename))
                        print('Loading validation loss file: {}'.format(validation_loss_filename))
                        with open(validation_loss_filename, 'r') as f:
                            for line in f.readlines(): # should just have one line
                                for pair in line.split('\t'):
                                    if len(pair) > 0:
                                        epoch, value = pair.strip().split(',', 1)
                                        validation_loss_avgs.append((int(epoch), float(value)))
                        # Save the values in the new log directory
                        with open(VALIDATION_LOSS_FILE_FORMAT.format(self.log.dir), 'w+') as f:
                            for item in validation_loss_avgs:
                                f.write('{},{}\t'.format(item[0], item[1]))
            else:
                self.encoder = encoder.EncoderRNN(self.book.n_words, self.hidden_size, self.embedding_size).to(self.device)
                self.decoder = decoder.DecoderRNN(self.book.n_words, self.hidden_size, self.embedding_size, self.max_length).to(self.device)

        # Create the GloVe embedding's weight matrix:
        if embedding_type is not None:
            # Generates a dict of a word to its GloVe vector
            words2vec = glove.generate_glove(dim_size=self.embedding_size, embedding_type=embedding_type)
            # Create weight matrix:
            weights_matrix = np.zeros((self.book.n_words, self.embedding_size))
            words_found = 0
            for word in self.book.word2index:
                idx = self.book.word2index[word]
                try:
                    weights_matrix[idx] = words2vec[word]
                    words_found += 1
                except KeyError:
                    # Create random vector of dimension 'embedding_size', scale=0.6 taken from tutorial
                    weights_matrix[idx] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
            # Convert weights_matrix to a Tensor
            weights_matrix = torch.tensor(weights_matrix, device=self.device)
            print('We found {}/{} words in our GloVe words2vec dict!'.format(words_found, self.book.n_words))
            self.log.info(logfile, 'Found {}/{} words in the GloVe dict.'.format(words_found, self.book.n_words))
            # Set the embedding layer's state_dict for encoder and decoder
            self.encoder.embedding.load_state_dict({'weight': weights_matrix})
            self.decoder.embedding.load_state_dict({'weight': weights_matrix})
            self.log.info(logfile, 'Created encoder and decoder embeddings')

        start = time.time()
        print_loss_total = 0  # Reset every print_every

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        
        training_pairs = [tensorsFromPair(train_pair, self.book, self.device) for train_pair in train_pairs]
        
        # Create the validation set
        if validation_size > 1:
            self.log.error(logfile, 'Validation size must be less than 1, given={}'.format(validation_size))
            print('The validation size must be less than 1 (percentage of the training set)')
            exit(1)
        # This should use the same pairs if same train_pairs is passed
        validation_pairs = training_pairs[:int(len(training_pairs)*validation_size)]
        
        random.shuffle(training_pairs) # shuffle the train pairs
        
        self.criterion = nn.NLLLoss()

        # Iterate through the training set over a set amount of epochs
        # Output the progress and current loss value
        for i in range(start_epoch, epochs):
            self.log.debug(logfile, 'Processing epoch {}'.format(i))
            loss_avg = 0
            for j, pair in enumerate(training_pairs):
                input_tensor = pair[0]
                target_tensor = pair[1]
                loss = self._train(input_tensor, target_tensor)
                loss_avg += loss
                print_loss_total += loss

                if j % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    progress_percent = ((i-start_epoch)*len(training_pairs)+j)/((epochs-start_epoch)*len(training_pairs))
                    t = -1.0
                    if progress_percent > 0:
                        t = timeSince(start, progress_percent)
                    print('{} ({} {:.2f}%) {:.4f}'.format(t, ((i-start_epoch)*len(training_pairs)+j), progress_percent * 100, print_loss_avg))

            # Calculate loss on validation set:
            if i > 0 and i % validate_every == 0:
                validation_loss_avg = 0
                for j, pair in enumerate(validation_pairs):
                    input_tensor = pair[0]
                    target_tensor = pair[1]
                    validation_loss_avg += self._calculate_loss(input_tensor, target_tensor)
                # Save validation loss value
                validation_loss_avg /= len(validation_pairs)
                validation_loss_avgs.append((i, validation_loss_avg))
                with open(VALIDATION_LOSS_FILE_FORMAT.format(self.log.dir), 'a+') as f:
                    f.write('{},{}\t'.format(i, validation_loss_avg))

            # Save a checkpoint
            if save_temp_models:
                encoder_filename = ENCODER_FILE_FORMAT.format(CHECKPOINT_DIR, i+1, self.embedding_size, self.hidden_size, self.max_length)
                decoder_filename = DECODER_FILE_FORMAT.format(CHECKPOINT_DIR, i+1, self.embedding_size, self.hidden_size, self.max_length)
                encoder_file = Path(encoder_filename)
                decoder_file = Path(decoder_filename)
                # Save model at current epoch if doesn't exist
                if not encoder_file.is_file() or not decoder_file.is_file():
                    self.log.debug(logfile, 'Saving temporary model at epoch={}'.format(i))
                    self.saveToFiles(encoder_filename, decoder_filename)
                # Delete second previous model if not a multiple of 10
                if i > 1 and (i-1) % checkpoint_every != 0:
                    # Delete model with epoch = i-1
                    encoder_file = Path(ENCODER_FILE_FORMAT.format(CHECKPOINT_DIR, i-1, self.embedding_size, self.hidden_size, self.max_length))
                    decoder_file = Path(DECODER_FILE_FORMAT.format(CHECKPOINT_DIR, i-1, self.embedding_size, self.hidden_size, self.max_length))
                    if encoder_file.is_file() and decoder_file.is_file():
                        encoder_file.unlink()
                        decoder_file.unlink()
                        self.log.debug(logfile, 'Deleted temporary model at epoch={}'.format(i-1))
                    else:
                        self.log.error(logfile, 'Could not find temporary model at epoch={}'.format(i-1))
            
            # Save loss value
            loss_avg /= len(training_pairs)
            loss_avgs.append((i, loss_avg))
            with open(LOSS_FILE_FORMAT.format(self.log.dir), 'a+') as f:
                    f.write('{},{}\t'.format(i, loss_avg))

        # Save the entirety of the loss values
        loss_logfile = self.log.create('train_model-loss_values')
        self.log.debug(loss_logfile, 'Validation loss averages: {}'.format(validation_loss_avgs))
        self.log.debug(loss_logfile, 'Loss averages: {}'.format(loss_avgs))
        self.log.info(logfile, 'Finished training on data for {} epochs.'.format(epochs))
        self.log.debug(logfile, 'Average loss={:.4f}'.format(float(sum([item[1] for item in loss_avgs]))/len(loss_avgs)))
        self.log.debug(logfile, 'Average validation loss={:.4f}'.format(float(sum([item[1] for item in validation_loss_avgs])) / len(validation_loss_avgs)))
        plt.plot([item[0] for item in loss_avgs], [item[1] for item in loss_avgs], label='Training')
        plt.plot([item[0] for item in validation_loss_avgs], [item[1] for item in validation_loss_avgs], label='Validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('{}loss_figure.png'.format(self.log.dir)) # Save plot to log folder
        
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
    """ removed for now, had issues with passing data between functions???
    def _get_encoder_outputs(self, sentence):
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.input_book, sentence, self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden(self.device)

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            return encoder_outputs, encoder_hidden

    def _get_k_results(self, decoder_input, decoder_hidden, encoder_outputs, k):
        with torch.no_grad():
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(k)
            if self.i == 0:
                print('decoder_output.data.topk(k):\ntopv=%s\ntopi=%s'%(str(topv), str(topi)))
                self.i+=1

            topv = topv.squeeze()
            topi = topi.squeeze()

            # Store the k results in the format [(value_0, index_0, decoder_hidden_0), ..., (value_k, index_k, decoder_hidden_k)]
            return [(topv[i].item(), topi[i].item(), decoder_hidden) for i in range(topv.size()[0])]

            # Converting to BeamSearchResult class
            #return [BeamSearchResult(topv[i].item(), topi[i].item(), decoder_hidden) for i in range(topv.size()[0])]
    """
        # Performs beam search on the data to find better scoring sentences (evaluate uses a "beam search" of k=1)
    def beam_search(self, sentence, k):
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.book, sentence, self.device)
                
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden(self.device)

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOL

            decoder_hidden = encoder_hidden

            # Get initial k results:
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(k)
            results = []
            for i in range(k):
                # Score, item tensor, hidden layer
                results.append(BeamSearchResult(topv.squeeze()[i].item(), topi.squeeze()[i].detach(), decoder_hidden))
            ###
            # Expand the search for topk for each result until we have 5 sentences:
            sentence_length = 0
            while sentence_length <= self.max_length:
                new_results = [] # We will have k*k results in this after for-loop, then sort and take best k
                still_searching = False
                for result in results:
                    if not result.stopped:
                        still_searching = True
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(result.get_latest_item(), result.hidden, encoder_outputs)
                        topv, topi = decoder_output.data.topk(k)
                        for i in range(k):
                            new_result = result.add_result(BeamSearchResult(topv.squeeze()[i].item(), topi.squeeze()[i].detach(), decoder_hidden))
                            # If the next generated word is EOL, stop the sentence
                            if topi.squeeze()[i].item() == STOP_ID:
                                new_result.stopped = True
                            else:
                                new_result.words.append(self.book.index2word[topi.squeeze()[i].item()])
                            new_results.append(new_result)
                    else: # make sure to re-add currently stopped sentences
                        new_results.append(result)
                results = sorted(new_results, key=operator.attrgetter('score'))[::-1][:k]
                if not still_searching:
                    break
                # Prevent beam_search from being stuck in an infinite loop
                sentence_length += 1
            ###
            
            return results

    # Forces the model to generate the 'sentence_to_evaluate' and records its perplexity per word
    def _evaluate_specified(self, sentence, sentence_to_evaluate):
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.book, sentence, self.device)
                
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden(self.device)

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOL

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            summation = 0.0
            N = 0

            evaluate_tensor = tensorFromSentence(self.book, sentence_to_evaluate, self.device)

            evaluate_items = [t.item() for t in evaluate_tensor.squeeze()]
            for evaluate_item in evaluate_items:
                N += 1
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                #decoder_attentions[di] = decoder_attention.data
                
                topv, topi = decoder_output.data.topk(1)

                ## Perplexity code ##
                # We need to get the value for the item we're evaluating, not what was predicted:
                # decoder_output.data is of form tensor([[x, y, ..., z]]) where each value is a log value
                # The index in this tensor is the index of the word in the book
                summation += decoder_output.data.squeeze()[evaluate_item].item()
                
                if evaluate_item == STOP_ID:
                    break
                else:
                    # Decode the predicted word from the book 
                    decoded_words.append(self.book.index2word[topi.item()])

                decoder_input = tensorFromIndex(evaluate_item, self.device)

            perplexity = pow(math.e, -summation / N)# / N because the evaluate sentence is converted to a tensor where the last item will be STOP_ID

            # note: decoder_attentions not properly set up in this function

            #return decoded_words, decoder_attentions, perplexity
            return perplexity

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.book, sentence, self.device)
                
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden(self.device)

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOL

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                
                # output.data contains all log probabilities and distribution(?)
                # force the program to generate a sentence and record the probability of doing so
                topv, topi = decoder_output.data.topk(1)

                if topi.item() == STOP_ID:
                    #decoded_words.append('<EOL>')
                    break
                else:
                    # Decode the predicted word from the book 
                    decoded_words.append(self.book.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]
    
    ######################################################################
    # Evaluates each pair given in the test set
    # Returns BLEU, METEOR, and perplexity scores
    def evaluate_test_set(self, test_pairs, use_beam_search=True, k=5, print_first=15, print_every=100):
        logfile = self.log.create('evaluate_test_set')

        print('Printing first {} evaluations:'.format(print_first))
        self.log.info(logfile, 'Using {}'.format('beam search (k={})'.format(k) if use_beam_search else 'evaluate'))
        
        perplexity_total = 0.0
        
        bleu_total = 0.0
        meteor_total = 0.0

        beam_bleu_total = 0.0
        beam_meteor_total = 0.0

        # Keep track of how many empty sentences were generated        
        empty_sentences = 0
        beam_empty_sentences = 0
        
        # Loop through the test set
        for i, test_pair in enumerate(test_pairs):
            if (i+1)%print_every == 0:
                self.log.info(logfile, 'Evaluating test pair {}'.format(i+1))
                self.log.debug(logfile, '> {}'.format(test_pair[0]))
                self.log.debug(logfile, '= {}'.format(test_pair[1]))
            
            # Calculate perplexity of the test pair
            perplexity = self._evaluate_specified(test_pair[0], test_pair[1])
            perplexity_total += perplexity
            
            #Predict using beam search
            if use_beam_search:
                k = 5
                beam_results = self.beam_search(test_pair[0], k) # Get the best result
                # TODO: get first result that isn't empty
                skipped_num = 0
                beam_result = None
                for result in beam_results:
                    if len(result.words) > 0:
                        beam_result = result
                        break
                    skipped_num += 1
                if beam_result is None:
                    print('We outputted an empty beam sentence! Skipping...')
                    print('Test pair:\n\t> [{}]\n\t= [{}]'.format(test_pair[0], test_pair[1]))
                    beam_empty_sentences += 1
                    continue
                beam_sentence = ' '.join(beam_result.words)
                if (i+1)%print_every == 0:
                    self.log.debug(logfile, '< {}'.format(beam_sentence))
                    print('{}: {:.4f}'.format(i, perplexity))
                
                # Calculate BLEU and METEOR for beam search
                beam_bleu_score = 0#calculateBleu(beam_sentence, test_pair[1])
                beam_meteor_score = 0#pymeteor.meteor(beam_sentence, test_pair[1])
                # Add to totals
                beam_bleu_total += beam_bleu_score
                beam_meteor_total += beam_meteor_score
                
                # Print first print_first results
                if i < print_first:
                    print('> [{}]'.format(test_pair[0]))
                    print('= [{}]'.format(test_pair[1]))
                    print('\tPerplexity: {:.4f}'.format(perplexity))
                    print('Beam search (k={}):'.format(k))
                    print('< [{}]'.format(beam_sentence))
                    print('\tSkipped sentences: {}'.format(beam_empty_sentences))
                    print('\tBLEU:       {:.4f}'.format(beam_bleu_score))
                    print('\tMETEOR:     {:.4f}'.format(beam_meteor_score))
            else:
                # Predict using evaluate
                output_words, attentions = self.evaluate(test_pair[0])
                if len(output_words) == 0:
                    print('We outputted an empty sentence! Skipping...')
                    print('Test pair:\n\t> [{}]\n\t= [{}]'.format(test_pair[0], test_pair[1]))
                    empty_sentences += 1
                    continue
                output_sentence = ' '.join(output_words)
                if (i+1)%print_every == 0:
                    self.log.debug(logfile, '< {}'.format(output_sentence))
                    print('{}: {:.4f}'.format(i, perplexity))
                # Calculate BLEU and METEOR for evaluate
                bleu_score = 0#calculateBleu(output_sentence, test_pair[1])
                meteor_score = 0#pymeteor.meteor(output_sentence, test_pair[1])
                # Add to totals
                bleu_total += bleu_score
                meteor_total += meteor_score
            
        self.log.info(logfile, 'Finished evaluating test set\n')
        
        # Calculate the average perplexity score
        avg_perplexity = perplexity_total / len(test_pairs)
        print('\tAverage Perplexity per word = {:.4f}'.format(avg_perplexity))
        
        if use_beam_search:
            # Calculate averages for beam search
            beam_avg_bleu = beam_bleu_total / (len(test_pairs) - beam_empty_sentences)
            beam_avg_meteor = beam_meteor_total / (len(test_pairs) - beam_empty_sentences)
            print('Beam search (k=%d):'%k)
            print('\tPredicted a total of %d empty sentences.'%beam_empty_sentences)
            print('\tAverage BLEU score = ' + str(beam_avg_bleu))
            print('\tAverage METEOR score = ' + str(beam_avg_meteor))
            self.log.info(logfile, '\n\tAverage perplexity={:.4f}\n\tGenerated {} empty sentences\n\tAverage BLEU Score={:.4f}\n\tAverage METEOR score={:.4f}'.format(
                avg_perplexity, beam_empty_sentences, beam_avg_bleu, beam_avg_meteor))
            
            return avg_perplexity, beam_avg_bleu, beam_avg_meteor
        else:
            # Calculate averages for evaluate    
            avg_bleu = bleu_total / (len(test_pairs) - empty_sentences)
            avg_meteor = meteor_total / (len(test_pairs) - empty_sentences)
        
            print('evaluate:')
            print('\tPredicted a total of %d empty sentences.'%empty_sentences)
            print('\tAverage BLEU score = ' + str(avg_bleu))
            print('\tAverage METEOR score = ' + str(avg_meteor))
            self.log.info(logfile, '\n\tAverage perplexity={:.4f}\n\tGenerated {} empty sentences\n\tAverage BLEU Score={:.4f}\n\tAverage METEOR score={:.4f}'.format(
                avg_perplexity, empty_sentences, avg_bleu, avg_meteor))
        
            return avg_perplexity, avg_bleu, avg_meteor
