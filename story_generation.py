#!/usr/bin/env python3

# story_generation.py

from io import open
from pathlib import Path

import random
import sys
import torch
import getopt

from storygen import *

# Constants used to create the sequence 2 sequence network
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = None
HIDDEN_SIZE = 256
EMBEDDING_SIZE = glove.DIMENSION_SIZES[-1]
DATA_FILE_FORMAT = 'data/{}_{}_{}.txt'
EMBEDDINGS = ['glove', 'cbow', 'sg']

# Help message for command line arguments
HELP_MSG = '\n'.join([
		'Usage:',
		'python3 story_generation.py [-h, --help] [--epoch <epoch_value>] [--embedding <embedding_type>] [--loss <loss_dir>]',
		'\tAll command line arguments are optional, and any combination (beides -h) can be used',
		'\t-h, --help: Provides help on command line parameters',
		'\t--epoch <epoch_value>: specify an epoch value to train the model for or load a checkpoint from',
		'\t--embedding <embedding_type>: specify an embedding to use from: [glove, cbow, sg]',
		'\t--loss <loss_dir>: specify a directory to load loss values from (requires files loss.dat and validation.dat)',
		'\t-u, --update: specify that if previous train/test pairs exists, overwrite them with re-parsed data'])

# Creates sentences from the given book (text file)
def get_sentences(book_title, pre_parsed=False):
	global MAX_LENGTH # We set MAX_LENGTH to the longest sentence within the book
	if pre_parsed:
		with open('data/books/{}.txt'.format(book_title)) as f:
			sentences = f.read().strip().split('\n')
		# Set MAX_LENGTH const:
		sentences_split = [sentence.split() for sentence in sentences]
		MAX_LENGTH = max(map(len, sentences_split)) + 1
		print('max_length={}'.format(MAX_LENGTH))
		return sentences
	else:
		# Create contractions dictionary
		with open(book.CONTRACTIONS_FILE, 'r') as f:
			s = f.read()
			contractions = eval(s)
		contraction_dict = book.ContractionDict(contractions)
		
		# Open the given text files (ignore bytes not in utf-8)
		with open('data/{}.txt'.format(book_title), encoding='utf-8', errors='ignore') as f:
			lines = f.read().strip().split('\n')
		sentences = book.convertLinesToSentences(lines, contraction_dict)

		# Set MAX_LENGTH const:		
		sentences_split = [sentence.split() for sentence in sentences]
		MAX_LENGTH = max(map(len, sentences_split)) + 1
		print('max sentence len={}'.format(MAX_LENGTH))

		return sentences
		
# Read all the lines from a book and convert them to an array of sentences
def get_pairs(book_title, percentage, pre_parsed=False, load_previous=True):
	print('Reading book...')
	train_pairs = None
	test_pairs = None
	# Check if the train/test data exists for the given book with the given percentage
	train_file = Path(DATA_FILE_FORMAT.format(book_title, int(percentage*100), 'train'))
	test_file = Path(DATA_FILE_FORMAT.format(book_title, int(percentage*100), 'test'))

	# load_previous:
	# Gives the user the option to re-parse the train/test pairs
	# Useful if parse methods have been updated since when the pairs were created,
	# if the user wants to reshuffle the pairs, or if the data in the input file has changed

	if train_file.is_file() and test_file.is_file() and load_previous:
		print('Reading train/test data from file')
		# Read train/test data from the files
		train_data = train_file.open(mode='r').read().split('\n')
		test_data = test_file.open(mode='r').read().split('\n')
		print('len(train/test_data)={}, {}'.format(len(train_data), len(test_data)))
		train_iter = iter(train_data)
		train_pairs = [item for item in zip(train_iter, train_iter)]
		test_iter = iter(test_data)
		test_pairs = [item for item in zip(test_iter, test_iter)]
	else:
		print('Writing to train/test data to file')
		sentences = get_sentences(book_title, pre_parsed=pre_parsed)
		# Convert all sentences into pairs and split into training and testing data 
		sentences_iter = iter(sentences)
		next(sentences_iter)
		pairs = [(prev, curr) for prev, curr in zip(sentences, sentences_iter)]
		random.shuffle(pairs)
		train_size = int(len(pairs)*percentage)
		train_pairs = pairs[:train_size]
		test_pairs = pairs[train_size:]
		# write the train/test pairs into files
		# Condense the train/test pairs down into a singular list to write
		train_data = [item for pair in train_pairs for item in pair]
		test_data = [item for pair in test_pairs for item in pair]
		# Write train data
		with train_file.open(mode='w') as f:
			for i, item in enumerate(train_data):
				f.write(item)
				if i+1 < len(train_data):
					f.write('\n')
		# Write test data
		with test_file.open('w') as f:
			for i, item in enumerate(test_data):
				f.write(item)
				if i+1 < len(test_data):
					f.write('\n')
	
	print('len(train/test)={}, {}'.format(len(train_pairs), len(test_pairs)))
	print('train[0]={}\ntest[0]={}'.format(train_pairs[0], test_pairs[0]))
	return train_pairs, test_pairs

# Creates a book object from the given train/test pairs
def get_book(book_title, train_pairs, test_pairs):
	bk = book.Book(book_title)
	
	pairs = train_pairs + test_pairs
	for i, pair in enumerate(pairs):
		bk.addSentence(pair[0])
		if i+1 == len(pairs):
			bk.addSentence(pair[1])
	        
	return bk

def main(argv):
	global MAX_LENGTH

	book_title = '1_sorcerers_stone'

	# Get command line arguments
	try:
		opts, _ = getopt.getopt(argv, 'hu', ['epoch=', 'embedding=', 'loss=', 'help', 'update'])
	except getopt.GetoptError as e:
		print(e)
		print(HELP_MSG)
		exit(2)

	# Default values
	epoch_size = 100
	embedding_type = None
	loss_dir = None
	load_previous = True

	# Set values from command line
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print(HELP_MSG)
			exit()
		# How many epochs to train for
		elif opt == '--epoch':
			try:
				epoch_size = int(arg)
			except ValueError:
				print('{} is not an integer. Argument must be an int.'.format(arg))
				exit()
		# The type of embedding to use
		elif opt == '--embedding':
			embedding_type = arg
		# Directory to load previous loss values from
		elif opt == '--loss':
			loss_dir = arg
		# Flag to overwrite saved train/test pairs (if they exist)
		elif opt in ('-u', '--update'):
			load_previous = False

	print('Epoch size        = {}'.format(epoch_size))
	print('Embedding type    = {}'.format(embedding_type))
	print('Loss directory    = {}'.format(loss_dir))
	print('Hidden layer size = {}'.format(HIDDEN_SIZE))

	# Create (or load) train/test pairs from the given book (text file)
	train_pairs, test_pairs = get_pairs(book_title, 0.8, pre_parsed=False, load_previous=load_previous)
	print('max_len={}'.format(MAX_LENGTH))

	# Check that we set MAX_LENGTH:
	if MAX_LENGTH is None:
		MAX_LENGTH = max(
			max(map(len, [sentence.split() for pair in train_pairs for sentence in pair])),
			max(map(len, [sentence.split() for pair in test_pairs for sentence in pair])))
		MAX_LENGTH += 1 # for <EOL> token
	print('max_len2={}'.format(MAX_LENGTH))

	# Create a book object from the train/test pairs
	book = get_book(book_title, train_pairs, test_pairs)

	# Determine the directory to store and/or load checkpoints for the model
	print('Epoch size: {}'.format(epoch_size))
	obj_dir = 'obj'
	if embedding_type is not None and embedding_type in EMBEDDINGS:
		obj_dir = 'obj_{}'.format(embedding_type)
	else:
		print('Incorrect embedding type given! Please choose one of ["glove", "sg", "cbow"]')
		exit(1)
	
	# Set the encoder and decoder files to look for
	encoder_filename = seq2seq.ENCODER_FILE_FORMAT.format(obj_dir, epoch_size, EMBEDDING_SIZE, HIDDEN_SIZE, MAX_LENGTH)
	decoder_filename = seq2seq.DECODER_FILE_FORMAT.format(obj_dir, epoch_size, EMBEDDING_SIZE, HIDDEN_SIZE, MAX_LENGTH)
	print('enc_filename: {}'.format(encoder_filename))
	print('dec_filename: {}'.format(decoder_filename))

	# Train the seq2seq network on the training data
	network = seq2seq.Seq2Seq(book, MAX_LENGTH, HIDDEN_SIZE, EMBEDDING_SIZE, DEVICE)
	if not network.loadFromFiles(encoder_filename, decoder_filename):
		network.train_model(train_pairs, epoch_size, embedding_type=embedding_type, validate_every=25, save_temp_models=True, loss_dir=loss_dir)
		network.saveToFiles(encoder_filename, decoder_filename)
	
	# Evaluate the testing data
	perplexity_score, bleu_score, meteor_score = network.evaluate_test_set(test_pairs)

if __name__ == '__main__':
        main(sys.argv[1:])

