# story_generation.py

from __future__ import unicode_literals, print_function, division
from io import open
from pathlib import Path

import collections
import random
import re
import string
import unicodedata

import nltk
import pymeteor.pymeteor as pymeteor
import torch

from storygen import *

#Consts
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 1024
HIDDEN_SIZE = 256
ENCODER_FILE_FORMAT = 'obj/encoder_%d.torch'
DECODER_FILE_FORMAT = 'obj/decoder_%d.torch'
TRAIN_TEST_DATA_FILE_FORMAT = 'data/%s_%d_%s.txt'

# Read all the lines from a book and convert them to an array of sentences
def get_pairs(book_title, percentage):
	print('Reading book...')
	train_pairs = None
	test_pairs = None
	# Check if the train/test data exists for the given book with the given percentage
	train_file = Path(TRAIN_TEST_DATA_FILE_FORMAT % (book_title, int(percentage*100), 'train'))
	test_file = Path(TRAIN_TEST_DATA_FILE_FORMAT % (book_title, int(percentage*100), 'test'))
	if train_file.is_file() and test_file.is_file():
		print('Reading from file')
		# Read train/test data from the files
		train_data = train_file.open(mode='r').read().split('\n')
		test_data = test_file.open(mode='r').read().split('\n')
		print('len(train/test_data)=%d, %d'%(len(train_data), len(test_data)))
		train_iter = iter(train_data)
		train_pairs = [item for item in zip(train_iter, train_iter)]
		test_iter = iter(test_data)
		test_pairs = [item for item in zip(test_iter, test_iter)]
	else:
		print('Writing to file')
		lines = open('data/%s.txt' % book_title, encoding='utf-8').read().strip().split('\n')
		lines = [book.normalizeString(line) for line in lines if len(line) > 0]
		with open('data/contractions_dictionary.txt', 'r') as f:
			s = f.read()
			contractions = eval(s)
		contraction_dict = book.ContractionDict(contractions)
		sentences = book.convertLinesToSentences(lines, contraction_dict)
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
		f = train_file.open(mode='w')
		for i, item in enumerate(train_data):
			if i+1 < len(train_data):
				f.write('%s\n'%item)
			else:
				f.write('%s'%item)
		f.close()
		# Write test data
		f = test_file.open('w')
		for i, item in enumerate(test_data):
			if i+1 < len(test_data):
				f.write('%s\n'%item)
			else:
				f.write('%s'%item)
		f.close()
	print('len(train/test)=%d, %d'%(len(train_pairs), len(test_pairs)))
	print('train[0]=%s\ntest[0]=%s'%(str(train_pairs[0]), str(test_pairs[0])))
	return train_pairs, test_pairs

def get_books(book_title, train_pairs, test_pairs):
        input_book = book.Book("input_%s" % book_title)
        output_book = book.Book("output_%s" % book_title)

        for pair in train_pairs + test_pairs:
                input_book.addSentence(pair[0])
                output_book.addSentence(pair[1])
                
        return input_book, output_book
	
"""
def createStory(input_sentence, input_book, output_book, encoder, decoder):
    #input_sentence = normalizeString(input_sentence)
	#TODO: create a way to convert this string?
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
"""

def main():

	book_title = '1_sorcerers_stone'
	
	train_pairs, test_pairs = get_pairs(book_title, 0.8)
        
	input_book, output_book = get_books(book_title, train_pairs, test_pairs)
	
	epoch_sizes = [25]#[100, 200, 400, 600]
	for epoch_size in epoch_sizes:
		print('Epoch size: %d' % epoch_size)
		encoder_filename = ENCODER_FILE_FORMAT % epoch_size
		decoder_filename = DECODER_FILE_FORMAT % epoch_size

		network = seq2seq.Seq2Seq(input_book, output_book, MAX_LENGTH, HIDDEN_SIZE, DEVICE)
		if not network.loadFromFiles(encoder_filename, decoder_filename):
			network.trainIters(train_pairs, epoch_size)
			network.saveToFiles(encoder_filename, decoder_filename)
		
		bleu_score, meteor_score, perplexity_score, beam_bleu_score, beam_meteor_score, beam_perplexity_score = network.evaluateTestSet(test_pairs)
		
		print('evaluate:')
		print('\tBLEU Score for %d epochs: %.4f' % (epoch_size, bleu_score))
		print('\tMETEOR Score for %d epochs: %.4f' % (epoch_size, meteor_score))
		print('\tPerplexity score for %d epochs: %.4f' % (epoch_size, perplexity_score))

		print('Beam search:')
		print('\tBLEU Score for %d epochs: %.4f' % (epoch_size, beam_bleu_score))
		print('\tMETEOR Score for %d epochs: %.4f' % (epoch_size, beam_meteor_score))
		print('\tPerplexity score for %d epochs: %.4f' % (epoch_size, beam_perplexity_score))

		# Generate test story
		"""
		story1 = createStory("You're a wizard Harry .", input_book, output_book, encoder1, attn_decoder1)
		for line in story1:
			print('> %s' % line)
		"""

if __name__ == '__main__':
        main()

