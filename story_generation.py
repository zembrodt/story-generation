# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch

import pickle
from pathlib import Path

import nltk
import collections

import storygen.seq2seq as seq2seq
import storygen.book as book
import storygen.perplexity as perplexity

# Consts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 1024
HIDDEN_SIZE = 256
ENCODER_FILE_FORMAT = "obj/encoder_%d.p"
DECODER_FILE_FORMAT = "obj/decoder_%d.p"

# Read all the lines from a book and convert them to an array of sentences
def getSentencesFromBook(book_title):
	print('Reading book...')
	
	lines = open('data/%s.txt' % book_title, encoding='utf-8').read().strip().split('\n')
	lines = [book.normalizeString(line) for line in lines if len(line) > 0]
	with open('data/contractions_dictionary.txt', 'r') as f:
		s = f.read()
		contractions = eval(s)
	contraction_dict = book.ContractionDict(contractions)
	
	return book.convertLinesToSentences(lines, contraction_dict)

# Convert all sentences into pairs and split into training and testing data
def getPairs(sentences):
	pairs = [[sentences[i], sentences[i+1]] for i,_ in enumerate(sentences[:len(sentences)-1])]
	
	random.shuffle(pairs)
	train_size = int(len(pairs)*0.8)
	train_pairs = pairs[:train_size]
	test_pairs = pairs[train_size:]
	
	return train_pairs, test_pairs

def getBooks(book_title, train_pairs, test_pairs):
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

def getSeq2SeqNetwork(encoder_filename, decoder_filename, input_book, output_book, train_pairs, test_pairs):
	
	
	encoder_file = Path(encoder_filename)
	decoder_file = Path(decoder_filename)
	network = None

	if encoder_file.is_file() and decoder_file.is_file():
		print("Loading encoder and decoder from files...")
		encoder = pickle.load(open(encoder_filename, "rb"))
		decoder = pickle.load(open(decoder_filename, "rb"))
		network = seq2seq.Seq2Seq(encoder, decoder, train_pairs, test_pairs, MAX_LENGTH)
	else:
		print("Training encoder and decoder...")
		encoder = seq2seq.EncoderRNN(input_book.n_words, HIDDEN_SIZE).to(DEVICE)
		decoder = seq2seq.DecoderRNN(HIDDEN_SIZE, output_book.n_words, MAX_LENGTH).to(DEVICE)
		network = seq2seq.Seq2Seq(encoder, decoder, train_pairs, test_pairs, MAX_LENGTH)
		
		ITER_AMOUNT = 10000
		network.trainIters(ITER_AMOUNT, input_book, output_book)
	
		## Dump to pickle ##
		print("Dumping encoder and decoder to files...")
		pickle.dump(encoder, open(encoder_filename, "wb"))
		pickle.dump(decoder, open(decoder_filename, "wb"))
	return network



def main():

	book_title = '1_sorcerers_stone'

	sentences = getSentencesFromBook(book_title)
	
	perplexity_model = perplexity.PerplexityModel(sentences)
	
	train_pairs, test_pairs = getPairs(sentences)
	print('TEST')
	input_book, output_book = getBooks(book_title, train_pairs, test_pairs)
	print('TEST2')
	epoch_sizes = [0]#[100, 200, 400, 600]
	for epoch_size in epoch_sizes:
		print('Epoch size: %d' % epoch_size)
		encoder_filename = ENCODER_FILE_FORMAT % epoch_size
		decoder_filename = DECODER_FILE_FORMAT % epoch_size
	
		network = getSeq2SeqNetwork(encoder_filename, decoder_filename, input_book, output_book, train_pairs, test_pairs)
	
		#bleu_score, perplexity_score = evaluateRandomly(encoder1, attn_decoder1, test_pairs, input_book, output_book, perplexity_model)
		bleu_score, perplexity_score = network.evaluateRandomly(input_book, output_book, perplexity_model)
		print('BLEU Score for %d epochs: %.4f' % (bleu_score, epoch_size))
		print('Perplexity score for %d epochs: %.4f' % (perplexity_score, epoch_size))

		# Generate test story
		"""
		story1 = createStory("You're a wizard Harry .", input_book, output_book, encoder1, attn_decoder1)
		for line in story1:
			print('> %s' % line)
		"""

main()

