# story_generation.py

from io import open
from pathlib import Path

import random
import sys
import torch

from storygen import *

#Consts
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = None
HIDDEN_SIZE = 256
EMBEDDING_SIZE = glove.DIMENSION_SIZES[-1]
DATA_FILE_FORMAT = 'data/{}_{}_{}.txt'

def get_sentences(book_title):
	global MAX_LENGTH # We set MAX_LENGTH to the longest sentence within the book
	
	lines = open('data/%s.txt' % book_title, encoding='utf-8').read().strip().split('\n')
	lines = [book.normalizeString(line) for line in lines if len(line) > 0]
	with open('data/contractions_dictionary.txt', 'r') as f:
		s = f.read()
		contractions = eval(s)
	contraction_dict = book.ContractionDict(contractions)
	sentences = book.convertLinesToSentences(lines, contraction_dict)
	
	# Set MAX_LENGTH const:
	sentences_split = [sentence.split() for sentence in sentences]
	MAX_LENGTH = max(map(len, sentences_split)) + 1
	print('max sentence len={}'.format(MAX_LENGTH))
	
	return sentences
		
# Read all the lines from a book and convert them to an array of sentences
def get_pairs(book_title, percentage):
	print('Reading book...')
	train_pairs = None
	test_pairs = None
	# Check if the train/test data exists for the given book with the given percentage
	train_file = Path(DATA_FILE_FORMAT.format(book_title, int(percentage*100), 'train'))
	test_file = Path(DATA_FILE_FORMAT.format(book_title, int(percentage*100), 'test'))
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
		sentences = get_sentences(book_title)
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
	
	print('len(train/test)={}, {}'.format(len(train_pairs), len(test_pairs)))
	print('train[0]={}\ntest[0]={}'.format(train_pairs[0], test_pairs[0]))
	return train_pairs, test_pairs

def get_book(book_title, train_pairs, test_pairs):
	bk = book.Book(book_title)
	
	pairs = train_pairs + test_pairs
	for i, pair in enumerate(pairs):
		bk.addSentence(pair[0])
		if i+1 == len(pairs):
			bk.addSentence(pair[1])
	        
	return bk
	
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
	global MAX_LENGTH
	print('Hidden layer size: {}'.format(HIDDEN_SIZE))

	book_title = '1_sorcerers_stone'
	
	train_pairs, test_pairs = get_pairs(book_title, 0.8)
        
        
	# Check that we set MAX_LENGTH:
	if MAX_LENGTH is None:
		MAX_LENGTH = max(
			max(map(len, [sentence.split() for pair in train_pairs for sentence in pair])),
			max(map(len, [sentence.split() for pair in test_pairs for sentence in pair])))
		MAX_LENGTH += 1 # for <EOL> token
    
	#input_book, output_book = get_books(book_title, train_pairs, test_pairs)
	book = get_book(book_title, train_pairs, test_pairs)
	
	epoch_sizes = [2]#[100, 200, 400, 600]
	for epoch_size in epoch_sizes:
		print('Epoch size: {}'.format(epoch_size))
		encoder_filename = seq2seq.ENCODER_FILE_FORMAT.format(epoch_size, EMBEDDING_SIZE, HIDDEN_SIZE, MAX_LENGTH)
		decoder_filename = seq2seq.DECODER_FILE_FORMAT.format(epoch_size, EMBEDDING_SIZE, HIDDEN_SIZE, MAX_LENGTH)

		#network = seq2seq.Seq2Seq(input_book, output_book, MAX_LENGTH, HIDDEN_SIZE, DEVICE)
		network = seq2seq.Seq2Seq(book, MAX_LENGTH, HIDDEN_SIZE, EMBEDDING_SIZE, DEVICE)
		if not network.loadFromFiles(encoder_filename, decoder_filename):
			loss_dir = None
			if len(sys.argv) > 1:
				loss_dir = sys.argv[1]
			network.train_model(train_pairs, epoch_size, use_glove_embeddings=True, save_temp_models=True, loss_dir=loss_dir)
			network.saveToFiles(encoder_filename, decoder_filename)
		
		perplexity_score, bleu_score, meteor_score, beam_bleu_score, beam_meteor_score = network.evaluate_test_set(test_pairs)
		
		"""output in evaluateTestSet
		print('evaluate:')
		print('\tBLEU Score for %d epochs: %.4f' % (epoch_size, bleu_score))
		print('\tMETEOR Score for %d epochs: %.4f' % (epoch_size, meteor_score))
		print('\tPerplexity score for %d epochs: %.4f' % (epoch_size, perplexity_score))

		print('Beam search:')
		print('\tBLEU Score for %d epochs: %.4f' % (epoch_size, beam_bleu_score))
		print('\tMETEOR Score for %d epochs: %.4f' % (epoch_size, beam_meteor_score))
		print('\tPerplexity score for %d epochs: %.4f' % (epoch_size, beam_perplexity_score))
		"""

		# Generate test story
		"""
		story1 = createStory("You're a wizard Harry .", input_book, output_book, encoder1, attn_decoder1)
		for line in story1:
			print('> %s' % line)
		"""

if __name__ == '__main__':
        main()

