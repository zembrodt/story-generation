# seq2seq.py
import time
import random
import time
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu

import storygen.book as bk

## HELPER FUNCTIONS ##
def indexesFromSentence(book, sentence):
    return [book.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(book, sentence, device):
    indexes = indexesFromSentence(book, sentence)
    indexes.append(bk.STOP_ID)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_book, output_book, device):
    input_tensor = tensorFromSentence(input_book, pair[0], device)
    target_tensor = tensorFromSentence(output_book, pair[1], device)
    return (input_tensor, target_tensor)

	# Calculates the BLEU score
def calculateBleu(candidate, reference, n_gram=2):
    # looks at ration of n-grams between 2 texts
    # Break candidate/reference into the format below
    candidate = candidate.split(' ')
    reference = reference.split(' ')
    return sentence_bleu(reference, candidate)#, weights=(1,0,0,0))
	
######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

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
######################


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

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# Attention Decoder
# -----------------
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
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

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
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

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
		
class Seq2Seq:
	def __init__(self, train_pairs, test_pairs, max_length, hidden_size, device):
		self.encoder = None
		self.decoder = None
		self.train_pairs = train_pairs
		self.test_pairs = test_pairs
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.device = device
		self.encoder_optimizer = None
		self.decoder_optimizer = None
		self.criterion = None

		#for testing
		self.i = 0

	def loadFromFiles(self, encoder_filename, decoder_filename, input_book, output_book):
		# Check that the path for both files exists
		os.makedirs(os.path.dirname(encoder_filename), exist_ok=True)
		os.makedirs(os.path.dirname(decoder_filename), exist_ok=True)
		
		encoder_file = Path(encoder_filename)
		decoder_file = Path(decoder_filename)

		if encoder_file.is_file() and decoder_file.is_file():
			print("Loading encoder and decoder from files...")
                        self.encoder = EncoderRNN(input_book.n_words, self.hidden_size).to(self.device)
                        self.encoder.load_state_dict(torch.load(encoder_file))
                        
                        self.decoder = DecoderRNN(self.hidden_size, output_book.n_words, self.max_length).to(self.device)
                        self.decoder.load_state_dict(torch.load(decoder_file))

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
	
	def train(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
		encoder_hidden = self.encoder.initHidden(self.device)

		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		input_length = input_tensor.size(0)
		target_length = target_tensor.size(0)

		encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

		loss = 0

		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(
				input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[bk.START_ID]], device=self.device)

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
				if decoder_input.item() == bk.STOP_ID:
					break

		loss.backward()

		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return loss.item() / target_length
	
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
	def trainIters(self, n_iters, input_book, output_book, print_every=1000, learning_rate=0.01):
		if self.encoder is None or self.decoder is None:
			self.encoder = EncoderRNN(input_book.n_words, self.hidden_size).to(self.device)
			self.decoder = DecoderRNN(self.hidden_size, output_book.n_words, self.max_length).to(self.device)

		#n_iters == iterations
		#epochs = iterations / num of examples
		start = time.time()
		print_loss_total = 0  # Reset every print_every

		self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
		self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
		#training_pairs = [tensorsFromPair(random.choice(pairs))
		#                  for i in range(n_iters)]
		training_pairs = [tensorsFromPair(random.choice(self.train_pairs), input_book, output_book, self.device) for i in range(n_iters)]
		self.criterion = nn.NLLLoss()

		
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

			loss = self.train(input_tensor, target_tensor)#, encoder,
						 #decoder, encoder_optimizer, decoder_optimizer, criterion)
			print_loss_total += loss

			if iter % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
											 iter, iter / n_iters * 100, print_loss_avg))

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

	def evaluate(self, sentence, input_book, output_book):
		with torch.no_grad():
			input_tensor = tensorFromSentence(input_book, sentence, self.device)
			input_length = input_tensor.size()[0]
			encoder_hidden = self.encoder.initHidden(self.device)

			encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

			for ei in range(input_length):
				encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
														 encoder_hidden)
				encoder_outputs[ei] += encoder_output[0, 0]

			decoder_input = torch.tensor([[bk.START_ID]], device=self.device)  # SOL

			decoder_hidden = encoder_hidden

			decoded_words = []
			decoder_attentions = torch.zeros(self.max_length, self.max_length)

			for di in range(self.max_length):
				decoder_output, decoder_hidden, decoder_attention = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs)
				decoder_attentions[di] = decoder_attention.data
				# output.data contains all log probabilities and distribution(?)
				# force the program to generate a sentence and record the probability of doing so
				if self.i == 0:
				    print('decoder_output.data:\n%s'%str(decoder_output.data))
				    topv,topi=decoder_output.data.topk(1)
				    print('\ndecoder_output.data.topk(1):\n%s'%str(decoder_output.data.topk(1)))
				    print('\ttopv = %s\n\ttopi = %s\n\ttopi.item = %s\n\tword = %s'%(str(topv), str(topi), str(topi.item()), output_book.index2word[topi.item()]))
				    topv,topi=decoder_output.data.topk(2)
				    topi = topi[0][1]
				    print('\ndecoder_output.data.topk(2):\n%s'%str(decoder_output.data.topk(2)))
				    print('\ttopv = %s\n\ttopi = %s\n\ttopi.item = %s\n\tword = %s'%(str(topv), str(topi), str(topi.item()), output_book.index2word[topi.item()]))
				    self.i += 1
				topv, topi = decoder_output.data.topk(1)
				if topi.item() == bk.STOP_ID:
					# DON'T NEED? Will add this token to the end of some sentences
					# May ruin our BLEU/METEOR scores
					#decoded_words.append('<EOL>')
					break
				else:
					# Decode the predicted word from the book 
					decoded_words.append(output_book.index2word[topi.item()])

				decoder_input = topi.squeeze().detach()

			return decoded_words, decoder_attentions[:di + 1]
	
	######################################################################
	# We can evaluate random sentences from the training set and print out the
	# input, target, and output to make some subjective quality judgements:
	# THIS ACTUALLY NOW EVALUATES ALL SENTENCES IN THE TEST SET
	def evaluateRandomly(self, input_book, output_book, perplexity_model):
		i = 0
		print('Printing first 10 evaluations:')
		bleu_total = 0.0
		perplexity_total = 0.0
		empty_sentences = 0
		for test_pair in self.test_pairs:
			output_words, attentions = self.evaluate(test_pair[0], input_book, output_book)
			if len(output_words) == 0:
			    print('We outputted an empty sentence! Skipping...')
			    print('Test pair:\n\t>%s\n\t=%s'%(test_pair[0], test_pair[1]))
			    empty_sentences += 1
			    continue
			output_sentence = ' '.join(output_words)
			# TODO: need to access this function somehow
			bleu_score = calculateBleu(output_sentence, test_pair[1])
			perplexity_score = perplexity_model.perplexity(output_sentence)
			#TODO:
			#meteor_score = ...
			if i < 20:
				print('> [%s]'%test_pair[0])
				print('= [%s]'%test_pair[1])
				print('< [%s]'%output_sentence)
				print('BLEU: %.4f'%bleu_score)
				print('Perplexity: %.4f'%perplexity_score)
			i += 1
			# Calculate BLEU score
			bleu_total += bleu_score
			perplexity_total += perplexity_score
			
		avg_bleu = bleu_total / (len(self.test_pairs) - empty_sentences)
		avg_perplexity = perplexity_total / (len(self.test_pairs) - empty_sentences)
		print('Predicted a total of %d empty sentences.'%empty_sentences)
		print('Average BLEU score = ' + str(avg_bleu))
		print('Average Perplexity score = ' + str(avg_perplexity))
		return avg_bleu, avg_perplexity
