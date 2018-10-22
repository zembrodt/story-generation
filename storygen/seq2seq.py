# seq2seq.py
import time
import random
import time
import math
import os
import operator
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu

import storygen.book as bk
import pymeteor.pymeteor.pymeteor as pymeteor

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
    indexes.append(bk.STOP_ID)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Converts a pair of sentences into a pair of pytorch tensors
def tensorsFromPair(pair, input_book, output_book, device):
    input_tensor = tensorFromSentence(input_book, pair[0], device)
    target_tensor = tensorFromSentence(output_book, pair[1], device)
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

        # log_softmax uses log base e
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
                return 'BeamSearchResult: score=%.4f, stopped=%s, words="%s"'%(self.score, str(self.stopped), ' '.join(self.words))

# Represents a Sequence to Sequence network made up of an encoder RNN and decoder RNN with attention weights
class Seq2Seq:
	def __init__(self, input_book, output_book, max_length, hidden_size, device):
		self.encoder = None
		self.decoder = None
		self.input_book = input_book
		self.output_book = output_book
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.device = device
		self.encoder_optimizer = None
		self.decoder_optimizer = None
		self.criterion = None

		#for testing
		self.i = 0

	def loadFromFiles(self, encoder_filename, decoder_filename):
		# Check that the path for both files exists
		os.makedirs(os.path.dirname(encoder_filename), exist_ok=True)
		os.makedirs(os.path.dirname(decoder_filename), exist_ok=True)
		
		encoder_file = Path(encoder_filename)
		decoder_file = Path(decoder_filename)

		if encoder_file.is_file() and decoder_file.is_file():
			print("Loading encoder and decoder from files...")
			self.encoder = EncoderRNN(self.input_book.n_words, self.hidden_size).to(self.device)
			self.encoder.load_state_dict(torch.load(encoder_file, map_location=self.device))
			
			self.decoder = DecoderRNN(self.hidden_size, self.output_book.n_words, self.max_length).to(self.device)
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
	# Then we call "train" many times and occasionally print the progress (%
	# of examples, time so far, estimated time) and average loss.
	def trainIters(self, train_pairs, epochs, print_every=1000, learning_rate=0.01):
                # If we didn't load the encoder/decoder from files, create new ones to train
		if self.encoder is None or self.decoder is None:
			self.encoder = EncoderRNN(self.input_book.n_words, self.hidden_size).to(self.device)
			self.decoder = DecoderRNN(self.hidden_size, self.output_book.n_words, self.max_length).to(self.device)

		start = time.time()
		print_loss_total = 0  # Reset every print_every

		self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
		self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
		
		training_pairs = [tensorsFromPair(train_pair, self.input_book, self.output_book, self.device) for train_pair in train_pairs]
		random.shuffle(training_pairs)
		
		self.criterion = nn.NLLLoss()
		
		# Iterate through the training set over a set amount of epochs
		# Output the progress and current loss value
		for i in range(epochs):
			for j, pair in enumerate(training_pairs):
				input_tensor = pair[0]
				target_tensor = pair[1]
				loss = self._train(input_tensor, target_tensor)
				print_loss_total += loss

				if j % print_every == 0:
					print_loss_avg = print_loss_total / print_every
					print_loss_total = 0
					progress_percent = (i*len(training_pairs)+j)/(epochs*len(training_pairs))
					t = -1.0
					if progress_percent > 0:
					    t = timeSince(start, progress_percent)
					print('%s (%d %.2f%%) %.4f' % (t, (i*len(training_pairs)+j), progress_percent * 100, print_loss_avg))
		
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
			input_tensor = tensorFromSentence(self.input_book, sentence, self.device)
				
			input_length = input_tensor.size()[0]
			encoder_hidden = self.encoder.initHidden(self.device)

			encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

			for ei in range(input_length):
				encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] += encoder_output[0, 0]

			decoder_input = torch.tensor([[bk.START_ID]], device=self.device)  # SOL

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
			while True:
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
							if topi.squeeze()[i].item() == bk.STOP_ID:
								new_result.stopped = True
							else:
								new_result.words.append(self.output_book.index2word[topi.squeeze()[i].item()])
							new_results.append(new_result)
					else: # make sure to re-add currently stopped sentences
						new_results.append(result)
				results = sorted(new_results, key=operator.attrgetter('score'))[::-1][:k]
				if not still_searching:
					break
			###
			
			return results

	def _evaluate_specified(self, sentence, sentence_to_evaluate):
		with torch.no_grad():
			input_tensor = tensorFromSentence(self.input_book, sentence, self.device)
				
			input_length = input_tensor.size()[0]
			encoder_hidden = self.encoder.initHidden(self.device)

			encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

			for ei in range(input_length):
				encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] += encoder_output[0, 0]

			decoder_input = torch.tensor([[bk.START_ID]], device=self.device)  # SOL

			decoder_hidden = encoder_hidden

			decoded_words = []
			decoder_attentions = torch.zeros(self.max_length, self.max_length)

			summation = 0.0
			N = 0

			evaluate_tensor = tensorFromSentence(self.input_book, sentence_to_evaluate, self.device)

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
				
				if evaluate_item == bk.STOP_ID:
					break
				else:
					# Decode the predicted word from the book 
					decoded_words.append(self.output_book.index2word[topi.item()])

				decoder_input = tensorFromIndex(evaluate_item, self.device)

			perplexity = pow(math.e, -summation / N)# / N because the evaluate sentence is converted to a tensor where the last item will be bk.STOP_ID

			# note: decoder_attentions not properly set up in this function

			return decoded_words, decoder_attentions, perplexity

	def evaluate(self, sentence):
		with torch.no_grad():
			input_tensor = tensorFromSentence(self.input_book, sentence, self.device)
				
			input_length = input_tensor.size()[0]
			encoder_hidden = self.encoder.initHidden(self.device)

			encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

			for ei in range(input_length):
				encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] += encoder_output[0, 0]

			decoder_input = torch.tensor([[bk.START_ID]], device=self.device)  # SOL

			decoder_hidden = encoder_hidden

			decoded_words = []
			decoder_attentions = torch.zeros(self.max_length, self.max_length)

			summation = 0.0
			N = 0

			for di in range(self.max_length):
				N += 1
				decoder_output, decoder_hidden, decoder_attention = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs)
				decoder_attentions[di] = decoder_attention.data
				
				# output.data contains all log probabilities and distribution(?)
				# force the program to generate a sentence and record the probability of doing so
				topv, topi = decoder_output.data.topk(1)

				## Perplexity code ##
				summation += topv.item()
				
				if topi.item() == bk.STOP_ID:
					#decoded_words.append('<EOL>')
					break
				else:
					# Decode the predicted word from the book 
					decoded_words.append(self.output_book.index2word[topi.item()])

				decoder_input = topi.squeeze().detach()

			## More code for perplexity ##
			perplexity = math.pow(math.e, -summation / N)
			##

			return decoded_words, decoder_attentions[:di + 1], perplexity
	
	######################################################################
	# Evaluates each pair given in the test set
	# Returns BLEU, METEOR, and perplexity scores
	def evaluateTestSet(self, test_pairs):
		i = 0
		CAP = 15
		print('Printing first %d evaluations:'%CAP)
		bleu_total = 0.0
		meteor_total = 0.0
		perplexity_total = 0.0

		beam_bleu_total = 0.0
		beam_meteor_total = 0.0
		beam_perplexity_total = 0.0
		
		# may no longer need empty_sentences...
		empty_sentences = 0
		beam_empty_sentences = 0
		for test_pair in test_pairs:
                        # Predict using evaluate
			output_words, attentions, perplexity_score = self.evaluate(test_pair[0])
			if len(output_words) == 0:
			    print('We outputted an empty sentence! Skipping...')
			    print('Test pair:\n\t>%s\n\t=%s'%(test_pair[0], test_pair[1]))
			    empty_sentences += 1
			    beam_empty_sentences += 1
			    continue
			output_sentence = ' '.join(output_words)
			# Calculate BLEU and METEOR for evaluate
			bleu_score = calculateBleu(output_sentence, test_pair[1])
			meteor_score = pymeteor.meteor(output_sentence, test_pair[1])

			#Predict using beam search
			k = 5
			beam_result = self.beam_search(test_pair[0], k)[0] # Get the best result
			if len(beam_result.words) == 0:
			    print('We outputted an empty beam sentence! Skipping...')
			    print('Test pair:\n\t>%s\n\t=%s'%(test_pair[0], test_pair[1]))
			    beam_empty_sentences += 1
			    continue
			beam_sentence = ' '.join(beam_result.words)
			# Calculate BLEU and METEOR for beam search
			beam_bleu_score = calculateBleu(beam_sentence, test_pair[1])
			beam_meteor_score = pymeteor.meteor(beam_sentence, test_pair[1])
			beam_perplexity_score = beam_result.calculate_perplexity()
			
			if i < CAP:
				print('> [%s]'%test_pair[0])
				print('= [%s]'%test_pair[1])
				print('evaluate:')
				print('< [%s]'%output_sentence)
				print('\tBLEU:       %.4f'%bleu_score)
				print('\tMETEOR:     %.4f'%meteor_score)
				print('\tPerplexity: %.4f'%perplexity_score)
				print('Beam search (k=%d):'%k)
				print('< [%s]'%beam_sentence)
				print('\tBLEU:       %.4f'%beam_bleu_score)
				print('\tMETEOR:     %.4f'%beam_meteor_score)
				print('\tPerplexity: %.4f'%beam_perplexity_score)
				i += 1
				
			bleu_total += bleu_score
			meteor_total += meteor_score
			perplexity_total += perplexity_score

			beam_bleu_total += beam_bleu_score
			beam_meteor_total += beam_meteor_score
			beam_perplexity_total += beam_perplexity_score
			
		avg_bleu = bleu_total / (len(test_pairs) - empty_sentences)
		avg_meteor = meteor_total / (len(test_pairs) - empty_sentences)
		avg_perplexity = perplexity_total / (len(test_pairs) - empty_sentences)
		
		beam_avg_bleu = beam_bleu_total / (len(test_pairs) - beam_empty_sentences)
		beam_avg_meteor = beam_meteor_total / (len(test_pairs) - beam_empty_sentences)
		beam_avg_perplexity = beam_perplexity_total / (len(test_pairs) - beam_empty_sentences)

		print('evaluate:')
		print('\tPredicted a total of %d empty sentences.'%empty_sentences)
		print('\tAverage BLEU score = ' + str(avg_bleu))
		print('\tAverage METEOR score = ' + str(avg_meteor))
		print('\tAverage Perplexity score = ' + str(avg_perplexity))
		
		print('Beam search (k=%d):'%k)
		print('\tPredicted a total of %d empty sentences.'%beam_empty_sentences)
		print('\tAverage BLEU score = ' + str(beam_avg_bleu))
		print('\tAverage METEOR score = ' + str(beam_avg_meteor))
		print('\tAverage Perplexity score = ' + str(beam_avg_perplexity))
		
		return avg_bleu, avg_meteor, avg_perplexity, beam_avg_bleu, beam_avg_meteor, beam_avg_perplexity
