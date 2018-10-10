import collections
import math

# Sentence start and stop tokens
START = '<START>'
STOP = '<STOP>'

class PerplexityModel:
	def __init__(self, sentences, default_prob=0.01):
		self.default_prob = default_prob
		# Set the default value for START
		# STOP defaults to 0 since no word should follow the STOP token
		self.model = {START: collections.defaultdict(lambda: self.default_prob), 
			STOP: collections.defaultdict(lambda: 0.0)}
			
		# Probability of each word given the previous word
		# model[w_i][w_j] = P(w_j|w_i)
		for sentence in sentences:
			words = sentence.split()
			for i, word in enumerate(words):
				if word not in self.model:
					# Set the default value for the current word
					self.model[word] = collections.defaultdict(lambda: self.default_prob)
				if i == len(words) - 1:
					if STOP in self.model[word]:
						self.model[word][STOP] += 1
					else:
						self.model[word][STOP] = 1.0
				else:
					if i == 0:
						if word in self.model[START]:
							self.model[START][word] += 1
						else:
							self.model[START][word] = 1.0
					if words[i+1] in self.model[word]:
						self.model[word][words[i+1]] += 1
					else:
						self.model[word][words[i+1]] = 1.0
		# Normalize the probabilities
		for word in self.model:
			total = 0
			keys = list(self.model[word].keys())
			# Get the total for each word
			for key in keys:
				total += self.model[word][key]
			# Normalize the values
			for key in keys:
				self.model[word][key] /= total

	# Quick method to nicely print out the probability model
	def print_model(self):
		for word in self.model:
			print('%s:'%word)
			for key in self.model[word]:
				print('\t%s: %.4f'%(key, self.model[word][key]))

	# Calculates the perplexity of a given sentence
	# The sentence should be a string where each word is separated by white space (including punctuations)
	def perplexity(self, sentence, b=2):
		sentence = sentence.split()

		# Calculate the sum of all probabilties
		# A probability is P(w_j | w_i) where w_i is the word immeidately preceding w_j
		# The first word in a sentence is preceded by the START token
		# The last word in a sentence is followed by the STOP token
		summation = 0.0
		for i, word in enumerate(sentence):
			try:
				if i == 0:
					summation += math.log(self.model[START][word], b)
				elif i == len(sentence) - 1:
					summation += math.log(self.model[word][STOP], b)
				else:
					summation += math.log(self.model[sentence[i-1]][word], b)
			except KeyError:
				# This word is not in the corpus so use the default probability
				summation += self.default_prob
				
		return pow(b, (-1/float(len(sentence)))*summation)
