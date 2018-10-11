# book.py
import unicodedata
import string
import re

START_ID = 0
START_TOKEN = '<START>'
STOP_ID = 1
STOP_TOKEN = '<STOP>'

## Classes ##
class Book:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        # Start Of Line, End Of Line
        self.index2word = {START_ID: START_TOKEN, STOP_ID: STOP_TOKEN}
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

class ContractionDict:
	def __init__(self, contractions):
		self.contractions = contractions
	def contains(self, word):
		return word in self.contractions and len(self.contractions[word]) > 0
	def get(self, word):
		return self.contractions[word]
	def convert(self, word):
		if self.contains(word):
			return self.get(word)
		else:
			return word

## Functions ##
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

# Converts an array of strings representing lines in a file to an array of strings
# where each string is a roughly a setnence. Removes punctuation.
def convertLinesToSentences(lines, contraction_dict):
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
                    convertedWords = contraction_dict.convert(word).split()
                    sentenceSplit[j] = convertedWords[0]
                    for k, convertedWord in enumerate(convertedWords[1:]):
                        sentenceSplit.insert(j+k+1, convertedWord)
            sentences[i] = ' '.join(sentenceSplit)

    # Final catch to remove any empty sentences
    return [sentence for sentence in sentences if len(sentence) > 0]
