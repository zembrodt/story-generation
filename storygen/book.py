# book.py
import unicodedata
from unidecode import unidecode
import string
import os
import re
import sys
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk import word_tokenize

# Ids and tokens to define the start and end of sentences
START_ID = 0
START_TOKEN = '<START>'
STOP_ID = 1
STOP_TOKEN = '<STOP>'

# Pre-defined contractions to keep
CONTRACTIONS = [
    'n\'t', '\'ll', '\'s', '\'re', '\'m', '\'d', '\'ve'
]

CONTRACTIONS_FILE = 'data/contractions_dictionary.txt'

# Harry Potter books have a max of 38 chapters and an epilogue
CHAPTER_NUMBERS = [
    'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
    'thirty', 'epilouge'
]

## Classes ##
# Defines the word-to-index of the vocabulary in a book
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

# Wrapper for accessing a dictionary of contractions
class ContractionDict:
	def __init__(self, contractions):
		self.contractions = contractions
	def contains(self, word):
		return word in self.contractions and len(self.contractions[word]) > 0
	def get(self, word):
		return self.contractions[word]
	def convert(self, word):
		if self.contains(word):
			return self.get(word), True
		else:
			return word, False

## Functions ##
# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
	c for c in unicodedata.normalize('NFD', s)
	if unicodedata.category(c) != 'Mn'
    )

# Removes unneeded punctuation in the data
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # Check if hyphens are against single/double quotes (we need to keep the quotes and not add a space)
    if re.search(r'((\'|\")\-+|\-+(\'|\"))', s):
        s = re.sub(r'(\'\-+\s*|\s*\-+\')', '\'', s)
        s = re.sub(r'(\"\-+\s*|\s*\-+\")', '\"', s)
    s = re.sub(r'[\(\)\*\_\\\/\-,:]', ' ', s) # Punctuations to remove, replace with space
    # Special cases to parse
    s = re.sub(r'j\.k\.', 'jk', s)
    s = re.sub(r'b\.c\.', 'bc', s)
    s = re.sub(r'a\.d\.', 'ad', s)
    return s

# Handle multiple periods by determining if they end a sentence or not
# If they end a sentence, replace with a single period
# If they do not, replace with whitespace
def replace_multiple_periods(lines):
    # Helper function
    def replace_periods(word, replace_text=''):
        return re.sub(r'[\.]{2,}', replace_text, word)

    for i, line in enumerate(lines):
        words = line.strip().split()
        for j, word in enumerate(words):
            # Separate sentences if multiple periods exist based on the following conditions:
            # First char after periods is capital, and not "I" or "I'*".
            if re.search(r'[\.]{2,}', word): # Checks [word]..[.*]
                # Check if this is multiple periods separating two words, no spaces
                if re.search(r'[\.]{2,}\w', word):
                    word_split = re.sub(r'[\.]{2,}', ' ', word).split()
                    word_builder = word_split[0]
                    # Combine the words together, deciding if there should be a period or space
                    for k in range(len(word_split)):
                        if k+1 < len(word_split):
                            # If the word following the periods is capital, and is not "I", "I'*", etc, add a period
                            if re.search(r'^[A-Z]', word_split[k+1]) and not re.search(r'^(I\'.+|I[.!?]*$)', word_split[k+1]):
                                word_builder += '. ' + word_split[k+1]
                            else:
                                word_builder += ' ' + word_split[k+1]
                    # Replace our current word
                    word = word_builder
                # Check the next word
                elif j+1 < len(words):
                    # First char is capital, and not "I", "I'*", etc
                    if re.search(r"^[A-Z]", words[j+1]) and not re.search(r"^(I'.+|I[.!?]*$)", words[j+1]):
                        # Replace "..+" with "."
                        word = replace_periods(word, '.')
                    else:
                        # Replace "..+" with " "
                        word = replace_periods(word)
                else:
                    # Check the next line
                    if i+1 < len(lines):
                        next_words = lines[i+1].strip().split()
                        if len(next_words) > 0:
                            # First char is capital, or begins with dialogue (double quotes)
                            if re.search(r'^[A-Z\"]', next_words[0]):
                                # Replace "..+" with "."
                                word = replace_periods(word, '.')
                            else:
                                # Next sentence begins with a lower case letter, let's assume the sentence continues
                                word = replace_periods(word)
                        else:
                            # Empty line next, assume the sentence ended
                            word = replace_periods(word, '.')
                    else:
                        # EOL, and EOF, replace "..+" with " "
                        word = replace_periods(word)
            elif re.search(r'^(\'|\")?[\.]{2,}\w', word):
                word = replace_periods(word)
            words[j] = word
        lines[i] = ' '.join(words)
    return lines

# Finds if a chapter number exists in the text and removes it
def remove_chapter_number(text):
    for chapter_num in CHAPTER_NUMBERS:
        chapter_num_regex = '^' + ''.join([c+'+' for c in chapter_num])
        if re.search(chapter_num_regex, text):
            # We found the chapter number!
            text = re.sub(chapter_num_regex, '', text)
            return text
    return text

def separate_terminator(text):
    if re.search(r'[\.!?;]$', text):
        if text[-1] == '!' or text[-1] == '?':
            return [text[:-1], text[-1]]
        else:
            return [text[:-1]]
    else:
        return [text]

def remove_stuttering(lines):
    for i, line in enumerate(lines):
        # Tokenize the line
        words = line.strip().split()
        word_updated = False
        for j, word in enumerate(words):
            # Match for stuttering word, with optional dialogue opening (ex: ["b-b-but] -> ["but])
            if re.match(r'^(\'|\")?(\w\-)+\w+', word):
                # Get the word root as everything following the last hyphen
                word_root = word.split('-')[-1]
                # Check if the word stutter was at the start of dialogue/inner-dialogue
                if word[0] in ['\'','"']:
                    word_root = '{}{}'.format(word[0], word_root)
                words[j] = word_root
                word_updated = True
        # Assign the updated line
        if word_updated:
            lines[i] = ' '.join(words)
    return lines

def remove_chapter_titles(lines):
    # Create a new list as we may need to remove some lines
    parsed_lines = []
    found_chapter = False
    for line in lines:
        # Check if we previously found a chapter number without a title
        if found_chapter:
            if len(line) > 0:
                found_chapter = False
            # Continue until we find the chapter title or skip it
            continue
        # Check if page number on line (format: '#' or '*#*')
        if re.match(r'^\**\d+\**$', line):
            line = ''
        # Check if chapter title (remove hyphens)
        line_nospace = ''.join(re.split(r'\s+|-+', line)).lower()
        chapter_regex = r'^c+h+a+p+t+e+r+'
        if re.search(chapter_regex, line_nospace):
            # We found a chapter line!
            chapter_line = re.sub(chapter_regex, '', line_nospace)
            # Check for number
            chapter_line = remove_chapter_number(chapter_line)
            if len(chapter_line) == 0:
                # The chapter title should be on the next non-whitespace line
                found_chapter = True
            else:
                chapter_line = remove_chapter_number(chapter_line)
                if len(chapter_line) == 0:
                    # The remainder of the chapter line was another number (i.e. twenty one)
                    found_chapter = True # We still need to find the chapter title
            line = ''
        parsed_lines.append(line)
    return parsed_lines

def parse_dialogue(lines):
    sentences = []
    currSentence = []
    inDialogue = False
    inInnerDialogue = False
    for i, line in enumerate(lines):
        # Correction for inDialogue bool
        if inDialogue and len(line) == 0:
            inDialogue = False
        if '"' in line:
            line_words = line.split()
            for j, word in enumerate(line_words):
                # Search for dialogue to find double quotes
                if '"' in word:
                    # Special case where double quote is separated as its own token
                    if word == '"':
                        if inDialogue:
                            if len(currSentence) > 0:
                                sentences.append(' '.join(currSentence))
                                currSentence = []
                            inDialogue = False
                        else:
                            # We are not in dialogue, will need to do a few checks
                            # Check if the previous word contained a double quote
                            prev_line_words = []
                            if i > 0:
                                prev_line_words = lines[i-1].split()
                            if (j > 0 and '"' in line_words[j-1]) or \
                                    (j == 0 and len(prev_line_words) > 0 and '"' in prev_line_words[-1]):
                                # Current sentence should be empty, but if not, let's empty it
                                if len(currSentence) > 0:
                                    sentences.append(' '.join(currSentence))
                                    currSentence = []
                                inDialogue = True # Set dialogue for next sentence
                            #elif (j+1 < len(words) and '"' in words[j+1]) or (i+1 < len(words) and '"' in line[i+1].split()[0]):
                            else:
                                # Previous word did not have double quotes, next does.
                                # End current sentence, inDialogue already set to False
                                if len(currSentence) > 0:
                                    sentences.append(' '.join(currSentence))
                                    currSentence = []
                    elif re.match(r'^\".+\"$', word):
                        if not inDialogue:
                            # Remove double quotes from word
                            word = re.sub('"', '', word)
                            # End current sentence if it exists
                            if len(currSentence) > 0:
                                sentences.append(' '.join(currSentence))
                                currSentence = []
                            # Add the single word (with possible terminator split)
                            sentences.append(' '.join(separate_terminator(word)))
                        else:
                            print('Possible incorrect dialogue construction around line [{}]'.format(line))
                            inDialogue = False # Correct dialogue errors?
                    elif re.match(r'^\".+$', word):
                        # Starting dialogue
                        inDialogue = True
                        # An existing sentence was not terminated before entering dialogue
                        if len(currSentence) > 0:
                            sentences.append(' '.join(currSentence)) # Add the sentence
                        word = re.sub('"', '', word)
                        currSentence = [word] # Start a new sentence with the dialogue
                    elif re.match(r'^.+\"$', word):
                        # End of dialogue
                        inDialogue = False
                        # Check if the last word ended in one of our sentence terminators
                        word = re.sub('"', '', word)
                        words = separate_terminator(word)
                        currSentence += words
                        sentences.append(' '.join(currSentence)) # end the current sentence
                        currSentence = []
                    else:
                        print('Double quote inconsistency around line: [{}]'.format(line))
                        # Remove double quote and add word
                        word = re.sub('"', '', word)
                        currSentence.append(word)
                elif '\'' in word:
                    if re.match(r'^\'.+\'$', word):
                        word = re.sub('\'', '', word)
                        # End current sentence if it exists
                        if len(currSentence) > 0:
                            sentences.append(' '.join(currSentence))
                            currSentence = []
                        # Add the single word (with possible terminator split)
                        sentences.append(' '.join(separate_terminator(word)))
                    elif re.match(r'^\'.+$', word):
                        inInnerDialogue = True
                        # An existing sentence was not terminated before entering dialogue
                        if len(currSentence) > 0:
                            sentences.append(' '.join(currSentence)) # Add the sentence
                        word = re.sub('\'', '', word)
                        currSentence = [word] # Start a new sentence with the dialogue
                    elif re.match(r'^.+\'$', word):
                        # Naive check that this is the end of inner dialogue and not a plural possessive word
                        if inInnerDialogue or word[-1] != 's':
                            # End the inner dialogue
                            inInnerDialogue = False
                            word = re.sub('\'', '', word)
                            # Check if the last word ended in one of our sentence terminators
                            words = separate_terminator(word)
                            currSentence += words
                            sentences.append(' '.join(currSentence)) # end the current sentence
                            currSentence = []
                        else:
                            # Potential plural possessive, add it and continue the current sentence
                            word = re.sub('\'', '', word)
                            currSentence.append(word)
                    else:
                        print('Single quote inconsistency around line: [{}]'.format(line))
                        # Remove single quote and add word
                        word = re.sub('\'', '', word)
                        currSentence.append(word)
                else:
                    currSentence.append(word)
        else:
            sentences.append(line)
        
    return sentences

# Removes all punctuation from the given lines
def remove_punctuation(lines, replace_str=''):
    for i, line in enumerate(lines):
        # Single quotes are not handled by this as they are a special case and already removed
        lines[i] = re.sub(r'[\"\(\)\-\_\+\=\&\^\$\%\#\@\~\`\.\;\:\\\/\<\>]', replace_str, line)
    return lines

# Converts an array of strings representing lines in a file to an array of strings
# where each string is a roughly a setnence. Removes punctuation.
def convertLinesToSentences(lines, contraction_dict, debug_results=False):
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except Exception:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Pre-processing    
    # Remove page numbers and chapter titles
    lines = remove_chapter_titles(lines)

    # Convert unicode characters to their nearest ASCII equivalent
    lines = [unidecode(line) for line in lines]

    # Use nltk to parse into sentences
    sentences = []
    for sentence in tokenizer.tokenize('\n'.join(lines)):
        # Replace newlines with spaces
        sentences.append(' '.join(sentence.split('\n')))
    
    # Parse multiple periods to determine if they end a sentence
    sentences = replace_multiple_periods(sentences)

    # Replace double single quotes with double quotes
    sentences = [re.sub(r'\'\'', '"', sentence) for sentence in sentences]

    # Special case: remove "stuttering" dialogue (ex: b-b-but)
    sentences = remove_stuttering(sentences)
        
    # Lower and normalize the text
    sentences = [normalizeString(line) for line in sentences]

    # Remove all single quotes via the contraction dictionary or predefined contraction pairs
    for i, sentence in enumerate(sentences):
        if '\'' in sentence:
            words = sentence.split()
            for j, word in enumerate(words):
                if '\'' in word:
                    # Remove any punctuation from word to expand:
                    word = re.sub(r'[\.\!\?\;\:\"]', '', word)
                    # Expand words within our contraction dictionary
                    convertedWords, found_contraction = contraction_dict.convert(word)
                    # Check if we expanded the words
                    if found_contraction:
                        convertedWords = convertedWords.split()
                        #words[j] = convertedWords[0]
                        words[j] = ' '.join(convertedWords)
                        # Insert the additional words (if they exist) into the sentence
                        # This may cause incosistencies with 'j' and 'word' as we're now enumerating over a modified list
                        #for k, convertedWord in enumerate(convertedWords[1:]):
                        #    words.insert(j+k+1, convertedWord)
            sentences[i] = ' '.join(words)
    
    # Tokenize the sentences
    for i, sentence in enumerate(sentences):
        sentence = ' '.join(word_tokenize(sentence))
        sentence = re.sub(r'(\'\'|\`\`)', '"', sentence)
        sentence = re.sub(r'\s\'\s', ' " ', sentence)
        # Append newlines on terminators to split on later
        sentence = re.sub(r'\s\!\s', ' !\n ', sentence)
        sentence = re.sub(r'\s\?\s', ' ?\n ', sentence)
        sentence = re.sub(r'\s[\.\;]\s', ' \n ', sentence)
        sentences[i] = sentence
    
    # Repleace known contractions with asterisks for placeholders
    for i, sentence in enumerate(sentences):
        if '\'' in sentence:
            words = sentences[i].split()
            for j, word in enumerate(words):
                if '\'' in word:
                    # Remove any punctuation from word to expand:
                    word = re.sub(r'[\.\!\?\;\:\"]', '', word)
                    # Check if we can expand out any contractions in the words
                    if word in CONTRACTIONS:
                        words[j] = re.sub(r'\'', '\*', word)
                    else:
                        words[j] = re.sub(r'\'', '', word)
            sentences[i] = ' '.join(words)

    # Separate dialogue into its own sentences
    sentences = parse_dialogue(sentences)

    # Remove any leftover single quotes
    sentences = [re.sub(r'\'', '', sentence) for sentence in sentences]

    # Need one final pass to split on newlines gathered in terminator section
    final_sentences = []
    for sentence in sentences:
        final_sentences += [s.strip() for s in sentence.split('\n') if len(s) > 0]
    
    # Remove placeholder asterisks with single quotes
    final_sentences = [re.sub(r'\*', '\'', sentence) for sentence in final_sentences]

    # Final pass to remove all punctuation
    if not debug_results:
        return remove_punctuation(final_sentences)
    else:
        # Have the option to return punctuations to further debug parsing on specific texts
        return final_sentences

# Called when book.py is executed directly
def main():
    # Get directory from command line
    if len(sys.argv) < 2:
        print('book.py requires at least 1 argument! (directory)')
        exit()

    # Check for special case parameters
    # TODO: convert this to use getopt
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print('book.py\nUsage:')
        print('\t--build <directory>: Combines all text files within the specified directory into a single combined file')
        print('\t--filter <text file>: filters all stopwords out of the specified text file')
        print('\t--contractions <directory>: builds/updates a contractions dictionary based on the corpora in the specified directory')
        exit(0)
    if len(sys.argv) > 2:
        if sys.argv[1] == '--build':
            arg = sys.argv[2]
            if os.path.isdir(arg):
                output_dir = arg if arg[-1] == '/' or arg[-1] == '\\' else arg + '/'
                # Build all files within the specified directory into a single file
                with open('{}combined.txt'.format(output_dir), 'w+') as output_f:
                    for book_file in os.listdir(arg):
                        book_file_path = '{}{}'.format(arg if arg[-1] == '/' or arg[-1] == '\\' else arg+'/', book_file)
                        with open(book_file_path, encoding='utf-8', errors='ignore') as input_f:
                            sentences = input_f.read().split('\n')
                            for sentence in sentences:
                                output_f.write('{}\n'.format(sentence))
                print('combined.txt built!')
            else:
                print('{} is not a directory!'.format(arg))
            exit()
        elif sys.argv[1] == '--filter':
            stop_words = set(stopwords.words('english'))
            arg = sys.argv[2]
            if os.path.isfile(arg):
                arg_split = arg.split('.')
                filter_file = '{}_filtered.{}'.format('.'.join(arg_split[:-1]), arg_split[-1])
                with open(arg) as f:
                    sentences = f.readlines()
                filtered_sentences = []
                for sentence in sentences:
                    filtered_sentences.append(' '.join([word for word in sentence.split() if word not in stop_words]))
                with open(filter_file, 'w+') as f:
                    for i, filtered_sentence in enumerate(filtered_sentences):
                        f.write(filtered_sentence)
                        if i+1 < len(filtered_sentences):
                            f.write('\n')
                print('{} filtered.\nSaved at {}'.format(arg, filter_file))
            else:
                print('{} is not a file!'.format(arg))
            exit()
        elif sys.argv[1] == '--contractions':
            arg = sys.argv[2]
            if os.path.isdir(arg):
                # Create the contractions dictionary file if it does not exist
                if not os.path.isfile(CONTRACTIONS_FILE):
                    with open(CONTRACTIONS_FILE, 'w+') as f:
                        f.write('{}')
                # Build/update contractions dictionary template based on files in given directory
                with open(CONTRACTIONS_FILE, 'r') as f:
                    s = f.read()
                    contractions = eval(s)
                for book_file in os.listdir(arg):
                    # Get lines from file
                    book_file_path = '{}{}'.format(arg if arg[-1] == '/' or arg[-1] == '\\' else arg+'/', book_file)
                    # Open the given text files (ignore bytes not in utf-8)
                    with open(book_file_path, encoding='utf-8', errors='ignore') as f:
                        lines = f.read().strip().split('\n')
                    # Add any words that aren't already in the dictionary
                    for line in lines:
                        line = unidecode(line) # Convert unicode characters to ascii
                        line = re.sub(r'[\,\!\.\?\-\(\)\;\:\=\\\/\"]', ' ', line.lower())
                        for word in line.split():
                            # If the word has a single quote
                            if '\'' in word:
                                # Remove any punctuation from the word
                                # if the word is not in the dictionary and not possessive (*'s/*s')
                                if word not in contractions and not re.search(r'(\'s$|s\'$)', word):
                                    contractions[word.lower()] = ""
                # Save the dictionary with json
                with open(CONTRACTIONS_FILE, 'w') as f:
                    f.write('{\n')
                    keys = sorted(contractions)
                    for key in keys:
                        f.write('"{}": "{}",\n'.format(key, contractions[key]))
                    f.write('}')
                exit()
            else:
                print('{} is not a directory!'.format(arg))
    # TODO: possible parsing: '' -> "
    # Create contractions dictionary
    with open(CONTRACTIONS_FILE, 'r') as f:
        s = f.read()
        contractions = eval(s)
    contraction_dict = ContractionDict(contractions)
    
    # Directory for input files
    arg = sys.argv[1]

    if os.path.isdir(arg):
        # Variables to keep track of statistics
        max_lengths = {}
        num_sentences = {}
        # Directory to store parsed files
        parsed_dir = '{}_parsed/'.format(arg[:-1] if arg[-1] == '/' or arg[-1] == '\\' else arg)
        os.makedirs(parsed_dir, exist_ok=True)
        for book_file in os.listdir(arg):
            book_file_split = book_file.split('.')
            parsed_filename = '.'.join(book_file_split[:-1]) + '_parsed.' + book_file_split[-1]
            # Get lines from file
            book_file_path = '{}{}'.format(arg if arg[-1] == '/' or arg[-1] == '\\' else arg+'/', book_file)
            #try:
            # Open the given text files (ignore bytes not in utf-8)
            with open(book_file_path, encoding='utf-8', errors='ignore') as f:
                lines = f.read().strip().split('\n')
            sentences = convertLinesToSentences(lines, contraction_dict)
            max_lengths[book_file] = max(map(len, [sentence.split() for sentence in sentences]))
            num_sentences[book_file] = len(sentences)
            # Saved parsed file
            with open('{}{}'.format(parsed_dir, parsed_filename), 'w+') as f:
                for i, sentence in enumerate(sentences):
                    f.write(sentence)
                    if i+1 < len(sentences):
                        f.write('\n')
                print('Saved {}.'.format(parsed_filename))
            #except Exception as e:
            #    print('Failed converting \'{}\'. Cause:\n{}'.format(book_file_path, e))
        print ('Max sentence lengths:')
        for book in max_lengths:
            print('\t{}: {}'.format(book, max_lengths[book]))
        print('Number of sentences:')
        for book in num_sentences:
            print('\t{}: {}'.format(book, num_sentences[book]))
    else:
        print('{} is not a directory!'.format(arg))

if __name__ == '__main__':
    main()