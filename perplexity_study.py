# perplexity_study.py
import getopt
import math
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
import re
import sys

import story_generation as storygen
import storygen.seq2seq as seq2seq
import storygen.log as log

OBJ_DIR = 'obj'
EMBEDDING_TYPES = ['glove', 'sg', 'cbow']
HELP_MSG = ''.join([
    'Usage:\n',
    'python3 perplexity_study.py [-h, --help] [--embedding]\n',
    '\t-h, --help: Provides help on command line parameters\n',
	'\t--embedding <embedding_type>: specify an embedding to use from: {}'.format(EMBEDDING_TYPES),
    '\t--build <filename>: builds a graphical representation of several perplexity studies',
])


def calculate_perplexities(network, words, pairs):
     # Actual book sentences
    actual_sentences_score = 0.0

    # Random words
    random_words_score = 0.0
    
    # Random sentences
    random_sentences_score = 0.0
    
    # Calculate scores
    i = 0
    percentages = {0}
    curr_len = len(pairs)
    for pair in pairs:
        # Calculate the actual perplexity
        actual_perplexity = network._evaluate_specified(pair[0], pair[1])

        # Calculate the perplexity of sentences (of the same length) build of random words in the vocab 
        random_words = ' '.join([random.choice(words) for i in range(len(pair[1].split()))])
        words_perplexity = network._evaluate_specified(pair[0], random_words)
        
        # Calculate the perplexity of sentences taken at random from the pairs
        random_sentence = random.choice(random.choice(pairs))
        sentence_perplexity = network._evaluate_specified(pair[0], random_sentence)

        if actual_perplexity is not None and words_perplexity is not None and sentence_perplexity is not None:
            actual_sentences_score += actual_perplexity
            random_words_score += words_perplexity
            random_sentences_score += sentence_perplexity
        else:
            curr_len -= 1
            if actual_perplexity is None:
                print('(i={}) Actual: retrieved a 0-d tensor from ([{}], [{}])'.format(i, pair[0], pair[1]))
            if words_perplexity is None:
                print('(i={}) Random words: retrieved a 0-d tensor from ([{}], [{}])'.format(i, pair[0], random_words))
            if sentence_perplexity is None:
                print('(i={}) Random sentences: retrieved a 0-d tensor from ([{}], [{}])'.format(i, pair[0], random_sentence))

        i += 1
        percentage = math.floor(100 * (float(i) / len(pairs)))
        if percentage % 5 == 0 and percentage not in percentages:
            print('{}% complete.'.format(percentage))
            percentages.add(percentage)

    return actual_sentences_score / curr_len, random_words_score / curr_len, random_sentences_score / curr_len

# Study on the perplexity module using the provided text
def main(argv):
    # Get command line arguments
    try:
        opts, _ = getopt.getopt(argv, 'h', ['embedding=', 'build=', 'help'])
    except getopt.GetoptError as e:
        print(e)
        print(HELP_MSG)
        exit(2)

    # Logger
    logger = log.Log()
    logfile = logger.create('perplexity_study')

    # Default value
    embedding_type = None

    # Set values from command line
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(HELP_MSG)
            exit()
        elif opt == '--embedding':
            embedding_type = arg
            if embedding_type not in EMBEDDING_TYPES:
                print('{} is not a valid embedding type'.format(embedding_type))
                print(HELP_MSG)
                exit(2)
        elif opt == '--build':
            build_file = Path(arg)
            if build_file.is_file():
                x_labels = ['Actual score', 'Random words', 'Random sentences']
                with open(arg, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        #embedding_type,epochs,actual_score_train,actual_score_test,random_words_train,random_words_test,random_sentences_train,random_sentences_test
                        embedding_type, epochs, actual_scores, random_words_scores, random_sentences_scores = line.split(';')
                        actual_score_train, actual_score_test = actual_scores.split(',')
                        random_words_train, random_words_test = random_words_scores.split(',')
                        random_sentences_train, random_sentences_test = random_sentences_scores.split(',')
                        label_train = '{}_{}_train'.format(embedding_type, epochs)
                        label_test = '{}_{}_test'.format(embedding_type, epochs)
                        plt.plot(x_labels, [actual_score_train, random_words_train, random_sentences_train], label=label_train)
                        plt.plot(x_labels, [actual_score_test, random_words_test, random_sentences_test], label=label_test)
                plt.legend()
                plt.xlabel('Score type')
                plt.xlabel('Perplexity')
                plt.show()
                exit(0)
            else:
                print('{} is not a file'.format(arg))
                exit(1)

    print('Embedding type = {}'.format(embedding_type))
    logger.info(logfile, 'Embedding type = {}'.format(embedding_type))
    # Get directory name for our embedding type
    obj_dir = OBJ_DIR
    if embedding_type is not None:
        obj_dir += '_{}/'.format(embedding_type)
    else:
        obj_dir += '/'

    # Title of book that was trained on
    book_title = '1_sorcerers_stone'

    # Get train and test pairs
    train_pairs, test_pairs = storygen.get_pairs(book_title, 0.8, pre_parsed=False)
        
	# Set MAX_LENGTH:	
    MAX_LENGTH = max(
        max(map(len, [sentence.split() for pair in train_pairs for sentence in pair])),
        max(map(len, [sentence.split() for pair in test_pairs for sentence in pair])))
    MAX_LENGTH += 1 # for <EOL> token

    # Create book of vocab indexes
    book = storygen.get_book(book_title, train_pairs, test_pairs)

    # Find the largest checkpoint (code lifted from seq2seq.train_iters)
    encoders = set()
    decoders = set()
    for filename in os.listdir(obj_dir):
        r_enc = re.search(seq2seq.CHECKPOINT_FORMAT.format('encoder', storygen.EMBEDDING_SIZE, storygen.HIDDEN_SIZE, MAX_LENGTH), filename)
        if r_enc:
            encoders.add(int(r_enc.group(1)))
        else:
            r_dec = re.search(seq2seq.CHECKPOINT_FORMAT.format('decoder', storygen.EMBEDDING_SIZE, storygen.HIDDEN_SIZE, MAX_LENGTH), filename)
            if r_dec:
                decoders.add(int(r_dec.group(1)))

    # A checkpoint needs a valid encoder and decoder 
    checkpoints = encoders.intersection(decoders)
    
    if len(checkpoints) > 0:
        print('Checkpoints found at: {}'.format(checkpoints))
        logger.debug(logfile, 'Checkpoints found at: {}'.format(checkpoints))
    else:
        print('No checkpoints found for {}'.format(embedding_type))
        logger.error(logfile, 'No checkpoints found for {}'.format(embedding_type))
        exit(1)

    epoch_size = max(checkpoints)
    print('Epoch size used for study: {}'.format(epoch_size))
    logger.info(logfile, 'Epoch size used for study: {}'.format(epoch_size))

    # Get encoder/decoder file formats
    encoder_filename = seq2seq.ENCODER_FILE_FORMAT.format(obj_dir, epoch_size, storygen.EMBEDDING_SIZE, storygen.HIDDEN_SIZE, MAX_LENGTH)
    decoder_filename = seq2seq.DECODER_FILE_FORMAT.format(obj_dir, epoch_size, storygen.EMBEDDING_SIZE, storygen.HIDDEN_SIZE, MAX_LENGTH)

    network = seq2seq.Seq2Seq(book, MAX_LENGTH, storygen.HIDDEN_SIZE, storygen.EMBEDDING_SIZE, storygen.DEVICE)
    if not network.loadFromFiles(encoder_filename, decoder_filename):
        print('Error loading encoder at {}\nError loading decoder at {}'.format(encoder_filename, decoder_filename))
        logger.error(logfile, 'Error loading encoder at {}\nError loading decoder at {}'.format(encoder_filename, decoder_filename))
        exit(1)

    # Actual book sentences
    actual_sentences_score_train = 0.0
    actual_sentences_score_test = 0.0

    # Random words
    random_words_score_train = 0.0
    random_words_score_test = 0.0
    words = list(book.word2index)
    
    # Random sentences
    random_sentences_score_train = 0.0
    random_sentences_score_test = 0.0
    
    # Calculate scores on training data
    print('Calculating scores on training data...')
    actual_sentences_score_train, random_words_score_train, random_sentences_score_train = calculate_perplexities(network, words, train_pairs)

    print('Calculating scores on test data...')
    actual_sentences_score_test, random_words_score_test, random_sentences_score_test = calculate_perplexities(network, words, test_pairs)
    
    print('Actual sentences score (train): {:.4f}'.format(actual_sentences_score_train))
    print('Actual sentences score (test): {:.4f}'.format(actual_sentences_score_test))
    logger.info(logfile, 'Actual sentences score (train): {:.4f}'.format(actual_sentences_score_train))
    logger.info(logfile, 'Actual sentences score (test): {:.4f}'.format(actual_sentences_score_test))

    print('Random words score (train): {:.4f}'.format(random_words_score_train))
    print('Random words score (test): {:.4f}'.format(random_words_score_test))
    logger.info(logfile, 'Random words score (train): {:.4f}'.format(random_words_score_train))
    logger.info(logfile, 'Random words score (test): {:.4f}'.format(random_words_score_test))

    print('Random sentences score (train): {:.4f}'.format(random_sentences_score_train))
    print('Random sentences score (test): {:.4f}'.format(random_sentences_score_test))
    logger.info(logfile, 'Random sentences score (train): {:.4f}'.format(random_sentences_score_train))
    logger.info(logfile, 'Random sentences score (test): {:.4f}'.format(random_sentences_score_test))

    # Sentence segments (n-grams)
    sentences = storygen.get_sentences(book_title)
    n_gram_lengths = [1,2,3,4,6,8,12]
    for n_gram in n_gram_lengths:
        n_grams = []
        print('Processing n_gram: {}'.format(n_gram))
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= n_gram:
                n_grams += [words[i:i+n_gram] for i in range(len(words)-n_gram+1)]

        n_gram_score = 0.0
        incorrect_lengths = 0
        i = 0
        percentages = {0}
        for sentence in sentences:
            n_gram_sentence = sentence.split()
            if len(n_gram_sentence) >= n_gram:
                for i in range(int(len(n_gram_sentence)/n_gram)):
                    n_gram_sentence[n_gram*i:n_gram*(i+1)] = random.choice(n_grams)
                    perplexity = network._evaluate_specified(sentence, ' '.join(n_gram_sentence))
                    n_gram_score += perplexity
            else:
                incorrect_lengths += 1

            i += 1
            percentage = math.floor(100 * (float(i) / len(sentences)))
            if percentage % 5 == 0 and percentage not in percentages:
                print('{}% complete.'.format(percentage))
                percentages.add(percentage)
        n_gram_score /= (len(sentences) - incorrect_lengths)
        print('{}-gram score: {:.4f}\n{:.2f%}% of sentences not included.'.format(n_gram, n_gram_score, float(incorrect_lengths)/len(sentences) * 100))
        logger.info(logfile, '{}-gram score: {:.4f}\n{:.2f%}% of sentences not included.'.format(n_gram, n_gram_score, float(incorrect_lengths)/len(sentences) * 100))
    
if __name__ == '__main__':
    main(sys.argv[1:])
