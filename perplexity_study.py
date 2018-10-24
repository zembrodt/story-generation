# perplexity_study.py
import random

import story_generation as storygen
import storygen.seq2seq as seq2seq

# Study on the perplexity module using the provided text
def main():
    book_title = '1_sorcerers_stone'
    sentences = storygen.get_sentences(book_title)
    train_pairs, test_pairs = storygen.get_pairs(book_title, 0.8)
    pairs = train_pairs + test_pairs
    input_book, output_book = storygen.get_books(book_title, train_pairs, test_pairs)

    ## Code lifted from story_generation.py's main method ##
    epoch_size = 25
    encoder_filename = storygen.ENCODER_FILE_FORMAT % epoch_size
    decoder_filename = storygen.DECODER_FILE_FORMAT % epoch_size

    network = seq2seq.Seq2Seq(input_book, output_book, storygen.MAX_LENGTH, storygen.HIDDEN_SIZE, storygen.DEVICE)
    if not network.loadFromFiles(encoder_filename, decoder_filename):
        network.trainIters(train_pairs, epoch_size)
        network.saveToFiles(encoder_filename, decoder_filename)

    # Actual book sentences
    actual_sentences_score = 0.0

    # Random words
    random_words_score = 0.0
    words = list(input_book.word2index)
    
    for pair in pairs:
        #_,_, perplexity = network._evaluate_specified(sentence, sentence)
        _,_, perplexity = network._evaluate_specified(pair[0], pair[1])
        actual_sentences_score += perplexity

        random_sentence = ' '.join([random.choice(words) for i in range(len(pair[1].split()))])
        _,_, perplexity = network._evaluate_specified(pair[0], random_sentence)
        random_words_score += perplexity
    
    actual_sentences_score /= len(sentences)
    print('Actual sentences score: %.4f'%actual_sentences_score)

    random_words_score /= len(sentences)
    print('Random words score: %.4f'%random_words_score)
    
    # Sentence segments (n-grams)
    n_gram_lengths = [1,2,3,4,6,8,12]
    for n_gram in n_gram_lengths:
        n_grams = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= n_gram:
                n_grams += [words[i:i+n_gram] for i in range(len(words)-n_gram+1)]

        n_gram_score = 0.0
        incorrect_lengths = 0
        for sentence in sentences:
            n_gram_sentence = sentence.split()
            if len(n_gram_sentence) >= n_gram:
                for i in range(int(len(n_gram_sentence)/n_gram)):
                    n_gram_sentence[n_gram*i:n_gram*(i+1)] = random.choice(n_grams)
                    _,_, perplexity = network._evaluate_specified(sentence, ' '.join(n_gram_sentence))
                    n_gram_score += perplexity
            else:
                incorrect_lengths += 1
        n_gram_score /= (len(sentences) - incorrect_lengths)
        print('%d-gram score: %.4f\n%.2f%% of sentences not included.'%(n_gram, n_gram_score, float(incorrect_lengths)/len(sentences) * 100))
    
if __name__ == '__main__':
    main()
