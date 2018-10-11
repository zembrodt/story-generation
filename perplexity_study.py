# perplexity_study.py
import random

import pyplexity.pyplexity as pyplexity
import story_generation as storygen

# Study on the perplexity module using the provided text
def main():
    book_title = '1_sorcerers_stone'

    sentences = storygen.getSentencesFromBook(book_title)
    
    perplexity_model = pyplexity.PerplexityModel(sentences)

    # Actual book sentences
    actual_sentences_score = 0.0
    # Random words
    words = list(perplexity_model.model.keys())
    words.remove(pyplexity.STOP)
    random_words_score = 0.0
    
    for sentence in sentences:
        actual_sentences_score += perplexity_model.perplexity(sentence)
        
        random_sentence = ' '.join([random.choice(words) for i in range(len(sentence.split()))])
        random_words_score += perplexity_model.perplexity(random_sentence)
        
    actual_sentences_score /= len(sentences)
    print('Actual sentences score: %.4f'%actual_sentences_score)

    random_words_score /= len(sentences)
    print('Random words score: %.4f'%random_words_score)

    # Sentence segments (n-grams)
    n_gram_lengths = [1,2,3,4,6]
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
                    n_gram_score += perplexity_model.perplexity(' '.join(n_gram_sentence))
            else:
                incorrect_lengths += 1
        n_gram_score /= (len(sentences) - incorrect_lengths)
        print('%d-gram score: %.4f\n%d sentences were found with an incorrect length.'%(n_gram, n_gram_score, incorrect_lengths))

if __name__ == '__main__':
    main()
