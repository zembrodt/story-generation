# translation_study.py
import random
import torch

import storygen.seq2seq as seq2seq
import story_generation as storygen
import pymeteor.pymeteor as pymeteor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Code lifted from story_generation.py's main() function
    book_title = '1_sorcerers_stone'

    sentences = storygen.getSentencesFromBook(book_title)
    train_pairs, test_pairs = storygen.getPairs(sentences)
    input_book, output_book = storygen.getBooks(book_title, train_pairs, test_pairs)
    
    epoch_size = 1
    
    encoder_filename = storygen.ENCODER_FILE_FORMAT % epoch_size
    decoder_filename = storygen.DECODER_FILE_FORMAT % epoch_size

    EPOCHS = 100
    
    network = seq2seq.Seq2Seq(train_pairs, test_pairs, storygen.MAX_LENGTH, storygen.HIDDEN_SIZE, DEVICE)
    if not network.loadFromFiles(encoder_filename, decoder_filename):
        network.train_model(train_pairs, EPOCHS)
        network.saveToFiles(encoder_filename, decoder_filename)
    # end story_generation.py code

    pairs = train_pairs + test_pairs

    words = set()
    for sentence in sentences:
        for word in sentence.split():
            words.add(word)
    
    # BLEU and METEOR scores for all predicted sentences
    bleu_predicted_score = 0.0
    meteor_predicted_score = 0.0
    # BLEU and METEOR scores for random sentences
    bleu_random_score = 0.0
    meteor_random_score = 0.0
    for pair in pairs:
        output_words, attentions = network.evaluate(pair[0])
        output_sentence = ' '.join(output_words)
        
        bleu_predicted_score += seq2seq.calculateBleu(output_sentence, pair[1])
        # getting errors if the two strings have 0 matching words...
        meteor_predicted_score += pymeteor.meteor(output_sentence, pair[1])

        random_sentence = ' '.join([random.sample(words, 1)[0] for i in range(len(output_words))])

        bleu_random_score += seq2seq.calculateBleu(random_sentence, pair[1])
        meteor_random_score += pymeteor.meteor(random_sentence, pair[1])
        
    bleu_predicted_score /= len(pairs)
    meteor_predicted_score /= len(pairs)
    print('BLEU predicted score: %.4f\nMETEOR predicted score: %.4f'%(bleu_predicted_score, meteor_predicted_score))
    bleu_random_score /= len(pairs)
    meteor_random_score /= len(pairs)
    print('BLEU random score: %.4f\nMETEOR random score: %.4f'%(bleu_random_score, meteor_random_score))


if __name__ == '__main__':
    main()
