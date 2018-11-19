# glove.py
import numpy as np
import os
import pathlib
import pickle
import timeit

DIMENSION_SIZES  = [50, 100, 200, 300]
DATA_FILE_FORMAT = 'data/glove.6B/glove.6B.{}d.txt'
OBJ_FILE_FORMAT  = 'obj_glove/glove.6B/globe.6B.{}d.pickle'
# Custom embeddings
DATA_SG_FILE = 'data/vectors-sg.txt'
OBJ_SG_FILE = 'obj_sg/vectors-sg.pickle'
DATA_CBOW_FILE = 'data/vectors-cbow.txt'
OBJ_CBOW_FILE = 'obj_cbow/vectors-cbow.pickle'

def generate_glove(dim_size=50, local=False, embedding_type='glove'):
    if embedding_type == 'glove':
        if dim_size not in DIMENSION_SIZES:
            print('Incorrect dimension size given, please use one of the following sizes: {}'.format(DIMENSION_SIZES))
            return None
        file_format = DATA_FILE_FORMAT
        obj_format = OBJ_FILE_FORMAT
        if local:
            file_format = '../' + file_format
            obj_format = '../' + obj_format
        # Check if an obj file exists
        obj_file = pathlib.Path(obj_format.format(dim_size))
        if obj_file.exists():
            # Load from file with pickle
            with open(obj_format.format(dim_size), 'rb') as f:
                words2vec = pickle.load(f)
            return words2vec
        else:
            os.makedirs(os.path.dirname(obj_format.format(dim_size)), exist_ok=True)
            words2vec = {}
            with open(file_format.format(dim_size), 'rb') as f:
                for line in f:
                    line = line.decode().split()
                    word = line[0]
                    vector = np.array(line[1:]).astype(np.float)
                    words2vec[word] = vector
            with open(obj_format.format(dim_size), 'wb') as f:
                pickle.dump(words2vec, f)
            return words2vec
    else:
        data_custom_file = None
        obj_custom_file = None
        if embedding_type == 'sg':
            data_custom_file = DATA_SG_FILE
            obj_custom_file = OBJ_SG_FILE
        elif embedding_type == 'cbow':
            data_custom_file = DATA_CBOW_FILE
            obj_custom_file = OBJ_CBOW_FILE
        else:
            print('Incorrect embedding type given! Please choose one of ["glove", "sg", "cbow"]')
            exit()
        # Check if an obj file exists
        obj_file = pathlib.Path(obj_custom_file)
        if obj_file.exists():
            # Load from file with pickle
            with open(obj_custom_file, 'rb') as f:
                words2vec = pickle.load(f)
            return words2vec
        else:
            # Create the new word2vec and save it from the custom embeddings
            os.makedirs(os.path.dirname(obj_custom_file), exist_ok=True)
            words2vec = {}
            with open(data_custom_file, 'rb') as f:
                for line in f:
                    line = line.decode().split()
                    word = line[0]
                    vector = np.array(line[1:]).astype(np.float)
                    words2vec[word] = vector
            with open(obj_custom_file, 'wb') as f:
                pickle.dump(words2vec, f)
            return words2vec

def main():
    start = timeit.default_timer()
    words2vec = generate_glove(dim_size=300, local=True)
    stop = timeit.default_timer()
    print(words2vec['test'])
    print('Took: {} seconds.'.format(stop - start))

if __name__ == '__main__':
    main()