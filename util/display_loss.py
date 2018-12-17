# display_loss.py

# Displays the loss values of select word embeddings as a graph, with or without validation values

import matplotlib.pyplot as plt
import getopt
import sys

FILE_FORMAT = '{}_{}.dat'
LOSS = 'loss'
VALIDATION = 'validation'
TYPES = ['none', 'glove', 'sg', 'cbow']

def main(argv):
    opts, _ = getopt.getopt(argv, 'vnscg', ['validation']+TYPES)
    display_validation = False
    # Check if any non-validation options were selected
    types_selected = []
    for opt, _ in opts:
        if '-v' == opt or '--validation' == opt:
            display_validation = True
        else:
            for t in TYPES:
                if '-'+t[0] == opt or '--'+t == opt:
                    types_selected.append(t)
                    break

    # Load loss.dat files
    for t in TYPES:
        if len(types_selected) == 0 or t in types_selected:
            with open(FILE_FORMAT.format(LOSS, t), 'r') as f:
                loss_values = [item.split(',') for item in f.read().strip().split('\t')]
                label = '{} training'.format(t)
                plt.plot([int(item[0]) for item in loss_values], [float(item[1]) for item in loss_values], label=label)
            if display_validation:
                with open(FILE_FORMAT.format(VALIDATION, t), 'r') as f:
                    validation_values = [item.split(',') for item in f.read().strip().split('\t')]
                    label = '{} validation'.format(t)
                    plt.plot([int(item[0]) for item in validation_values], [float(item[1]) for item in validation_values], label=label)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
    if display_validation:
        plt.show()
        plt.savefig('loss_figure_with_validation.png') # Save plot to log folder
    else:
        plt.show()
        plt.savefig('loss_figure.png') # Save plot to log folder
        
if __name__ == '__main__':
        main(sys.argv[1:])
