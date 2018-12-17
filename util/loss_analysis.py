# loss_analysis.py

# Allows the user to view min/max loss values of a given file, or look at the loss value at a specific epoch

import getopt, sys
from operator import itemgetter

def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'h', ['min=', 'at='])
    except getopt.GetoptError as e:
        print(e)
        print('--min <value>: the threshold for the epoch to be above')
        print('--at <epoch>: get the loss value at the specified epoch')

    with open('loss.dat', 'r') as f:
        lines = f.readlines()
    
    epoch_threshold = 0
    epoch_to_find = None
    for opt, val in opts:
        if opt == '--min':
            epoch_threshold = int(val)
        elif opt == '--at':
            epoch_to_find = int(val)

    loss_vals = []
    for line in lines:
        loss_vals += [val.split(',') for val in line.strip().split('\t')]
    loss_vals = [(int(val[0]), float(val[1])) for val in loss_vals if int(val[0]) >= epoch_threshold]

    if epoch_to_find is None:
        max_val = max(loss_vals, key=itemgetter(1))
        min_val = min(loss_vals, key=itemgetter(1))

        print('Total loss values:  {}'.format(len(loss_vals)))
        print('Highest loss value: {} at epoch {}'.format(max_val[1], max_val[0]))
        print('Lowest loss value:  {} at epoch {}'.format(min_val[1], min_val[0]))
    else:
        loss_vals = dict(loss_vals)
        print('Loss value at epoch {}: {}'.format(epoch_to_find, loss_vals[epoch_to_find]))

if __name__ == '__main__':
        main(sys.argv[1:])