# -*- coding: utf-8 -*-

import numpy as np
import csv
from sklearn.preprocessing import normalize as norm


def window_data( data, window_size ):
    data = np.array( data )
    w_data = [ data[ i*window_size : (i+1)*window_size ]
               for i in range( len(data) // window_size ) ]
    return w_data


def load_multimodal( split_num,
                     num_activities=20,
                     num_seqs=10,
                     window_size = 10 ):
    train_data   = list()
    test_data    = list()
    train_labels = list()
    test_labels  = list()
    for act_id in range( num_activities ):
        for seq_id in range( num_seqs ):
            act_num = '{:02d}'.format( act_id+1 )
            seq_num = '{:02d}'.format( seq_id+1 )
            with open('data/act' + act_num + 'seq' + seq_num + '.csv', 'r') as f:
                inp = list( csv.reader(f) )
                inps = window_data( inp, window_size )
                # Provide one-hot label
                label = np.zeros( [len(inps), num_activities] )
                label[ :, act_id ] = 1
                label = label.tolist()
                # Split data into train and test sets
                if seq_id+1 == split_num:
                    test_data   += inps
                    test_labels += label
                else:
                    train_data   += inps
                    train_labels += label
    train_data = [ norm(x) for x in train_data ]
    test_data  = [ norm(x) for x in test_data ]
    return train_data, train_labels, test_data, test_labels

