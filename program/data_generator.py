'''
Data generation for input as sequences and output as boundaries
'''
import numpy as np
import keras
import h5py
import re
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Boundary prediction'

    def __init__(self, data_file, label_file, batch_size=32, dim=(1000, 5), n_channels=1,
                 shuffle=True, rnn_len=0, use_reverse=True, rnn_only=False,n_class=1, nbr_batch=0):
        '''
        shuffle: shuffle data for training, in validation and testing set it to False
        rnn_len: len of the RNN layer
        use_reverse: whether to consider the reverse strand (with label ends with _2), can be used during testing
        rnn_only: return data for RNN only
        '''
        'Initialization'

        self.data_file = data_file
        self.label_file = label_file

        self.dim = dim
        self.batch_size = batch_size

        self.rnn_len = rnn_len
        self.rnn_only = rnn_only

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.use_reverse = use_reverse

        self.ids_list = []

        self.ids_pos = [] # positive samples used when batches are sampled randomly
        self.ids_neg = [] # negative samples

        self.n_class = n_class # number of classes for classification

        self.nbr_batch = nbr_batch # to indicate number of batches to use, if used, epoch will be 1 and batches are sampled
        # self.indexes = []

        self.h5_data = None
        self.h5_label = None

        self.h5_data = h5py.File(self.data_file, 'r')

        if self.label_file:
            self.h5_label = h5py.File(self.label_file, 'r')

        for k, v in self.h5_data.items():

            lb = v.value[0]
            if not self.use_reverse:

                tmpk = k
                # if re.search('\.[0-9]+$', k):  # take care of the case .[0-9] is appended
                #     tmpk = re.sub('\.[0-9]+$', '', k)

                if tmpk.endswith("_1"):  # only take the forward strand
                    # if not use random batches
                    if self.nbr_batch == 0:
                        self.ids_list.append(k)
                    # use random batches
                    else:
                        if lb == 1:
                            self.ids_pos.append(k)
                        else:
                            self.ids_neg.append(k)

            else:
                if self.nbr_batch == 0:
                    self.ids_list.append(k)
                # use random batches
                else:
                    if lb == 1:
                        self.ids_pos.append(k)
                    else:
                        self.ids_neg.append(k)



        le = int(np.floor(len(self.ids_list) / self.batch_size))
        # taking care of the case when number of samples < batch_size: randomly add samples to make a full batch size
        if self.nbr_batch == 0 and le * self.batch_size < len(self.ids_list):

            shortage = (le + 1) * self.batch_size - len(self.ids_list)  # add this number of samples to

            i = 1
            while len(self.ids_list) * i < shortage:
                i += 1

            add = random.sample(self.ids_list * i, shortage)  # randomly sample
            self.ids_list = self.ids_list + add


        self.on_epoch_end()


    def __del__(self):
        if self.h5_data:
            self.h5_data.close()

        if self.h5_label:
            self.h5_label.close()


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.nbr_batch != 0:
            return self.nbr_batch

        le = int(np.floor(len(self.ids_list) / self.batch_size))

        return le

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.nbr_batch == 0:
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

            # Find list of IDs
            ids_list_temp = [self.ids_list[k] for k in indexes]

        else:
            half_batch = min(len(self.ids_pos), len(self.ids_neg), int(self.batch_size/2))
            ids_list_temp = random.sample(self.ids_pos, half_batch) + random.sample(self.ids_neg, half_batch)


        # Generate data
        if self.label_file:
            X, y = self.__data_generation__(ids_list_temp)
        else:
            X = self.__data_generation__(ids_list_temp)

        # if generate additional data for RNN to run
        # RNN doesn't take the full length for computational efficiency purpose
        # if return data for RNN, the two alleles are returned separately
        if self.rnn_len > 0:
            seq_len = self.dim[0]
            rnn_start = int(seq_len / 2 - self.rnn_len / 2)
            rnn_end = int(seq_len / 2 + self.rnn_len / 2)

            if self.label_file:

                if not self.rnn_only:
                    return ([X, X[:, rnn_start:rnn_end, :, :].reshape(self.batch_size, -1, 5)], y)
                else:
                    return (X[:, rnn_start:rnn_end, :, :].reshape(self.batch_size, -1, 5), y)

            else:
                if not self.rnn_only:
                    return ([X, X[:, rnn_start:rnn_end, :, :].reshape(self.batch_size, -1, 5)])
                else:
                    return (X[:, rnn_start:rnn_end, :, :].reshape(self.batch_size, -1, 5))
        else:
            if self.label_file:
                return (X, y)
            else:
                return (X)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.ids_list))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, ids_list_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels > 0:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim))

        if self.n_class > 1:
            y = np.empty((self.batch_size, self.n_class), dtype=int)
        else:
            y = []


        # Generate data
        for i, ID in enumerate(ids_list_temp):
            # Store sample
            dt = self.h5_data[ID][:]
            if self.n_channels > 0:
                X[i,] = dt.reshape(*self.dim, self.n_channels)
            else:
                X[i,] = dt.reshape(*self.dim)

            if self.label_file:
                lb = self.h5_label[ID]
                if self.n_class > 1:
                    y[i,] = lb[:]
                else:
                    #y.append(lb[()])
                    y.append(lb.value)

        if self.label_file and self.n_class == 1:
            y = np.stack(y)

        if self.label_file:
            return (X, y)
        else:
            return (X)


class DataGeneratorLoopSeq(keras.utils.Sequence):
    'Generates sequence data to predict loop'

    def __init__(self, data_file, label_file, batch_size=32, dim=(4000, 10),
                 n_channels=1, shuffle=True, rnn_len=0, use_reverse=True,
                 nbr_batch=0):
        """
        rnn_len: len of the RNN layer
        use_reverse: whether to consider the reverse strand (with label ends with _2), can be used during testing
        """
        'Initialization'

        self.data_file = data_file
        self.label_file = label_file

        self.dim = dim
        self.batch_size = batch_size

        self.rnn_len = rnn_len

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.use_reverse = use_reverse

        self.nbr_batch = nbr_batch

        self.ids_list = []

        self.ids_pos = []

        self.ids_neg = []

        # self.indexes = []

        h5_data = None
        try:
            h5_data = h5py.File(data_file, 'r')
            for k, v in h5_data.items():
                # lb = v.value[0]

                # to avoid warning  'dataset.value has been deprecated',
                # the code was modify according to the warning's suggestion.
                lb = v[()][0]

                if not self.use_reverse:
                    tmpk = k
                    if re.search('\.[0-9]+$', k):  # take care of the case .[0-9] is appended
                        tmpk = re.sub('\.[0-9]+$', '', k)
                        # print(tmpk)

                    # print(tmpk, tmpk.endswith("_1") )
                    if tmpk.endswith("_1"):  # only take the forward strand
                        if self.nbr_batch == 0:
                            self.ids_list.append(k)
                        else:
                            if lb == 0:
                                self.ids_pos.append(k)
                            else:
                                self.ids_neg.append(k)

                else:
                    #self.ids_list.append(k)
                    if self.nbr_batch == 0:
                        self.ids_list.append(k)
                    else:
                        if lb == 0:
                            self.ids_pos.append(k)
                        else:
                            self.ids_neg.append(k)

            le = int(np.floor(len(self.ids_list) / self.batch_size))
            # taking care of the case when number of samples < batch_size: randomly add samples to make a full batch size
            if self.nbr_batch == 0 and le * self.batch_size < len(self.ids_list):

                shortage = (le + 1) * self.batch_size - len(self.ids_list)  # add this number of samples to
                i = 1
                while len(self.ids_list) * i < shortage:
                    i += 1

                add = random.sample(self.ids_list * i, shortage)  # randomly sample

                self.ids_list = self.ids_list + add

        finally:
            if h5_data is not None:
                h5_data.close()

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.nbr_batch > 0:
            return self.nbr_batch

        le = int(np.floor(len(self.ids_list) / self.batch_size))

        return le

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.nbr_batch == 0:
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size: min((index + 1) * self.batch_size, len(self.ids_list))]

            # Find list of IDs
            ids_list_temp = [self.ids_list[k] for k in indexes]

        else:
            half_batch = min(len(self.ids_pos), len(self.ids_neg), int(self.batch_size/2))
            ids_list_temp = random.sample(self.ids_pos, half_batch) + random.sample(self.ids_neg, half_batch)


        # Generate data
        X1, X2, y = self.__data_generation__(ids_list_temp)

        # if generate additional data for RNN to run
        # RNN doesn't take the full length for computational efficiency purpose
        # if return data for RNN, the two alleles are returned separately
        if self.rnn_len > 0:
            seq_len = self.dim[0]
            gru_start = int(seq_len / 2 - self.rnn_len / 2)
            gru_end = int(seq_len / 2 + self.rnn_len / 2)

            return ([X1, X1[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5),
                     X2, X2[:, gru_start:gru_end, 0:5, :].reshape(self.batch_size, -1, 5)],
                    y)
        else:
            return ([X1, X2], y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.ids_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, ids_list_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels > 0:
            X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X1 = np.empty((self.batch_size, *self.dim))
            X2 = np.empty((self.batch_size, *self.dim))

        # y = np.empty((self.batch_size), dtype=int)
        y = []

        h5_data = None
        h5_label = None

        try:
            h5_data = h5py.File(self.data_file, 'r')

            if self.label_file:
                h5_label = h5py.File(self.label_file, 'r')

            # Generate data
            for i, ID in enumerate(ids_list_temp):

                dt = h5_data[ID][:]

                # data shape is (1000, 10), X1 will be (1000,5), X2: (1000,5)
                half_nbr_seq = int(dt.shape[1] / 2)

                dt1 = dt[:, :half_nbr_seq]
                dt2 = dt[:, half_nbr_seq:]

                if self.n_channels > 0:
                    X1[i,] = dt1.reshape(*self.dim, self.n_channels)
                    X2[i,] = dt2.reshape(*self.dim, self.n_channels)
                else:
                    X1[i,] = dt1.reshape(*self.dim)
                    X2[i,] = dt2.reshape(*self.dim)

                if h5_label:
                    lb = h5_label[ID]
                    y.append(lb.value)
            if self.label_file:
                y = np.stack(y)

        finally:
            if h5_data is not None:
                h5_data.close()
            if h5_label is not None:
                h5_label.close()

        return (X1, X2, y)


