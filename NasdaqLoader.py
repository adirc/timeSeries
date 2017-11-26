import csv
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np


def read_nasdaq_csv_file(path):
    data = []
    with open(path, 'r') as csvfile:
        csv_data = csv.reader(csvfile)
        next(csv_data)
        for row in csv_data:
            data.append([float(i) for i in row])
        data = np.array(data)
    return data[:,:-1], data[:,-1]





### data is list of list. each list is a row in nasdaq csv data
class NasdaqDataset(Dataset):

    def collate_fn(self,batch):

        if (self.binaryLabel):
            return [b[0] for b in batch], Variable(torch.LongTensor(np.array([b[1] for b in batch], dtype=np.long)), requires_grad=False)
        else:
            return [b[0] for b in batch], Variable(torch.FloatTensor([b[1] for b in batch]), requires_grad=False)




    def normalize_data(self,mat):

        if (self.scaling):
            # axis=0 - over coloumns
            maxes = np.max(mat, axis=0)
            mins = np.min(mat, axis=0)
            tmp = (mat - mins) / (maxes - mins)
            return tmp,maxes,mins
        else:
            # axis=0 - over coloumns
            means = np.mean(mat, axis=0)
            stds = np.std(mat, axis=0)
            tmp = (mat - means) / (stds)
            return tmp,means,stds


    def getNormalizeFactors(self):
        if (self.scaling):
            return self.maxes,self.mins
        else:
            return self.means,self.stds

    def unNormalizedYs(self, x):
        if (not self.normalize_ys):
            return x

        if (self.scaling):
            # first var is maxes. second var is mins
            return x*(self.y_max - self.y_min)  + self.y_min
        else:
            return (x*self.y_std) + self.y_mean


    def __init__(self, root, history, useLabelAsFeatures,normalization = False, normalize_ys = False, convertToBinaryLabel=False,scaling = False):
        self.root = root
        self.data,self.y_s = read_nasdaq_csv_file(root)
        self.history = history
        self.normalization = normalization
        self.normalize_ys = normalize_ys
        self.scaling = scaling

        self.binaryLabel = convertToBinaryLabel

        if (useLabelAsFeatures):
            self.data = np.c_[self.data, self.y_s]

        if (normalization):
            if (scaling):
                self.data, self.maxes, self.mins = self.normalize_data(self.data)
            else:
                self.data, self.means,self.stds = self.normalize_data(self.data)

        if (normalize_ys):
            if (scaling):
                self.y_s, self.y_max, self.y_min = self.normalize_data(self.y_s)
            else:
                self.y_s, self.y_mean, self.y_std = self.normalize_data(self.y_s)

        if convertToBinaryLabel:
            self.y_s = (self.y_s[1:] - self.y_s[:-1]  > 0).astype(int)
            self.data = self.data[:-1,:] # match data and labels sizes

    def __getitem__(self, index):

        # label is the last coloumn
        #un comment below for not including last coloumn as feature
        #return [self.data[i][:-1] for i in range(index,index + self.history)] , self.data[index + self.history][-1]
        #return [self.data[i,:] for i in range(index,index + self.history)] , self.data[index + self.history,:][-1]

        X = self.data[index:(index + self.history), :]
        return  X  , self.y_s[index + self.history]

    def __len__(self):
        return self.data.shape[0] - self.history - 1

    def get_num_of_features(self):
        return self.data.shape[1]