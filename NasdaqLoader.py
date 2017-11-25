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
    return data, data[:,-1]







### data is list of list. each list is a row in nasdaq csv data
class NasdaqDataset(Dataset):

    def collate_fn(self,batch):

        if (self.binaryLabel):
            return [b[0] for b in batch], Variable(torch.LongTensor(np.array([b[1] for b in batch], dtype=np.long)), requires_grad=False)
        else:
            return [b[0] for b in batch], Variable(torch.FloatTensor([b[1] for b in batch]), requires_grad=False)




    def normalize_data(self,mat):

        if (self.scalingNorm):
            # axis=0 - over coloumns
            self.maxes = np.max(mat, axis=0)
            self.mins = np.min(mat, axis=0)
            tmp = (mat - self.mins) / (self.maxes - self.mins)
        else:
            # axis=0 - over coloumns
            self.means = np.mean(mat, axis=0)
            self.stds = np.std(mat, axis=0)
            tmp = (mat - self.means) / (self.stds)
        return tmp

    def getNormalizeFactors(self):
        if (self.scalingNorm):
            return self.maxes,self.mins
        else:
            return self.means,self.stds

    def unNormalizedYs(self, x):
        if (not self.normalize_ys):
            return x

        if (self.scalingNorm):
            return x*(self.maxes[-1] - self.mins[-1])  + self.mins[-1]
        else:
            return (x*self.stds[-1]) + self.means[-1]

    def __init__(self, root,history,normalization = False,normalize_ys = False,scalingNorm = True,binaryLabel=False):
        self.root = root
        self.data,self.y_s = read_nasdaq_csv_file(root)
        self.history = history
        self.normalization = normalization
        self.normalize_ys = normalize_ys
        self.scalingNorm = scalingNorm
        self.binaryLabel = binaryLabel

        if (normalization):
            self.data = self.normalize_data(self.data)

        if (normalize_ys):
            #take previous ys from normalized data
            self.y_s = self.data[:,-1]

        self.data = np.c_[self.data, self.y_s]

        if binaryLabel:
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