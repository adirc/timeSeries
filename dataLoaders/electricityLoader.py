import csv
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
import numpy as np



def read_electricity_csv_file(path):
    data = []
    with open(path, 'r') as csvfile:
        csv_data = csv.reader(csvfile,delimiter=';')
        next(csv_data)
        for row in csv_data:
            data.append([float(row[i].replace(',', '.')) for i in range(1,len(row))]) #firs col is date
        data = np.array(data)
    return data


### data is list of list. each list is a row in nasdaq csv data
class ElectricityDataset(object):


    def __init__(self, root, history,batch_size):
        self.root = root
        self.data = read_electricity_csv_file(root)
        self.history = history
        self.index = 0
        self.batch_number = 0
        self.batch_size = batch_size


    def get_num_batches_in_epoch(self):
        return self.data.shape[0] - self.history

    def get_next_batch(self):

        # data_to_return = (self.data[self.index: (self.index + self.history), 0:self.batch_size],
        #                   self.data[(self.index + self.history),0:self.batch_size])


        # TODO: add zero padding for first batches.
        # TODO: handle case when "finish data"

        series_of_batch = []
        y_of_batch = []
        for i in range(self.batch_size):
            series_of_batch.append(self.data[self.index: (self.index + self.history),i])
        #y_of_batch.append(self.data[(self.index + self.history),i])
        y_of_batch = torch.FloatTensor(self.data[(self.index + self.history),0:self.batch_size])

        #is_last_batch = self.data.shape[0] - self.index < self.history + 1
        is_last_batch =  self.index > 50000
        if is_last_batch:
            self.batch_number = 0
            self.index = 0
        else:
            self.index += 1
            self.batch_number += 1

        return series_of_batch,y_of_batch,self.batch_number,is_last_batch